import os
import re
import csv
import math
import glob
import pickle
import hashlib
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pypdf  # noqa: F401  (kept for doc-type awareness)
import docx
import fitz  # PyMuPDF
import faiss

from sentence_transformers import SentenceTransformer

import config

# Codes such as E031, AL-2047, H123, 4021, servo 33 ...
CODE_RE = re.compile(r"\b(?:[A-Z]{1,3}[-]?\d{2,4})\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def extract_fault_codes(text: str) -> List[str]:
    """Pulls likely fault/error codes out of free text, normalized to UPPER."""
    if not text:
        return []
    tokens = CODE_RE.findall(text)
    # Drop pure date-like / oversized patterns that are unlikely to be alarms
    cleaned = []
    for t in tokens:
        alpha = sum(c.isalpha() for c in t)
        digits = sum(c.isdigit() for c in t)
        if digits >= 2 and 0 <= alpha <= 3:
            cleaned.append(t.replace("-", "").replace(" ", "").upper())
    return sorted(set(cleaned))


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class DocumentLoader:
    """Handles loading of text and images from PDF, DOCX, TXT, and CSV files."""

    @staticmethod
    def load_pdf(file_path: str, extract_images: bool = False) -> Dict[str, Any]:
        text = ""
        images = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text += page.get_text() + "\n"

                if extract_images:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        img_filename = f"{os.path.basename(file_path)}_p{page_num}_i{img_index}.{image_ext}"
                        images.append({
                            "name": img_filename,
                            "bytes": image_bytes,
                            "page": page_num,
                            "ext": image_ext,
                        })
            doc.close()
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

        return {"text": text, "images": images}

    @staticmethod
    def load_docx(file_path: str) -> str:
        text = ""
        try:
            doc_file = docx.Document(file_path)
            for para in doc_file.paragraphs:
                text += para.text + "\n"
            for table in doc_file.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    text += " | ".join(cells) + "\n"
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
        return text

    @staticmethod
    def load_txt(file_path: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, Exception):
                continue
        return ""

    @staticmethod
    def load_csv(file_path: str) -> str:
        """Loads CSV files, useful for fault codes."""
        text = ""
        try:
            with open(file_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    text += row_str + "\n"
            return text
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
            return ""

    @staticmethod
    def load_file(file_path: str, extract_images: bool = False) -> Optional[Dict[str, Any]]:
        """Loads a file and returns its content (text + images) and metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        images = []

        if ext == ".pdf":
            data = DocumentLoader.load_pdf(file_path, extract_images=extract_images)
            content = data["text"]
            images = data["images"]
        elif ext == ".docx":
            content = DocumentLoader.load_docx(file_path)
        elif ext in [".txt", ".log"]:
            content = DocumentLoader.load_txt(file_path)
        elif ext == ".csv":
            content = DocumentLoader.load_csv(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            images = [{
                "name": os.path.basename(file_path),
                "bytes": img_bytes,
                "page": 0,
                "ext": ext.replace(".", ""),
            }]
        else:
            print(f"Unsupported file type: {ext}")
            return None

        if not content.strip() and not images:
            return None

        return {"source": file_path, "content": content, "images": images}


class TextSplitter:
    """Splits text into chunks, respecting semantic boundaries."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    def split_text(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        chunks = []
        current_chunk = []
        current_word_count = 0

        for line in lines:
            line_words = line.split()
            if not line_words:
                continue

            if len(line_words) > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

                sentences = re.split(r"(?<=[.!?])\s+", line)
                for sent in sentences:
                    if sent.strip():
                        chunks.append(sent.strip())
                continue

            if current_word_count + len(line_words) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                while current_chunk and current_word_count > max(0, self.chunk_overlap):
                    first_line = current_chunk.pop(0)
                    current_word_count -= len(first_line.split())

            current_chunk.append(line)
            current_word_count += len(line_words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]


class VectorStore:
    """
    A FAISS-backed vector store with hybrid (vector + BM25) retrieval.

    * Vectors stored in a normalized IndexFlatIP.
    * A lightweight BM25 lexicon enables exact fault-code / jargon matching that
      pure embeddings frequently miss.
    * Query embeddings are cached to avoid re-encoding repeated questions on CPU.
    """

    def __init__(self, model_name: str = None, model=None):
        model_name = model_name or config.EMBEDDING_MODEL_PATH

        # 1. Device Detection (CPU-safe default that never tries CUDA unless asked)
        use_gpu = config.USE_GPU
        try:
            import torch
            self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
            torch.set_num_threads(config.resolve_threads(config.EMBED_THREADS))
        except ImportError:
            self.device = "cpu"

        if model:
            self.model = model
        else:
            print(f"Loading embedding model: {model_name} on {self.device}...")
            try:
                self.model = SentenceTransformer(
                    model_name,
                    device=self.device,
                    model_kwargs={"low_cpu_mem_usage": True},
                )
            except Exception as e:
                print(f"Warning: Failed to load with optimization: {e}")
                self.model = SentenceTransformer(model_name, device=self.device)

        # Determine embedding dimension from the model (fallback 384 for safety)
        try:
            dim = int(self.model.get_embedding_dimension())
        except Exception:
            try:
                dim = int(self.model.get_sentence_embedding_dimension())
            except Exception:
                try:
                    dim = int(self.model[1].get_sentence_embedding_dimension()) if hasattr(self.model[1], "get_sentence_embedding_dimension") else None
                except Exception:
                    dim = None
        self.embedding_dim = dim or 384

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

        # Per-id vectors (used for exact code-match rescoring)
        self._vectors: Dict[int, np.ndarray] = {}

        # BM25 lexicon
        self._lex_terms: Dict[int, Dict[str, int]] = {}
        self._lex_df: Counter = Counter()
        self._lex_total_len = 0

        # Fault-code lookup index  (normalized code -> list of ids)
        self._code_index: Dict[str, List[int]] = {}

        # Query embedding cache (space-bounded)
        self._query_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _cache_query(self, query: str, vec: np.ndarray):
        if len(self._query_cache) > 512:
            self._query_cache.clear()
        self._query_cache[query] = vec

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Memory-friendly batched encoding tuned for CPU machines."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        vectors = []
        batch = config.EMBED_BATCH
        for i in range(0, len(texts), batch):
            part = self.model.encode(
                texts[i:i + batch],
                batch_size=batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            part = np.asarray(part, dtype=np.float32)
            if part.ndim == 1:
                part = part.reshape(1, -1)
            vectors.append(part)
        return np.vstack(vectors) if vectors else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        if query in self._query_cache:
            return self._query_cache[query]
        vec = self._encode_texts([query])
        self._cache_query(query, vec)
        return vec

    def _index_lexicon(self, doc_id: int, text: str):
        terms = Counter(_tokenize(text))
        if not terms:
            return
        self._lex_terms[doc_id] = dict(terms)
        self._lex_total_len += sum(terms.values())
        self._lex_df.update(terms.keys())

        for code in extract_fault_codes(text):
            self._code_index.setdefault(code, []).append(doc_id)

    def _unindex_lexicon(self, doc_id: int):
        terms = self._lex_terms.pop(doc_id, None)
        if terms is None:
            return
        for term in terms:
            self._lex_df[term] -= 1
        self._lex_total_len -= sum(terms.values())
        for code, ids in list(self._code_index.items()):
            if doc_id in ids:
                ids.remove(doc_id)
                if not ids:
                    del self._code_index[code]

    def _bm25_scores(self, query_terms: List[str]) -> Dict[int, float]:
        if not self._lex_terms or not query_terms:
            return {}
        n = len(self._lex_terms)
        avgdl = self._lex_total_len / n if n else 1.0
        scores: Dict[int, float] = {}
        k1, b = 1.5, 0.75

        for q in set(query_terms):
            df = self._lex_df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf_map in self._lex_terms.items():
                tf = tf_map.get(q)
                if not tf:
                    continue
                dl = sum(tf_map.values())
                denom = tf + k1 * (1 - b + b * (dl / avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1)) / denom

        return scores

    def _vec_sim(self, doc_id: int, query_vec: np.ndarray) -> Optional[float]:
        v = self._vectors.get(doc_id)
        if v is None or len(v) == 0:
            return None
        return float(np.dot(query_vec[0], v))

    @staticmethod
    def _fuse_rrf(pools: List[List[int]], k: int = 60) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for pool in pools:
            for rank, doc_id in enumerate(pool):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores

    # ------------------------------------------------------------------ add
    def add_documents(self, documents: List[Dict[str, Any]], splitter: TextSplitter,
                      media_dir: str = None, extract_images: bool = False):
        media_dir = media_dir or config.MEDIA_DIR
        new_texts: List[str] = []
        new_metas: List[Dict[str, Any]] = []

        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        for doc in documents:
            file_path = doc.get("source", "")
            file_hash = VectorStore.get_file_hash(file_path) if file_path and os.path.exists(file_path) else None

            if any(meta.get("file_hash") == file_hash for meta in self.metadata.values() if file_hash):
                print(f"Skipping {file_path} - already in index (hash match).")
                continue

            saved_images = []
            for img_data in doc.get("images", []):
                img_path = os.path.join(media_dir, img_data["name"])
                if not os.path.exists(img_path):
                    with open(img_path, "wb") as f:
                        f.write(img_data["bytes"])
                saved_images.append({"name": img_data["name"], "path": img_path, "page": img_data["page"]})

            content = doc.get("content", "")
            if not content.strip() and not saved_images:
                continue

            chunks = splitter.split_text(content) if content.strip() else []
            for chunk in chunks:
                new_texts.append(chunk)
                new_metas.append({
                    "source": file_path,
                    "file_hash": file_hash,
                    "machine": doc.get("machine", "Unknown"),
                    "subsystem": doc.get("subsystem", "General"),
                    "chunk_text": chunk,
                    "images": saved_images,
                    "page": doc.get("page"),
                })

        if not new_texts:
            print("No new chunks to add.")
            return

        print(f"Embedding {len(new_texts)} chunks on {self.device}...")
        embeddings = self._encode_texts(new_texts)
        faiss.normalize_L2(embeddings)

        ids = np.arange(self.next_id, self.next_id + len(new_texts), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        for i, doc_id in enumerate(ids):
            did = int(doc_id)
            self.metadata[did] = new_metas[i]
            self._vectors[did] = embeddings[i]
            self._index_lexicon(did, new_metas[i]["chunk_text"])

        self.next_id += len(new_texts)
        print(f"Added {len(new_texts)} chunks. Total: {self.index.ntotal}")

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = None, filter_criteria: Dict[str, str] = None,
               threshold: float = None, hybrid: bool = None) -> List[Dict[str, Any]]:
        """Hybrid retrieval: exact fault-code hits + BM25 + FAISS, RRF-fused.

        `score` in each result remains the normalized vector cosine similarity so
        downstream thresholds (e.g. feedback memory) keep their meaning.
        """
        k = k or config.RETRIEVAL_K
        threshold = config.RETRIEVAL_THRESHOLD if threshold is None else threshold
        hybrid = config.USE_HYBRID if hybrid is None else hybrid

        if self.index.ntotal == 0:
            return []

        query_vec = self._encode_query(query)

        # Broader candidate pool so metadata filters rarely starve the result set.
        if filter_criteria:
            fetch_k = min(self.index.ntotal, max(k * 20, 64))
        else:
            fetch_k = min(self.index.ntotal, max(k * 4, 32))

        sims, idxs = self.index.search(query_vec, fetch_k)

        vec_candidates: Dict[int, float] = {}
        vec_ranked: List[int] = []
        for i, faiss_id in enumerate(idxs[0]):
            faiss_id = int(faiss_id)
            if faiss_id == -1:
                continue
            score = float(sims[0][i])
            if score < threshold:
                continue
            vec_candidates[faiss_id] = score
            vec_ranked.append(faiss_id)

        # Exact fault-code match -> highest priority, always surfaced.
        code_matches: List[int] = []
        codes_found = extract_fault_codes(query)
        for code in codes_found:
            for doc_id in self._code_index.get(code, []):
                if doc_id not in code_matches:
                    code_matches.append(doc_id)
                    if doc_id not in vec_candidates:
                        s = self._vec_sim(doc_id, query_vec)
                        if s is not None:
                            vec_candidates[doc_id] = s

        if not hybrid:
            ordered = vec_ranked + [d for d in code_matches if d not in vec_ranked]
        else:
            bm25 = self._bm25_scores(_tokenize(query))
            lex_ranked = [d for d, _ in sorted(bm25.items(), key=lambda x: x[1], reverse=True)[:fetch_k * 2]]
            fused = self._fuse_rrf([vec_ranked, lex_ranked, code_matches])
            ordered = sorted(fused, key=fused.get, reverse=True)

        results: List[Dict[str, Any]] = []
        for doc_id in ordered:
            meta = self.metadata.get(doc_id)
            if not meta:
                continue
            if filter_criteria:
                match = True
                for key, val in filter_criteria.items():
                    if val and val != "All" and meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            results.append({
                "score": float(vec_candidates.get(doc_id, 0.0)),
                "metadata": meta,
                "code_match": doc_id in code_matches,
            })
            if len(results) >= k:
                break

        # Fallback: if hybrid pool was entirely filtered, re-run plain filtered vector search.
        if len(results) < k:
            seen = {r["metadata"].get("chunk_text") + str(r["metadata"].get("source"))
                    for r in results}
            for doc_id in vec_ranked:
                meta = self.metadata.get(doc_id)
                if not meta:
                    continue
                key = meta.get("chunk_text", "") + str(meta.get("source"))
                if key in seen:
                    continue
                if filter_criteria:
                    if any(val and val != "All" and meta.get(key_) != val
                           for key_, val in filter_criteria.items()):
                        continue
                results.append({"score": vec_candidates[doc_id], "metadata": meta, "code_match": False})
                seen.add(key)
                if len(results) >= k:
                    break

        return results

    def quick_fault_lookup(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Direct table lookup for a fault code without any embedding cost."""
        codes = extract_fault_codes(query)
        out = []
        for code in codes:
            for doc_id in self._code_index.get(code, []):
                meta = self.metadata.get(doc_id)
                if meta:
                    out.append({"score": float(self._vec_sim(doc_id, self._encode_query(query)) or 0.0),
                                "metadata": meta, "code_match": True})
        return out[:k]

    # ------------------------------------------------------------------ admin
    def get_ingested_documents(self) -> List[str]:
        sources = set(meta.get("source") for meta in self.metadata.values() if meta.get("source"))
        return sorted(s for s in sources if s and s != "user_feedback")

    def delete_document(self, source_name: str):
        if self.index.ntotal == 0:
            return

        ids_to_remove = [doc_id for doc_id, meta in self.metadata.items() if meta.get("source") == source_name]
        if not ids_to_remove:
            print(f"No chunks found for {source_name}")
            return

        self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
        for doc_id in ids_to_remove:
            del self.metadata[doc_id]
            self._vectors.pop(doc_id, None)
            self._unindex_lexicon(doc_id)
        print(f"Deleted {source_name}. Remaining chunks: {self.index.ntotal}")

    def get_unique_metadata_values(self) -> Tuple[List[str], List[str]]:
        machines = set()
        subsystems = set()
        for meta in self.metadata.values():
            if meta.get("machine"):
                machines.add(meta["machine"])
            if meta.get("subsystem"):
                subsystems.add(meta["subsystem"])
        return sorted(machines), sorted(subsystems)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "chunks": self.index.ntotal,
            "dim": self.embedding_dim,
            "device": self.device,
            "documents": len(self.get_ingested_documents()),
            "fault_codes_indexed": len(self._code_index),
            "hybrid": config.USE_HYBRID,
        }

    # ------------------------------------------------------------------ persist
    def save(self, path: str):
        base = path[:-4] if path.endswith(".pkl") else path
        tmp_index = base + ".faiss.tmp"
        faiss.write_index(self.index, tmp_index)
        os.replace(tmp_index, base + ".faiss")

        tmp_meta = base + ".meta.tmp"
        with open(tmp_meta, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "next_id": self.next_id,
                "vectors": self._vectors,
            }, f)
        os.replace(tmp_meta, base + ".meta")

        print(f"Vector store saved to {base}.faiss and {base}.meta")

    def load(self, path: str):
        base = path[:-4] if path.endswith(".pkl") else path

        # Backward compatibility for old numpy pickles
        if os.path.exists(path) and not path.endswith(".faiss"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if "vectors" in data and isinstance(data.get("metadata"), list):
                    print("Migrating old vector store format to FAISS...")
                    vectors = data["vectors"]
                    if len(vectors) > 0:
                        vectors = np.asarray(vectors, dtype=np.float32)
                        faiss.normalize_L2(vectors)
                        ids = np.arange(0, len(vectors), dtype=np.int64)
                        self.index.add_with_ids(vectors, ids)
                        for i, meta in enumerate(data["metadata"]):
                            self.metadata[i] = meta
                            self._vectors[i] = vectors[i]
                            self._index_lexicon(i, meta.get("chunk_text", ""))
                        self.next_id = len(vectors)
                        self.save(path)
                    return
            except Exception as e:
                print(f"Migration error: {e}")

        if not os.path.exists(base + ".faiss") or not os.path.exists(base + ".meta"):
            print(f"No vector store found at {base}")
            return

        self.index = faiss.read_index(base + ".faiss")
        with open(base + ".meta", "rb") as f:
            data = pickle.load(f)
        self.metadata = data.get("metadata", {})
        self.next_id = data.get("next_id", (max(list(self.metadata.keys()) + [-1]) + 1))

        # Restore per-id vectors (saved or reconstructed from the flat index)
        saved_vectors = data.get("vectors") or {}
        self._vectors = {}
        for did in self.metadata:
            if did in saved_vectors:
                self._vectors[did] = np.asarray(saved_vectors[did], dtype=np.float32)
            else:
                try:
                    self._vectors[did] = self.index.reconstruct(did)
                except Exception:
                    pass

        # Rebuild lexical + code index from stored text (cheap, deterministic)
        self._lex_terms = {}
        self._lex_df = Counter()
        self._lex_total_len = 0
        self._code_index = {}
        for did, meta in self.metadata.items():
            self._index_lexicon(did, meta.get("chunk_text", ""))

        print(f"Vector store loaded from {base} with {self.index.ntotal} chunks.")