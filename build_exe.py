import PyInstaller.__main__
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure we are in the root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEP = os.pathsep if os.name == "nt" else ":"


def _make_icon():
    """Generates a professional EquipAI .ico if none exists (uses Pillow)."""
    ico_path = os.path.join(BASE_DIR, "media", "equipai.ico")
    if os.path.exists(ico_path):
        return ico_path
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = 256
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Dark rounded square with cyan gradient-ish rings
        draw.rounded_rectangle([8, 8, size - 8, size - 8], radius=52, fill=(9, 16, 27, 255), outline=(6, 182, 212, 255), width=10)
        # Inner accent line
        draw.rounded_rectangle([70, 70, size - 70, size - 70], radius=30, outline=(30, 130, 160, 180), width=6)
        # "EQ" text
        try:
            font = ImageFont.truetype("arialbd.ttf", 110)
        except Exception:
            font = ImageFont.load_default()
        txt = "EQ"
        bbox = draw.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = ((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1])
        draw.text(pos, txt, font=font, fill=(241, 245, 249, 255))
        img.save(ico_path, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        return ico_path
    except Exception as e:
        print(f"  (icon generation skipped: {e})")
        return None


def build(onedir=False, console=False, clean=True):
    print("🚀 Starting EquipAI Enterprise Build Process...")

    icon = _make_icon()

    params = [
        'ui_main.py',            # Main script (contains the __main__ launcher)
        '--name=EquipAI_Enterprise',
        '--noconfirm',           # Overwrite output without asking
        '--version-file=version_info.txt',
    ]
    if console:
        params.append('--console')       # Show console (diagnostics / dev)
    else:
        params.append('--windowed')      # No console window (production)
    if clean:
        params.append('--clean')

    if onedir:
        params.append('--onedir')
    else:
        params.append('--onefile')   # Single portable .exe

    if icon:
        params.append(f'--icon={icon}')

    # --- Bundle our modules as data so config/helpers are present at runtime ---
    for py in ("ui_components.py", "rag_engine.py", "llm_client.py", "session_manager.py",
               "feedback_manager.py", "config.py", "analytics.py", "model_manager.py"):
        params.append(f'--add-data={py}.;.')

    # --- pkg_resources bootstrap fix ---
    # setuptools 81's pkg_resources injects 'setuptools/_vendor' into sys.path at
    # import time and then imports top-level 'jaraco.*' FROM that injected path.
    # PyInstaller bundles those as PYZ modules under the full name
    # 'setuptools._vendor.jaraco.text', which is NOT importable as 'jaraco.text'
    # at runtime, so the pyi_rth_pkgres boot hook dies with
    # 'No module named jaraco'. Shipping the physical _vendor tree (as data
    # files) fixes it for both onedir (_internal) and onefile (_MEIPASS).
    try:
        import setuptools
        _vendor_src = os.path.join(os.path.dirname(setuptools.__file__), "_vendor")
        if os.path.isdir(_vendor_src):
            params.append(f'--add-data={_vendor_src};setuptools/_vendor')
    except Exception as e:
        print(f"  (setuptools vendor bundling skipped: {e})")

    # --- transformers dynamic model registry fix ---
    # transformers/models/__init__.py builds its import structure by SCANNING the
    # physical 'transformers/models' directory at runtime
    # (create_import_structure_from_path). Inside a frozen app the .py modules
    # report __file__ as '<root>\\transformers\\models\\__init__.pyc' - a path that
    # does not physically exist, so os.path.isfile() returns False, os.scandir()
    # then raises FileNotFoundError and transformers import dies.
    # We stage a copy of the models tree, plant a placeholder '__init__.pyc' next to
    # every '__init__.py' (so the isfile() probe passes and the scan drops to the
    # real directory), and ship the whole staged tree as data. The scan then
    # iterates the bundled .py files as in a normal installation.
    import tempfile
    try:
        import transformers as _tf
        _tf_dir = os.path.dirname(_tf.__file__)
        _models_src = os.path.join(_tf_dir, "models")
        if os.path.isdir(_models_src):
            _stage = os.path.join(tempfile.gettempdir(), "eqpai_tf_models")
            if os.path.isdir(_stage):
                shutil.rmtree(_stage, ignore_errors=True)
            shutil.copytree(_models_src, _stage)
            for _root, _dirs, _files in os.walk(_stage):
                if "__init__.py" in _files and not os.path.exists(
                    os.path.join(_root, "__init__.pyc")
                ):
                    with open(os.path.join(_root, "__init__.pyc"), "wb"):
                        pass
            params.append(f'--add-data={_stage};transformers/models')
    except Exception as e:
        print(f"  (transformers models bundling skipped: {e})")

    # --- Native / data-heavy dependencies ---
    # Targeted collection: torch/faiss/llama_cpp ship compiled binaries and
    # transformers/sentence_transformers ship data files. We deliberately avoid
    # `--collect-all=transformers` because it drags in gradio, tkinter, sqlalchemy,
    # sympy and other optional dependencies that bloat the EXE massively.
    params += [
        '--collect-binaries=llama_cpp',
        '--hidden-import=llama_cpp',
        '--collect-binaries=faiss',
        '--hidden-import=faiss',
        '--collect-data=sentence_transformers',
        '--collect-data=transformers',
        '--collect-all=huggingface_hub',
        '--collect-all=qtawesome',
        '--hidden-import=torch',
        '--runtime-hook=rthook_equipai.py',
        '--noupx',
    ]

    # --- Exclude optional/irrelevant heavy packages that get dragged in ---
    # (tensorflow/keras/torchvision/datasets-optional stacks are NOT used at
    # runtime: we only run the CPU torch + sentence-transformers embedding path.)
    params += [
        '--exclude-module=tensorflow',
        '--exclude-module=keras',
        '--exclude-module=tensorboard',
        '--exclude-module=torchvision',
        '--exclude-module=torchaudio',
        '--exclude-module=onnxruntime',
        '--exclude-module=cv2',
        '--exclude-module=av',
        '--exclude-module=soundfile',
'--exclude-module=librosa',
     '--exclude-module=pydub',
     '--exclude-module=nltk',
     '--exclude-module=matplotlib',
     '--exclude-module=altair',
     '--exclude-module=jsonschema',
        '--exclude-module=boto3',
        '--exclude-module=botocore',
        '--exclude-module=gevent',
        '--exclude-module=zope.interface',
        '--exclude-module=httplib2',
        '--exclude-module=grpc',
        '--exclude-module=orjson',
        '--exclude-module=uvicorn',
        '--exclude-module=gradio',
        '--exclude-module=gradio_client',
'--exclude-module=streamlit',
     '--exclude-module=tkinter',
     '--exclude-module=pytest',
     '--exclude-module=torch.testing._internal.opinfo',
     '--exclude-module=notebook',
    ]

    print("   PyInstaller parameters:")
    for p in params:
        print(f"     {p}")

    PyInstaller.__main__.run(params)

    print("\n✅ Build Complete!")


def assemble_portable():
    """Copies the built EXE plus external runtime folders into dist/EquipAI_Portable
    so the app runs WITHOUT installation or internet."""
    dist_dir = os.path.join(BASE_DIR, "dist")
    exe = os.path.join(dist_dir, "EquipAI_Enterprise.exe")
    if not os.path.exists(exe):
        print("   No onefile EXE found; skipping portable assembly.")
        return

    port = os.path.join(dist_dir, "EquipAI_Portable")
    print(f"📦 Assembling portable package in {port} ...")
    os.makedirs(port, exist_ok=True)

    shutil.copy2(exe, os.path.join(port, "EquipAI_Enterprise.exe"))

    # Runtime data folders (models should stay OUTSIDE the exe to keep size sane)
    for folder in ("llm_model", "embedding_model"):
        src = os.path.join(BASE_DIR, folder)
        dst = os.path.join(port, folder)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"   Copying {folder} ...")
            shutil.copytree(src, dst)

    # Fresh (empty) runtime folders for the portable deployment
    for folder in ("chat_sessions", "media", "models", "scratch", "temp_uploads", "document_digests"):
        os.makedirs(os.path.join(port, folder), exist_ok=True)

    readme = os.path.join(port, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "EquipAI Enterprise - Portable Package\n"
            "=====================================\n\n"
            "1. Double-click EquipAI_Enterprise.exe - no installation needed.\n"
            "2. Keep llm_model/ and embedding_model/ in this same folder.\n"
            "3. Knowledge digests and sessions are stored in this folder "
            "(chat_sessions/, vector_store.pkl, feedback_vectors.pkl).\n"
            "4. To reset all data, delete this folder and re-extract it.\n\n"
            "Optional environment overrides (e.g. set EQPAI_LLM_MODEL to pick a "
            "specific model file; EQPAI_DEFAULT_MODE=fast|balanced|research).\n"
        )
    print("✅ Portable folder ready:", port)


if __name__ == "__main__":
    _onedir = "--onedir" in os.sys.argv[1:]
    _console = "--console" in os.sys.argv[1:]
    build(onedir=_onedir, console=_console, clean=not _onedir)
    if not _onedir:
        assemble_portable()