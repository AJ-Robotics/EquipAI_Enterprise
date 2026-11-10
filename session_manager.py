import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

SESSIONS_DIR = "chat_sessions"

class SessionManager:
    def __init__(self):
        if not os.path.exists(SESSIONS_DIR):
            os.makedirs(SESSIONS_DIR)

    def create_new_session(self) -> str:
        """Creates a new session ID and returns it."""
        return str(uuid.uuid4())

    def save_session(self, session_id: str, messages: List[Dict[str, Any]], title: str = None):
        """Saves the current session to a JSON file."""
        if not messages:
            return

        # Generate a title if not provided/exists, based on first user message
        if not title:
            for msg in messages:
                if msg['role'] == 'user':
                    title = msg['content'][:30] + "..."
                    break
            if not title:
                title = "New Chat"

        data = {
            "session_id": session_id,
            "title": title,
            "last_updated": datetime.now().isoformat(),
            "messages": messages
        }
        
        file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Loads a session by ID."""
        file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all available sessions sorted by date."""
        sessions = []
        if not os.path.exists(SESSIONS_DIR):
            return []
            
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data["session_id"],
                            "title": data.get("title", "Untitled Chat"),
                            "last_updated": data.get("last_updated", "")
                        })
                except:
                    continue
        
        # Sort by last_updated descending
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)
        return sessions

    def delete_session(self, session_id: str):
        file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
