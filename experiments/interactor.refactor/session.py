#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: session.py
# Description: EchoAI session manager with full CRUD and branching
# Author: Ms. White
# Created: 2025-05-02
# Modified: 2025-05-13 16:04:13

import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, List, Dict


class Session:
    def __init__(self, directory: str = None):
        """
        Initialize the session manager and ensure the session directory exists.

        Args:
            directory (str): Filesystem path for session storage. Must not be None or empty.

        Raises:
            ValueError: If directory is None or not a string.
            OSError: If the directory cannot be created or accessed.
        """
        if not directory:
            raise ValueError("Session directory must be a valid non-empty string path.")

        try:
            self.path = Path(os.path.expanduser(directory))
            self.path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to initialize session directory '{directory}': {e}")

    # ---------------------------
    # Core CRUD
    # ---------------------------

    def list(self) -> List[Dict]:
        """
        Return metadata for all sessions in the directory.

        Returns:
            List[Dict]: Sorted list of session metadata dictionaries.
        """
        out = []
        for file in self.path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    d = json.load(f)
                    out.append({
                        "id": d.get("id"),
                        "name": d.get("name"),
                        "created": d.get("created"),
                        "tags": d.get("tags", []),
                        "summary": d.get("summary")
                    })
            except Exception:
                continue
        return sorted(out, key=lambda x: x["created"], reverse=True)

    def create(self, name: str, tags: Optional[List[str]] = None) -> str:
        """
        Create and persist a new session.

        Args:
            name (str): Name of the new session.
            tags (List[str], optional): Optional list of tags.

        Returns:
            str: Unique session ID of the new session.
        """
        sid = str(uuid.uuid4())
        session = {
            "id": sid,
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(),
            "parent": None,
            "branch_point": None,
            "tags": tags or [],
            "summary": None,
            "messages": []
        }
        self._save_file(sid, session)
        return sid

    def load(self, session_id: str) -> List[Dict]:
        """
        Return OpenAI-compatible message list from a session.

        Filters out internal keys and leaves only standard API-compatible fields.

        Args:
            session_id (str): ID of the session to load.

        Returns:
            List[Dict]: List of clean message dictionaries.
        """
        session = self._read_file(session_id)
        return [
            {k: v for k, v in m.items() if k in {
                "role", "content", "tool_calls", "name", "function_call", "tool_call_id"
            }} for m in session.get("messages", [])
        ]

    def load_full(self, session_id: str) -> Dict:
        """
        Return the complete session file as-is.

        Args:
            session_id (str): ID of the session.

        Returns:
            Dict: Entire raw session data.
        """
        return self._read_file(session_id)

    def delete(self, session_id: str):
        """
        Delete a session file from disk.

        Args:
            session_id (str): ID of the session to delete.
        """
        file = self.path / f"{session_id}.json"
        if file.exists():
            file.unlink()

    def update(self, session_id: str, key: str, value: Any):
        """
        Update a top-level key in a session file.

        Args:
            session_id (str): Session ID.
            key (str): Field to update.
            value (Any): New value for the field.
        """
        session = self._read_file(session_id)
        session[key] = value
        self._save_file(session_id, session)

    # ---------------------------
    # Message Operations
    # ---------------------------

    def msg_insert(self, session_id: str, message: Dict) -> str:
        """
        Insert a new message into a session.

        Args:
            session_id (str): Session ID.
            message (Dict): Message dictionary to insert.

        Returns:
            str: ID of the inserted message.
        """
        session = self._read_file(session_id)
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **message
        }
        session["messages"].append(entry)
        self._save_file(session_id, session)
        return entry["id"]

    def msg_get(self, session_id: str, message_id: str) -> Optional[Dict]:
        """
        Retrieve a specific message from a session.

        Args:
            session_id (str): Session ID.
            message_id (str): ID of the message to retrieve.

        Returns:
            Optional[Dict]: The message if found, else None.
        """
        session = self._read_file(session_id)
        for msg in session.get("messages", []):
            if msg.get("id") == message_id:
                return msg
        return None

    def msg_index(self, session_id: str, message_id: str) -> Optional[int]:
        """
        Get the index of a message within a session.

        Args:
            session_id (str): Session ID.
            message_id (str): Message ID.

        Returns:
            Optional[int]: Index if found, else None.
        """
        session = self._read_file(session_id)
        for i, msg in enumerate(session.get("messages", [])):
            if msg.get("id") == message_id:
                return i
        return None

    def msg_update(self, session_id: str, message_id: str, new_content: str) -> bool:
        """
        Update the content of a specific message.

        Args:
            session_id (str): Session ID.
            message_id (str): Message ID.
            new_content (str): New content for the message.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        session = self._read_file(session_id)
        for m in session["messages"]:
            if m.get("id") == message_id:
                m["content"] = new_content
                self._save_file(session_id, session)
                return True
        return False

    def msg_delete(self, session_id: str, message_id: str) -> bool:
        """
        Delete a message from a session.

        Args:
            session_id (str): Session ID.
            message_id (str): Message ID.

        Returns:
            bool: True if deletion occurred, False otherwise.
        """
        session = self._read_file(session_id)
        before = len(session["messages"])
        session["messages"] = [m for m in session["messages"] if m.get("id") != message_id]
        self._save_file(session_id, session)
        return len(session["messages"]) < before

    # ---------------------------
    # Branching & Summarization
    # ---------------------------

    def branch(self, from_id: str, message_id: str, new_name: str) -> str:
        """Create a new session by branching from a specific message.

        This method creates a new session that branches from an existing one at a specific
        message point. The new session inherits all messages up to and including the
        specified message, then starts fresh from there.

        Args:
            from_id (str): ID of the source session to branch from.
            message_id (str): ID of the message to branch at.
            new_name (str): Name for the new branched session.

        Returns:
            str: ID of the newly created branched session.

        Raises:
            ValueError: If the source session or message ID is not found.
        """
        # Get source session
        source = self._read_file(from_id)
        if not source:
            raise ValueError(f"Source session '{from_id}' not found")

        # Find the branch point
        branch_index = self.msg_index(from_id, message_id)
        if branch_index is None:
            raise ValueError(f"Message '{message_id}' not found in session '{from_id}'")

        # Create new session
        new_id = self.create(new_name, source.get("tags", []))
        new_session = self._read_file(new_id)

        # Copy messages up to branch point
        new_session["messages"] = source["messages"][:branch_index + 1]
        new_session["parent"] = from_id
        new_session["branch_point"] = message_id

        # Save and return
        self._save_file(new_id, new_session)
        return new_id

    def summarize(self, interactor, session_id: str) -> str:
        """Generate a summary of the session using the provided interactor.

        This method uses the AI interactor to analyze the session content and generate
        a concise summary. The summary is stored in the session metadata and returned.

        Args:
            interactor: An AI interactor instance capable of generating summaries.
            session_id (str): ID of the session to summarize.

        Returns:
            str: The generated summary text.

        Note:
            The summary is automatically stored in the session metadata and can be
            retrieved later using load_full().
        """
        session = self._read_file(session_id)
        if not session:
            return ""

        # Get clean message list
        messages = self.load(session_id)
        if not messages:
            return ""

        # Generate summary
        summary = interactor.interact(
            "Summarize this conversation in 2-3 sentences:",
            tools=False,
            stream=False,
            markdown=False
        )

        # Store and return
        session["summary"] = summary
        self._save_file(session_id, session)
        return summary

    # ---------------------------
    # Search Capabilities
    # ---------------------------

    def search(self, query: str, session_id: Optional[str] = None) -> List[Dict]:
        """Search for messages containing the query text within a session or all sessions.

        This method performs a case-insensitive text search across message content.
        If a session_id is provided, only searches within that session. Otherwise,
        searches across all sessions.

        Args:
            query (str): Text to search for.
            session_id (Optional[str]): Optional session ID to limit search scope.

        Returns:
            List[Dict]: List of matching messages with their session context.
            Each dict contains:
                - session_id: ID of the containing session
                - message: The matching message
                - context: Surrounding messages for context
        """
        results = []
        query = query.lower()

        # Determine search scope
        if session_id:
            sessions = [(session_id, self._read_file(session_id))]
        else:
            sessions = [(f.stem, self._read_file(f.stem)) for f in self.path.glob("*.json")]

        # Search each session
        for sid, session in sessions:
            if not session:
                continue

            messages = session.get("messages", [])
            for i, msg in enumerate(messages):
                content = str(msg.get("content", "")).lower()
                if query in content:
                    # Get context (2 messages before and after)
                    start = max(0, i - 2)
                    end = min(len(messages), i + 3)
                    context = messages[start:end]

                    results.append({
                        "session_id": sid,
                        "message": msg,
                        "context": context
                    })

        return results

    def search_meta(self, query: str) -> List[Dict]:
        """Search session metadata (name, tags, summary) for matching sessions.

        This method performs a case-insensitive search across session metadata fields
        including name, tags, and summary. It returns matching sessions with their
        full metadata.

        Args:
            query (str): Text to search for in metadata.

        Returns:
            List[Dict]: List of matching session metadata dictionaries.
            Each dict contains:
                - id: Session ID
                - name: Session name
                - created: Creation timestamp
                - tags: List of tags
                - summary: Session summary if available
        """
        results = []
        query = query.lower()

        for file in self.path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    session = json.load(f)
                    
                    # Check metadata fields
                    name = str(session.get("name", "")).lower()
                    tags = [str(t).lower() for t in session.get("tags", [])]
                    summary = str(session.get("summary", "")).lower()
                    
                    if (query in name or
                        any(query in tag for tag in tags) or
                        query in summary):
                        results.append({
                            "id": session.get("id"),
                            "name": session.get("name"),
                            "created": session.get("created"),
                            "tags": session.get("tags", []),
                            "summary": session.get("summary")
                        })
            except Exception:
                continue

        return sorted(results, key=lambda x: x["created"], reverse=True)

    # ---------------------------
    # Internal I/O
    # ---------------------------

    def _read_file(self, session_id: str) -> Dict:
        """Read and parse a session file from disk.

        This internal method handles reading and parsing session files.
        It ensures proper error handling and returns an empty session
        structure if the file doesn't exist or is invalid.

        Args:
            session_id (str): ID of the session to read.

        Returns:
            Dict: Session data dictionary or empty session structure.
        """
        file = self.path / f"{session_id}.json"
        if not file.exists():
            return {
                "id": session_id,
                "name": "New Session",
                "created": datetime.now(timezone.utc).isoformat(),
                "messages": []
            }

        try:
            with open(file, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "id": session_id,
                "name": "New Session",
                "created": datetime.now(timezone.utc).isoformat(),
                "messages": []
            }

    def _save_file(self, session_id: str, data: Dict):
        """Write session data to disk.

        This internal method handles writing session data to disk.
        It ensures proper error handling and atomic writes.

        Args:
            session_id (str): ID of the session to save.
            data (Dict): Session data to write.

        Raises:
            OSError: If the file cannot be written.
        """
        file = self.path / f"{session_id}.json"
        temp_file = file.with_suffix(".tmp")

        try:
            # Write to temporary file first
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise OSError(f"Failed to save session '{session_id}': {e}")

