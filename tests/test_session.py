#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_session.py
# Description: Test suite for EchoAI session.py
# Author: Ms. White
# Created: 2025-05-02
# Modified: 2025-05-05 18:19:35

import unittest
import shutil
from pathlib import Path
from session import Session


class TestSession(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_sessions")
        self.test_dir.mkdir(exist_ok=True)
        self.s = Session(directory=str(self.test_dir))
        self.session_id = self.s.create("unit-test-session", tags=["unit", "test"])

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_and_list(self):
        sessions = self.s.list()
        ids = [s["id"] for s in sessions]
        self.assertIn(self.session_id, ids)

    def test_msg_insert_and_get(self):
        msg_id = self.s.msg_insert(self.session_id, {"role": "user", "content": "Hello"})
        msg = self.s.msg_get(self.session_id, msg_id)
        self.assertEqual(msg["content"], "Hello")
        self.assertEqual(msg["role"], "user")

    def test_msg_index(self):
        msg_id = self.s.msg_insert(self.session_id, {"role": "user", "content": "Test index"})
        index = self.s.msg_index(self.session_id, msg_id)
        self.assertIsInstance(index, int)

    def test_load_and_clean_fields(self):
        self.s.msg_insert(self.session_id, {"role": "user", "content": "Strip test"})
        clean = self.s.load(self.session_id)[0]
        self.assertIn("role", clean)
        self.assertNotIn("timestamp", clean)

    def test_load_full(self):
        full = self.s.load_full(self.session_id)
        self.assertIn("messages", full)
        self.assertEqual(full["id"], self.session_id)

    def test_update_summary(self):
        self.s.update(self.session_id, "summary", "Session summary here.")
        updated = self.s.load_full(self.session_id)
        self.assertEqual(updated["summary"], "Session summary here.")

    def test_msg_update_and_delete(self):
        msg_id = self.s.msg_insert(self.session_id, {"role": "user", "content": "To be updated"})
        self.s.msg_update(self.session_id, msg_id, "Updated!")
        updated = self.s.msg_get(self.session_id, msg_id)
        self.assertEqual(updated["content"], "Updated!")

        self.s.msg_delete(self.session_id, msg_id)
        deleted = self.s.msg_get(self.session_id, msg_id)
        self.assertIsNone(deleted)

    def test_branch(self):
        msg_id = self.s.msg_insert(self.session_id, {"role": "user", "content": "Fork point"})
        branch_id = self.s.branch(self.session_id, msg_id, "branch-test")
        branch_data = self.s.load_full(branch_id)
        self.assertEqual(branch_data["parent"], self.session_id)
        self.assertEqual(branch_data["branch_point"], msg_id)
        self.assertEqual(len(branch_data["messages"]), 1)

    def test_search_and_meta_search(self):
        self.s.msg_insert(self.session_id, {"role": "user", "content": "Searchable content"})
        results = self.s.search("Searchable")
        self.assertTrue(any(r["id"] == self.session_id for r in results))

        meta = self.s.search_meta("unit-test")
        self.assertTrue(any(r["id"] == self.session_id for r in meta))


if __name__ == "__main__":
    unittest.main()

