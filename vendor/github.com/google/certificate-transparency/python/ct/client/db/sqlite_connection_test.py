#!/usr/bin/env python

import unittest
import sqlite3

from ct.client.db import sqlite_connection as sqlitecon

class SQLiteConnectionTest(unittest.TestCase):
    def test_connection_works(self):
        with sqlitecon.SQLiteConnection(":memory:") as conn:
            conn.execute("CREATE TABLE words(word TEXT)")
            conn.execute("INSERT INTO words VALUES (?)", ("hello",))
            results = conn.execute("SELECT * FROM words")
            self.assertEqual("hello", results.next()["word"])
            self.assertRaises(StopIteration, results.next)

    def test_exit_autocommits(self):
        # Need keepalive=True as the memory db only lives as long as the
        # connection.
        with sqlitecon.SQLiteConnection(":memory:", keepalive=True) as conn:
            conn.execute("CREATE TABLE words(word TEXT)")
            conn.execute("INSERT INTO words VALUES (?)", ("hello",))
        results = conn.execute("SELECT * FROM words")
        self.assertEqual("hello", results.next()["word"])

    def test_no_keepalive_closes_connection(self):
        with sqlitecon.SQLiteConnection(":memory:", keepalive=False) as conn:
            conn.execute("CREATE TABLE words(word TEXT)")
        self.assertRaises(sqlite3.ProgrammingError, conn.execute,
                          "SELECT * FROM words")

class SQLiteConnectionManagerTest(unittest.TestCase):
    def test_get_connection(self):
        mgr = sqlitecon.SQLiteConnectionManager(":memory:")
        self.assertIsInstance(mgr.get_connection(), sqlitecon.SQLiteConnection)

    def test_keepalive_returns_same_connection(self):
        mgr = sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True)
        with mgr.get_connection() as conn:
            conn.execute("CREATE TABLE words(word TEXT)")
            conn.execute("INSERT INTO words VALUES (?)", ("hello",))

        with mgr.get_connection() as conn:
            results = conn.execute("SELECT * FROM words")
            self.assertEqual("hello", results.next()["word"])

    def test_no_keepalive_returns_new_connection(self):
        mgr = sqlitecon.SQLiteConnectionManager(":memory:", keepalive=False)
        with mgr.get_connection() as conn:
            conn.execute("CREATE TABLE words(word TEXT)")
            conn.execute("INSERT INTO words VALUES (?)", ("hello",))

        with mgr.get_connection() as conn:
            self.assertRaises(sqlite3.OperationalError, conn.execute,
                              "SELECT * FROM words")

if __name__ == '__main__':
    unittest.main()
