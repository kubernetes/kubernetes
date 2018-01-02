#!/usr/bin/env python

import unittest

import os
import tempfile

from ct.client.db import sqlite_connection as sqlitecon
from ct.client.db import sqlite_temp_db
from ct.client.db import temp_db_test

class SQLiteTempDBTest(unittest.TestCase, temp_db_test.TempDBTest):
    def setUp(self):
        self.database = sqlite_temp_db.SQLiteTempDB(
            sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True))
    def db(self):
        return self.database

class SQLiteTempDBFactoryTest(unittest.TestCase):
    def setUp(self):
        self.database_dir = tempfile.mkdtemp()
        self.factory = sqlite_temp_db.SQLiteTempDBFactory(
            sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True),
            self.database_dir)
        self.temp_stores = set()

    def create_storage(self, name):
        storage = self.factory.create_storage(name)
        self.temp_stores.add(storage.db_name())
        return storage

    def tearDown(self):
        for temp_store in self.temp_stores:
            os.remove(temp_store)
        os.rmdir(self.database_dir)

    def test_create_storage(self):
        temp = self.create_storage("log_server")
        self.assertEqual(self.database_dir, os.path.dirname(temp.db_name()))

    def test_create_storage_creates_unique(self):
        temp1 = self.create_storage("log_server1")
        temp2 = self.create_storage("log_server2")
        self.assertNotEqual(temp1.db_name(), temp2.db_name())

    def test_create_storage_remembers_mapping(self):
        temp = self.create_storage("log_server")
        temp2 = self.create_storage("log_server")
        self.assertEqual(temp.db_name(), temp2.db_name())

if __name__ == '__main__':
    unittest.main()
