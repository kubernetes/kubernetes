#!/usr/bin/env python

import unittest

from ct.client.db import sqlite_connection as sqlitecon
from ct.client.db import sqlite_log_db
from ct.client.db import log_db_test

class SQLiteLogDBTest(unittest.TestCase, log_db_test.LogDBTest):
    def setUp(self):
        self.database = sqlite_log_db.SQLiteLogDB(
            sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True))
    def db(self):
        return self.database

if __name__ == '__main__':
    unittest.main()
