#!/usr/bin/env python

import unittest

import sys
from ct.client.db import sqlite_connection as sqlitecon
from ct.client.db import sqlite_cert_db
from ct.client.db import cert_db_test
import gflags

class SQLiteCertDBTest(unittest.TestCase, cert_db_test.CertDBTest):
    def setUp(self):
        self.database = sqlite_cert_db.SQLiteCertDB(
            sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True))
    def db(self):
        return self.database

if __name__ == '__main__':
    sys.argv = gflags.FLAGS(sys.argv)
    unittest.main()
