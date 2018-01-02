#!/usr/bin/env python
import unittest

import mock
from ct.client import db_reporter


class DbReporterTest(unittest.TestCase):
    def test_report(self):
        db = mock.MagicMock()
        reporter = db_reporter.CertDBCertificateReport(db, 1, checks=[])
        for j in range(1, 6):
            for i in range(0, 10):
                reporter._batch_scanned_callback([(None, None, None)])
            reporter.report()
            self.assertEqual(db.store_certs_desc.call_count, 10 * j)

    def test_db_raising_does_not_stall_reporter(self):
        db = mock.Mock()
        db.store_certs_desc.side_effect = [ValueError("Boom!"), None]

        reporter = db_reporter.CertDBCertificateReport(db, 1, checks=[])
        reporter._batch_scanned_callback([(None, None, None)])
        reporter._batch_scanned_callback([(None, None, None)])
        reporter.report()
        self.assertEqual(db.store_certs_desc.call_count, 2)

if __name__ == '__main__':
    unittest.main()
