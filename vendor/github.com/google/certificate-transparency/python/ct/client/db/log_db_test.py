import abc
from ct.client.db import log_db
from ct.client.db import database
from ct.proto import client_pb2

# This class provides common tests for all CT log database implementations.
# It only inherits from object so that unittest won't attempt to run the test_*
# methods on this class. Derived classes should use multiple inheritance
# from LogDBTest and unittest.TestCase to get test automation.
class LogDBTest(object):
    """All LogDB tests should derive from this class as well as
    unittest.TestCase."""
    __metaclass__ = abc.ABCMeta

    # Set up a default fake test log server and STH.
    default_log = client_pb2.CtLogMetadata()
    default_log.log_server = "test"
    default_log.log_id = "c29tZWtleWlk"  # b64("somekeyid")
    default_log.public_key_info.type = client_pb2.KeyInfo.ECDSA
    default_log.public_key_info.pem_key = "base64encodedkey"

    default_sth = client_pb2.AuditedSth()
    default_sth.sth.timestamp = 1234
    default_sth.sth.sha256_root_hash = "base64hash"
    default_sth.audit.status = client_pb2.VERIFIED

    @abc.abstractmethod
    def db(self):
        """Derived classes must override to initialize a database."""
        pass

    def test_add_log(self):
        self.db().add_log(LogDBTest.default_log)
        generator = self.db().logs()
        metadata = generator.next()
        self.assertEqual(metadata, LogDBTest.default_log)
        self.assertRaises(StopIteration, generator.next)

    def test_update_log(self):
        self.db().add_log(LogDBTest.default_log)
        self.db().store_sth(LogDBTest.default_log.log_server,
                            LogDBTest.default_sth)

        new_log = client_pb2.CtLogMetadata()
        new_log.CopyFrom(LogDBTest.default_log)
        new_log.public_key_info.pem_key = "newkey"
        self.db().update_log(new_log)
        generator = self.db().logs()
        metadata = generator.next()
        self.assertEqual(metadata, new_log)
        self.assertRaises(StopIteration, generator.next)

        # Should still be able to access STHs after updating log metadata
        read_sth = self.db().get_latest_sth(new_log.log_server)
        self.assertTrue(read_sth)
        self.assertEqual(LogDBTest.default_sth, read_sth)

    def test_update_log_adds_log(self):
        self.db().update_log(LogDBTest.default_log)
        generator = self.db().logs()
        metadata = generator.next()
        self.assertEqual(metadata, LogDBTest.default_log)
        self.assertRaises(StopIteration, generator.next)

    def test_store_sth(self):
        self.db().add_log(LogDBTest.default_log)
        self.db().store_sth(LogDBTest.default_log.log_server,
                            LogDBTest.default_sth)
        read_sth = self.db().get_latest_sth(LogDBTest.default_log.log_server)
        self.assertTrue(read_sth)
        self.assertEqual(LogDBTest.default_sth, read_sth)

    def test_store_sth_ignores_duplicate(self):
        self.db().add_log(LogDBTest.default_log)
        self.db().store_sth(LogDBTest.default_log.log_server,
                            LogDBTest.default_sth)
        duplicate_sth = client_pb2.AuditedSth()
        duplicate_sth.audit.status = client_pb2.VERIFY_ERROR
        self.db().store_sth(LogDBTest.default_log.log_server, duplicate_sth)
        read_sth = self.db().get_latest_sth(LogDBTest.default_log.log_server)
        self.assertTrue(read_sth)
        self.assertEqual(LogDBTest.default_sth, read_sth)

    def test_log_not_found_raises(self):
        self.assertRaises(database.KeyError, self.db().store_sth,
                          LogDBTest.default_log.log_server,
                          LogDBTest.default_sth)

    def test_get_latest_sth_returns_latest(self):
        self.db().add_log(LogDBTest.default_log)
        self.db().store_sth(LogDBTest.default_log.log_server,
                            LogDBTest.default_sth)
        new_sth = client_pb2.AuditedSth()
        new_sth.CopyFrom(LogDBTest.default_sth)
        new_sth.sth.timestamp = LogDBTest.default_sth.sth.timestamp - 1
        self.db().store_sth(LogDBTest.default_log.log_server, new_sth)
        read_sth = self.db().get_latest_sth(LogDBTest.default_log.log_server)
        self.assertIsNotNone(read_sth)
        self.assertEqual(LogDBTest.default_sth, read_sth)

    def test_get_latest_sth_returns_none_if_empty(self):
        self.db().add_log(LogDBTest.default_log)
        self.assertIsNone(self.db().get_latest_sth(
            LogDBTest.default_log.log_server))

    def test_get_latest_sth_honours_log_server(self):
        self.db().add_log(LogDBTest.default_log)
        self.db().store_sth(LogDBTest.default_log.log_server,
                            LogDBTest.default_sth)
        new_sth = client_pb2.AuditedSth()
        new_sth.CopyFrom(LogDBTest.default_sth)
        new_sth.sth.timestamp = LogDBTest.default_sth.sth.timestamp + 1

        new_log = client_pb2.CtLogMetadata()
        new_log.log_server = "test2"
        new_log.log_id = "c29tZW90aGVya2V5aWQ="  # b64("someotherkeyid")
        self.db().add_log(new_log)

        new_sth.sth.sha256_root_hash = "hash2"
        self.db().store_sth(new_log.log_server, new_sth)
        read_sth = self.db().get_latest_sth(LogDBTest.default_log.log_server)
        self.assertIsNotNone(read_sth)
        self.assertEqual(LogDBTest.default_sth, read_sth)

    def test_scan_latest_sth_range_finds_all(self):
        self.db().add_log(LogDBTest.default_log)
        for i in range(4):
            sth = client_pb2.AuditedSth()
            sth.sth.timestamp = i
            sth.sth.sha256_root_hash = "hash-%d" % i
            self.db().store_sth(LogDBTest.default_log.log_server, sth)

        generator = self.db().scan_latest_sth_range(
            LogDBTest.default_log.log_server)
        for i in range(3, -1, -1):
            sth = generator.next()
            # Scan runs in descending timestamp order
            self.assertEqual(sth.sth.timestamp, i)
            self.assertEqual(sth.sth.sha256_root_hash, "hash-%d" % i)

        self.assertRaises(StopIteration, generator.next)

    def test_scan_latest_sth_range_honours_log_server(self):
        for i in range(4):
            log = client_pb2.CtLogMetadata()
            log.log_server = "test-%d" % i
            self.db().add_log(log)
        for i in range(4):
            sth = client_pb2.AuditedSth()
            sth.sth.timestamp = i
            sth.sth.sha256_root_hash = "hash-%d" % i
            self.db().store_sth("test-%d" % i, sth)

        for i in range(4):
            generator = self.db().scan_latest_sth_range("test-%d" % i)
            sth = generator.next()
            self.assertEqual(sth.sth.timestamp, i)
            self.assertEqual(sth.sth.sha256_root_hash, "hash-%d" % i)

    def test_scan_latest_sth_range_honours_range(self):
        self.db().add_log(LogDBTest.default_log)
        for i in range(4):
            sth = client_pb2.AuditedSth()
            sth.sth.timestamp = i
            sth.sth.sha256_root_hash = "hash-%d" % i
            self.db().store_sth(LogDBTest.default_log.log_server, sth)

        generator = self.db().scan_latest_sth_range("test", start=1, end=2)
        for i in range(2):
            sth = generator.next()
            self.assertEqual(sth.sth.timestamp, 2-i)
            self.assertEqual(sth.sth.sha256_root_hash, "hash-%d" % (2-i))

        self.assertRaises(StopIteration, generator.next)

    def test_scan_latest_sth_range_honours_limit(self):
        self.db().add_log(LogDBTest.default_log)
        for i in range(4):
            sth = client_pb2.AuditedSth()
            sth.sth.timestamp = i
            sth.sth.sha256_root_hash = "hash-%d" % i
            self.db().store_sth(LogDBTest.default_log.log_server, sth)

        generator = self.db().scan_latest_sth_range("test", limit=1)
        sth = generator.next()
        # Returns most recent
        self.assertEqual(sth.sth.timestamp, 3)
        self.assertEqual(sth.sth.sha256_root_hash, "hash-%d" % 3)

        self.assertRaises(StopIteration, generator.next)
