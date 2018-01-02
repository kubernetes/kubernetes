import abc

from ct.client.db import temp_db
from ct.client.db import database
from ct.proto import client_pb2

# This class provides common tests for all CT log TempDB implementations.
# It only inherits from object so that unittest won't attempt to run the test_*
# methods on this class. Derived classes should use multiple inheritance
# from TempDBTest and unittest.TestCase to get test automation.
class TempDBTest(object):
    """All TempDB tests should derive from this class as well as
    unittest.TestCase."""
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def make_entries(start, end):
        entries = []
        for i in range(start, end+1):
            entry = client_pb2.EntryResponse()
            entry.leaf_input = "leaf_input-%d" % i
            entry.extra_data = "extra_data-%d" % i
            entries.append((i, entry))
        return entries

    def verify_entries(self, entries, start, end, prefix=""):
        self.assertEqual(end-start+1, len(entries))
        for i in range(start, end+1):
            self.assertEqual("leaf_input-%d" % i, entries[i].leaf_input)
            self.assertEqual("extra_data-%d" % i, entries[i].extra_data)

    @abc.abstractmethod
    def db(self):
        """Derived classes must override to initialize a database."""

    def test_store_and_scan(self):
        entries = self.make_entries(0, 9)
        self.db().store_entries(entries)
        returned_entries = list(self.db().scan_entries(0, 9))
        self.verify_entries(returned_entries, 0, 9)

    def test_store_twice_doesnt_raise(self):
        entries = self.make_entries(0, 9)
        self.db().store_entries(entries)
        entries = self.make_entries(1, 10)
        self.db().store_entries(entries)
        self.verify_entries(list(self.db().scan_entries(0, 10)), 0, 10)

    def test_scan_out_of_range_fails(self):
        entries = self.make_entries(0, 9)
        self.db().store_entries(entries[:-1])
        returned_entries = self.db().scan_entries(0, 9)
        for _ in range(9):
            returned_entries.next()
        self.assertRaises(database.KeyError, returned_entries.next)
