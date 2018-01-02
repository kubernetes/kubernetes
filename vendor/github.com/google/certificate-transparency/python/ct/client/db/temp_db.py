import abc

class TempDB(object):
    """Database interface for storing unverified CT log entry data."""
    @abc.abstractmethod
    def drop_entries(self):
        """Drop all entries."""

    @abc.abstractmethod
    def store_entries(self, entries):
        """Batch store log entries.
        Args:
            entries: an iterable of (entry_number, client_pb2.EntryResponse)
                     tuples
       """

    @abc.abstractmethod
    def scan_entries(self, start, end):
        """Retrieve log entries.
        Args:
            start: index of the first entry to retrieve.
            end: index of the last entry to retrieve.
        Yields:
            client_pb2.EntryResponse protos.
        Raises:
            KeyError: an entry with a sequence number in the range does not
                      exist.
        """
