import abc

class LogDB(object):
    """Database interface for storing client-side CT data."""
    __metaclass__ = abc.ABCMeta

    # The largest BSON can handle
    timestamp_max = 2**63-1

    @abc.abstractmethod
    def add_log(self, metadata):
        """Store log metadata. This creates the necessary mappings between
        tables so all logs must be explicitly added.
        Params:
            metadata: a client_pb2.CtLogMetadata proto."""

    @abc.abstractmethod
    def update_log(self, metadata):
        """Add a new log or update existing log metadata. When updating, does
        not verify that the new metadata is consistent with stored values.
        Params:
            metadata: a client_pb2.CtLogMetadata proto."""

    @abc.abstractmethod
    def logs(self):
        """A generator that yields all currently known logs."""

    @abc.abstractmethod
    def store_sth(self, log_server, audited_sth):
        """Store the STH in the database.
        Will store the STH with a unique ID unless an exact copy already exists.
        Params:
            log_server: the server name, i.e., the <log_server> path prefix
            audited_sth: a client_pb2.AuditedSth proto
        """

    @abc.abstractmethod
    def get_latest_sth(self, log_server):
        """"Get the AuditedSth with the latest timestamp."""

    @abc.abstractmethod
    def scan_latest_sth_range(self, log_server, start=0, end=timestamp_max,
                              limit=0):
        """Scan STHs by timestamp
        Args:
            logid: CT log to scan
            start: earliest timestamp
            end: latest timestamp
            limit: maximum number of entries to return. Default is no limit.
        Yields:
            the AuditedSth protos in descending order of timestamps
        Note the scan may be keeping the database connection open until the
        generator is exhausted.
        """

    @abc.abstractmethod
    def get_log_id(self, log_server):
        """Get id in database of log_server."""
