import logging
import sqlite3
import time

from ct.client.db import log_db
from ct.client.db import database
from ct.proto import client_pb2

class SQLiteLogDB(log_db.LogDB):
    def __init__(self, connection_manager):
        """Initialize the database and tables.
        Args:
            connection_manager: an SQLiteConnectionManager object."""
        self.__mgr = connection_manager

        with self.__mgr.get_connection() as conn:
            # TODO(ekasper): give users control of table names via flags so
            # we can explicitly avoid conflicts between database objects
            # sharing the same underlying SQLiteConnection.
            conn.execute("CREATE TABLE IF NOT EXISTS logs("
                         "id INTEGER PRIMARY KEY, log_server TEXT UNIQUE, "
                         "metadata BLOB)")
            conn.execute("CREATE TABLE IF NOT EXISTS sths(log_id INTEGER, "
                         "fetch_timestamp INTEGER,"
                         "timestamp INTEGER, sth_data BLOB, "
                         "audit_info BLOB,"
                         "UNIQUE(log_id, timestamp, sth_data, audit_info) ON "
                         "CONFLICT IGNORE,"
                         "FOREIGN KEY(log_id) REFERENCES logs(id))")
            conn.execute("CREATE INDEX IF NOT EXISTS sth_by_timestamp on sths("
                         "log_id, timestamp)")
        self.__tables = ["logs", "sths"]

    def __repr__(self):
        return "%r(db: %r)" % (self.__class__.__name__, self.__mgr)

    def __str__(self):
        return "%s(db: %s, tables: %s): " % (self.__class__.__name__,
                                             self.__mgr, self.__tables)

    def __encode_log_metadata(self, metadata):
        log_server = metadata.log_server
        local_metadata = client_pb2.CtLogMetadata()
        local_metadata.CopyFrom(metadata)
        local_metadata.ClearField("log_server")
        return log_server, sqlite3.Binary(local_metadata.SerializeToString())

    def __decode_log_metadata(self, log_server, serialized_metadata):
        metadata = client_pb2.CtLogMetadata()
        metadata.ParseFromString(serialized_metadata)
        metadata.log_server = log_server
        return metadata

    def add_log(self, metadata):
        log_server, serialized_metadata = self.__encode_log_metadata(
            metadata)
        with self.__mgr.get_connection() as conn:
            try:
                conn.execute("INSERT INTO logs(log_server, metadata) "
                             "VALUES(?, ?)", (log_server, serialized_metadata))
            except sqlite3.IntegrityError:
                logging.warning("Ignoring duplicate log server %s", log_server)

    def update_log(self, metadata):
        log_server, serialized_metadata = self.__encode_log_metadata(
            metadata)
        with self.__mgr.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO logs(id, log_server, "
                         "metadata) VALUES((SELECT id FROM logs WHERE "
                         "log_server = ?), ?, ?) ", (log_server, log_server,
                                                     serialized_metadata))

    def logs(self):
        with self.__mgr.get_connection() as conn:
            for log_server, metadata in conn.execute(
                "SELECT log_server, metadata FROM logs"):
                yield self.__decode_log_metadata(log_server, metadata)

    def _get_log_id(self, conn, log_server):
        res = conn.execute("SELECT id FROM logs WHERE log_server = ?",
                           (log_server,))
        try:
            log_id = res.next()
        except StopIteration:
            raise database.KeyError("Unknown log server: %s", log_server)
        return log_id[0]

    def get_log_id(self, log_server):
        with self.__mgr.get_connection() as conn:
            return self._get_log_id(conn, log_server)

    def __encode_sth(self, audited_sth):
        timestamp = audited_sth.sth.timestamp
        sth = client_pb2.SthResponse()
        sth.CopyFrom(audited_sth.sth)
        sth.ClearField("timestamp")
        audit = client_pb2.AuditInfo()
        audit.CopyFrom(audited_sth.audit)
        return (timestamp, sqlite3.Binary(sth.SerializeToString()),
                sqlite3.Binary(audit.SerializeToString()))

    def __decode_sth(self, sth_row):
        _, _, timestamp, serialized_sth, serialized_audit = sth_row
        audited_sth = client_pb2.AuditedSth()
        audited_sth.sth.ParseFromString(serialized_sth)
        audited_sth.sth.timestamp = timestamp
        audited_sth.audit.ParseFromString(serialized_audit)
        return audited_sth

    # This ignores a duplicate STH even if the audit data differs.
    # TODO(ekasper): add an update method for updating audit data, as needed.
    def store_sth(self, log_server, audited_sth):
        """Store the STH in the database.
        Will store the STH with a unique ID unless an exact copy already exists.
        Note: the fetch_timestamp is time of calling this function, not actual
        fetching timestamp.

        Args:
            log_server: the server name, i.e., the <log_server> path prefix
            audited_sth: a client_pb2.AuditedSth proto
        """
        timestamp, sth_data, audit_info = self.__encode_sth(audited_sth)
        with self.__mgr.get_connection() as conn:
            log_id = self._get_log_id(conn, log_server)
            conn.execute("INSERT INTO sths(log_id, fetch_timestamp, timestamp, "
                         "sth_data, audit_info) VALUES(?, ?, ?, ?, ?)",
                         (log_id, int(time.time()), timestamp, sth_data, audit_info))

    def get_latest_sth(self, log_server):
        row = None
        with self.__mgr.get_connection() as conn:
            log_id = self._get_log_id(conn, log_server)
            res = conn.execute("SELECT * FROM sths WHERE log_id = ? "
                               "ORDER BY timestamp DESC LIMIT 1", (log_id,))
            try:
                row = res.next()
            except StopIteration:
                pass
        if row is not None:
            return self.__decode_sth(row)

    def scan_latest_sth_range(self, log_server, start=0,
                              end=log_db.LogDB.timestamp_max, limit=0):
        sql_limit = -1 if not limit else limit
        with self.__mgr.get_connection() as conn:
            log_id = self._get_log_id(conn, log_server)
            for row in conn.execute(
                "SELECT * FROM sths WHERE log_id = ? "
                "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC "
                "LIMIT ?", (log_id, start, end, sql_limit)):
                yield self.__decode_sth(row)
