import logging
import sqlite3

from ct.client.db import temp_db
from ct.client.db import database
from ct.client.db import sqlite_connection as sqlitecon
from ct.proto import client_pb2

class SQLiteTempDBFactory(object):
    """A database factory that manages mappings from public identifiers
    (log names) to SQLite databases."""
    def __init__(self, connection_manager, database_dir):
        """Initialize the database factory.
        Args:
            connection_manager: an SQLiteConnectionManager object
            database_dir: the directory where the database files reside.
        """
        self.__mgr = connection_manager
        self.__database_dir = database_dir
        # This is the meta-table mapping database IDs to server names.
        with self.__mgr.get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS database_mapping("
                         "id INTEGER PRIMARY KEY, server_name TEXT UNIQUE)")
        self.__tables = ["database_mapping"]

    def __repr__(self):
        return "%r(%r)" % (self.__class__.__name__, self.__mgr)

    def __str__(self):
        return "%s(%s, tables: %s): " % (self.__class__.__name__,
                                         self.__mgr, self.__tables)

    @staticmethod
    def __database_id_to_name(database_id):
        return "db%d" % database_id

    def __get_db_name(self, cursor, log_server):
        cursor.execute("SELECT id from database_mapping WHERE server_name = ?",
                       (log_server,))
        row = cursor.fetchone()
        if row is None:
            raise database.KeyError("No database for log server %s" %
                                    log_server)
        return self.__database_id_to_name(row["id"])

    # TODO(ekasper): add a remove_storage() for removing obsolete data.
    def create_storage(self, log_server):
        """Create a SQLiteTempDB object pointing to the temporary storage of a
        given log server. If the temporary storage does not exist, creates one.
        Args:
        log_server: the server name.
        """
        with self.__mgr.get_connection() as conn:
            cursor = conn.cursor()
            try:
                database_name = self.__get_db_name(cursor, log_server)
            except database.KeyError:
                try:
                    cursor.execute("INSERT INTO database_mapping(server_name) "
                                   "VALUES (?)", (log_server,))
                    database_name = self.__database_id_to_name(cursor.lastrowid)
                except sqlite3.IntegrityError as e:
                    raise database.KeyError("Failed to create a table mapping "
                                            "for server %s: is a concurrent "
                                            "factory running?\n%s" %
                                            (log_server, e))
        return SQLiteTempDB(sqlitecon.SQLiteConnectionManager(
            self.__database_dir + "/" + database_name))

class SQLiteTempDB(temp_db.TempDB):
    """SQLite implementation of TempDB."""
    def __init__(self, connection_manager):
        """Create an SQLiteTempDB object.
        Args:
            connection_manager: and SQLiteConnectionManager object
        """
        self.__mgr = connection_manager
        with self.__mgr.get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS entries("
                         "id INTEGER PRIMARY KEY, entry BLOB)")
        self.__tables = ["entries"]

    def __repr__(self):
        return "%r(%r)" % (self.__class__.__name__, self.__mgr)

    def __str__(self):
        return "%s(%s, tables: %s): " % (self.__class__.__name__,
                                         self.__mgr, self.__tables)

    # Not part of the abstract interface: used to identify the database file
    # we're writing to.
    def db_name(self):
        return self.__mgr.db_name

    def drop_entries(self):
        """Drop all entries."""
        with self.__mgr.get_connection() as conn:
            conn.execute("DELETE FROM entries")

    def store_entries(self, entries):
        """Batch store log entries.
        Args:
            entries: an iterable of (entry_number, client_pb2.EntryResponse)
                     tuples
        """
        with self.__mgr.get_connection() as conn:
            cursor = conn.cursor()
            serialized_entries = map(lambda x: (
                    x[0], sqlite3.Binary(x[1].SerializeToString())), entries)
            try:
                cursor.executemany("INSERT OR REPLACE INTO entries(id, entry) VALUES "
                                   "(?, ?)", serialized_entries)
            except sqlite3.IntegrityError as e:
                raise database.KeyError("Failed to insert entries: an entry "
                                        "with the given sequence number "
                                        "already exists\n%s" % e)

    def scan_entries(self, start, end):
        """Retrieve log entries.
        Args:
            start: index of the first entry to retrieve.
            end: index of the last entry to retrieve.
        Yields:
            client_pb2.EntryResponse protos
        Raises:
            KeyError: an entry with a sequence number in the range does not
                      exist.
        """
        with self.__mgr.get_connection() as conn:
            cursor = conn.cursor()
            next_id = start
            for row in cursor.execute("SELECT id, entry FROM entries WHERE id "
                                      "BETWEEN ? and ? ORDER BY id ASC",
                                      (start, end)):
                if row["id"] != next_id:
                    raise database.KeyError("No such entry: %d" % next_id)
                entry = client_pb2.EntryResponse()
                entry.ParseFromString(str(row["entry"]))
                yield entry
                next_id += 1
            if next_id != end + 1:
                raise database.KeyError("No such entry: %d" % next_id)
