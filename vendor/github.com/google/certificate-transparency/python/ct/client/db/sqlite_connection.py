import sqlite3

from ct.client.db import database

class SQLiteConnection(object):
    """A thin wrapper around sqlite3 Connection for automatically closing the
    connection."""
    def __init__(self, db, keepalive=False):
        """Create a new connection object.
        Args:
            db:        database file, or ":memory:"
            keepalive: If True, don't close upon __exit__'ing.
        Usage:
            with SQLiteConnection(db_name) as conn:
                # conn.execute(...)
                # ...
        """
        self.__keepalive = keepalive
        # The default timeout is 5 seconds.
        # TODO(ekasper): tweak this as needed
        try:
            self.__conn = sqlite3.connect(db, timeout=600)
        # Note: The sqlite3 module does not document its error conditions so
        # it'll probably take a few iterations to get the exceptions right.
        except sqlite3.OperationalError as e:
            raise database.OperationalError(e)
        self.__conn.row_factory = sqlite3.Row

    def __repr__(self):
        return "%r(%r, keepalive=%r)" % (self.__class__.__name__, self.__db,
                                         self.__keepalive)

    def __str__(self):
        return "%s(db: %s, keepalive: %s): " % (self.__class__.__name__,
                                                self.__db, self.__keepalive)

    def __enter__(self):
        """Return the underlying raw sqlite3 Connection object."""
        self.__conn.__enter__()
        return self.__conn

    def __exit__(self, exc_type, exc_value, traceback):
        """Commit or rollback, and close the connection."""
        ret = self.__conn.__exit__(exc_type, exc_value, traceback)
        if not self.__keepalive:
            self.__conn.close()
        return ret

# Currently a very stupid manager that doesn't limit the number of connections -
# connections are simply closed and reopened every time. This could be refined
# by having the manager maintain a connection pool.
class SQLiteConnectionManager(object):
    def __init__(self, db, keepalive=False):
        """Connection manager for a SQLite database.
        Args:
            db:        database file, or ":memory:"
            keepalive: If True, maintains a single open connection.
                       If False, returns a new connection to be created for each
                       call. keepalive=True is not thread-safe.
        """
        self.__db = db
        self.__keepalive = keepalive
        self.__conn = None
        if keepalive:
            self.__conn = SQLiteConnection(self.__db, keepalive=True)

    def __repr__(self):
        return "%r(%r, keepalive=%r)" % (self.__class__.__name__, self.__db,
                                         self.__keepalive)

    def __str__(self):
        return "%s(db: %s, keepalive: %s): " % (self.__class__.__name__,
                                                self.__db, self.__keepalive)

    @property
    def db_name(self):
        return self.__db

    def get_connection(self):
        """In keepalive mode, return the single persistent connection.
        Else return a new connection instance."""
        if self.__keepalive:
            return self.__conn
        else:
            return SQLiteConnection(self.__db, keepalive=False)
