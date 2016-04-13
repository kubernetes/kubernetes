package graphdb

import "database/sql"

// NewSqliteConn opens a connection to a sqlite
// database.
func NewSqliteConn(root string) (*Database, error) {
	conn, err := sql.Open("sqlite3", root)
	if err != nil {
		return nil, err
	}
	return NewDatabase(conn)
}
