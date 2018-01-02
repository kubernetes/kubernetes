package testdb

import (
	"os"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"           // register postgresql driver
	_ "github.com/mattn/go-sqlite3" // register sqlite3 driver
)

const (
	pgTruncateTables = `
CREATE OR REPLACE FUNCTION truncate_tables() RETURNS void AS $$
DECLARE
    statements CURSOR FOR
        SELECT tablename FROM pg_tables
        WHERE tablename != 'goose_db_version'
          AND tableowner = session_user
          AND schemaname = 'public';
BEGIN
    FOR stmt IN statements LOOP
        EXECUTE 'TRUNCATE TABLE ' || quote_ident(stmt.tablename) || ' CASCADE;';
    END LOOP;
END;
$$ LANGUAGE plpgsql;

SELECT truncate_tables();
`

	sqliteTruncateTables = `
DELETE FROM certificates;
DELETE FROM ocsp_responses;
`
)

// PostgreSQLDB returns a PostgreSQL db instance for certdb testing.
func PostgreSQLDB() *sqlx.DB {
	connStr := "dbname=certdb_development sslmode=disable"

	if dbURL := os.Getenv("DATABASE_URL"); dbURL != "" {
		connStr = dbURL
	}

	db, err := sqlx.Open("postgres", connStr)
	if err != nil {
		panic(err)
	}

	Truncate(db)

	return db
}

// SQLiteDB returns a SQLite db instance for certdb testing.
func SQLiteDB(dbpath string) *sqlx.DB {
	db, err := sqlx.Open("sqlite3", dbpath)
	if err != nil {
		panic(err)
	}

	Truncate(db)

	return db
}

// Truncate truncates teh DB
func Truncate(db *sqlx.DB) {
	var sql string
	switch db.DriverName() {
	case "postgres":
		sql = pgTruncateTables
	case "sqlite3":
		sql = sqliteTruncateTables
	default:
		panic("Unknown driver")
	}

	if _, err := db.Exec(sql); err != nil {
		panic(err)
	}
}
