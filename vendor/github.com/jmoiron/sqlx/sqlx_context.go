// +build go1.8

package sqlx

import (
	"context"
	"database/sql"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
)

// ConnectContext to a database and verify with a ping.
func ConnectContext(ctx context.Context, driverName, dataSourceName string) (*DB, error) {
	db, err := Open(driverName, dataSourceName)
	if err != nil {
		return db, err
	}
	err = db.PingContext(ctx)
	return db, err
}

// QueryerContext is an interface used by GetContext and SelectContext
type QueryerContext interface {
	QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
	QueryxContext(ctx context.Context, query string, args ...interface{}) (*Rows, error)
	QueryRowxContext(ctx context.Context, query string, args ...interface{}) *Row
}

// PreparerContext is an interface used by PreparexContext.
type PreparerContext interface {
	PrepareContext(ctx context.Context, query string) (*sql.Stmt, error)
}

// ExecerContext is an interface used by MustExecContext and LoadFileContext
type ExecerContext interface {
	ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
}

// ExtContext is a union interface which can bind, query, and exec, with Context
// used by NamedQueryContext and NamedExecContext.
type ExtContext interface {
	binder
	QueryerContext
	ExecerContext
}

// SelectContext executes a query using the provided Queryer, and StructScans
// each row into dest, which must be a slice.  If the slice elements are
// scannable, then the result set must have only one column.  Otherwise,
// StructScan is used. The *sql.Rows are closed automatically.
// Any placeholder parameters are replaced with supplied args.
func SelectContext(ctx context.Context, q QueryerContext, dest interface{}, query string, args ...interface{}) error {
	rows, err := q.QueryxContext(ctx, query, args...)
	if err != nil {
		return err
	}
	// if something happens here, we want to make sure the rows are Closed
	defer rows.Close()
	return scanAll(rows, dest, false)
}

// PreparexContext prepares a statement.
//
// The provided context is used for the preparation of the statement, not for
// the execution of the statement.
func PreparexContext(ctx context.Context, p PreparerContext, query string) (*Stmt, error) {
	s, err := p.PrepareContext(ctx, query)
	if err != nil {
		return nil, err
	}
	return &Stmt{Stmt: s, unsafe: isUnsafe(p), Mapper: mapperFor(p)}, err
}

// GetContext does a QueryRow using the provided Queryer, and scans the
// resulting row to dest.  If dest is scannable, the result must only have one
// column. Otherwise, StructScan is used.  Get will return sql.ErrNoRows like
// row.Scan would. Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func GetContext(ctx context.Context, q QueryerContext, dest interface{}, query string, args ...interface{}) error {
	r := q.QueryRowxContext(ctx, query, args...)
	return r.scanAny(dest, false)
}

// LoadFileContext exec's every statement in a file (as a single call to Exec).
// LoadFileContext may return a nil *sql.Result if errors are encountered
// locating or reading the file at path.  LoadFile reads the entire file into
// memory, so it is not suitable for loading large data dumps, but can be useful
// for initializing schemas or loading indexes.
//
// FIXME: this does not really work with multi-statement files for mattn/go-sqlite3
// or the go-mysql-driver/mysql drivers;  pq seems to be an exception here.  Detecting
// this by requiring something with DriverName() and then attempting to split the
// queries will be difficult to get right, and its current driver-specific behavior
// is deemed at least not complex in its incorrectness.
func LoadFileContext(ctx context.Context, e ExecerContext, path string) (*sql.Result, error) {
	realpath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	contents, err := ioutil.ReadFile(realpath)
	if err != nil {
		return nil, err
	}
	res, err := e.ExecContext(ctx, string(contents))
	return &res, err
}

// MustExecContext execs the query using e and panics if there was an error.
// Any placeholder parameters are replaced with supplied args.
func MustExecContext(ctx context.Context, e ExecerContext, query string, args ...interface{}) sql.Result {
	res, err := e.ExecContext(ctx, query, args...)
	if err != nil {
		panic(err)
	}
	return res
}

// PrepareNamedContext returns an sqlx.NamedStmt
func (db *DB) PrepareNamedContext(ctx context.Context, query string) (*NamedStmt, error) {
	return prepareNamedContext(ctx, db, query)
}

// NamedQueryContext using this DB.
// Any named placeholder parameters are replaced with fields from arg.
func (db *DB) NamedQueryContext(ctx context.Context, query string, arg interface{}) (*Rows, error) {
	return NamedQueryContext(ctx, db, query, arg)
}

// NamedExecContext using this DB.
// Any named placeholder parameters are replaced with fields from arg.
func (db *DB) NamedExecContext(ctx context.Context, query string, arg interface{}) (sql.Result, error) {
	return NamedExecContext(ctx, db, query, arg)
}

// SelectContext using this DB.
// Any placeholder parameters are replaced with supplied args.
func (db *DB) SelectContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return SelectContext(ctx, db, dest, query, args...)
}

// GetContext using this DB.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func (db *DB) GetContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return GetContext(ctx, db, dest, query, args...)
}

// PreparexContext returns an sqlx.Stmt instead of a sql.Stmt.
//
// The provided context is used for the preparation of the statement, not for
// the execution of the statement.
func (db *DB) PreparexContext(ctx context.Context, query string) (*Stmt, error) {
	return PreparexContext(ctx, db, query)
}

// QueryxContext queries the database and returns an *sqlx.Rows.
// Any placeholder parameters are replaced with supplied args.
func (db *DB) QueryxContext(ctx context.Context, query string, args ...interface{}) (*Rows, error) {
	r, err := db.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return &Rows{Rows: r, unsafe: db.unsafe, Mapper: db.Mapper}, err
}

// QueryRowxContext queries the database and returns an *sqlx.Row.
// Any placeholder parameters are replaced with supplied args.
func (db *DB) QueryRowxContext(ctx context.Context, query string, args ...interface{}) *Row {
	rows, err := db.DB.QueryContext(ctx, query, args...)
	return &Row{rows: rows, err: err, unsafe: db.unsafe, Mapper: db.Mapper}
}

// MustBeginTx starts a transaction, and panics on error.  Returns an *sqlx.Tx instead
// of an *sql.Tx.
//
// The provided context is used until the transaction is committed or rolled
// back. If the context is canceled, the sql package will roll back the
// transaction. Tx.Commit will return an error if the context provided to
// MustBeginContext is canceled.
func (db *DB) MustBeginTx(ctx context.Context, opts *sql.TxOptions) *Tx {
	tx, err := db.BeginTxx(ctx, opts)
	if err != nil {
		panic(err)
	}
	return tx
}

// MustExecContext (panic) runs MustExec using this database.
// Any placeholder parameters are replaced with supplied args.
func (db *DB) MustExecContext(ctx context.Context, query string, args ...interface{}) sql.Result {
	return MustExecContext(ctx, db, query, args...)
}

// BeginTxx begins a transaction and returns an *sqlx.Tx instead of an
// *sql.Tx.
//
// The provided context is used until the transaction is committed or rolled
// back. If the context is canceled, the sql package will roll back the
// transaction. Tx.Commit will return an error if the context provided to
// BeginxContext is canceled.
func (db *DB) BeginTxx(ctx context.Context, opts *sql.TxOptions) (*Tx, error) {
	tx, err := db.DB.BeginTx(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &Tx{Tx: tx, driverName: db.driverName, unsafe: db.unsafe, Mapper: db.Mapper}, err
}

// StmtxContext returns a version of the prepared statement which runs within a
// transaction. Provided stmt can be either *sql.Stmt or *sqlx.Stmt.
func (tx *Tx) StmtxContext(ctx context.Context, stmt interface{}) *Stmt {
	var s *sql.Stmt
	switch v := stmt.(type) {
	case Stmt:
		s = v.Stmt
	case *Stmt:
		s = v.Stmt
	case sql.Stmt:
		s = &v
	case *sql.Stmt:
		s = v
	default:
		panic(fmt.Sprintf("non-statement type %v passed to Stmtx", reflect.ValueOf(stmt).Type()))
	}
	return &Stmt{Stmt: tx.StmtContext(ctx, s), Mapper: tx.Mapper}
}

// NamedStmtContext returns a version of the prepared statement which runs
// within a transaction.
func (tx *Tx) NamedStmtContext(ctx context.Context, stmt *NamedStmt) *NamedStmt {
	return &NamedStmt{
		QueryString: stmt.QueryString,
		Params:      stmt.Params,
		Stmt:        tx.StmtxContext(ctx, stmt.Stmt),
	}
}

// MustExecContext runs MustExecContext within a transaction.
// Any placeholder parameters are replaced with supplied args.
func (tx *Tx) MustExecContext(ctx context.Context, query string, args ...interface{}) sql.Result {
	return MustExecContext(ctx, tx, query, args...)
}

// QueryxContext within a transaction and context.
// Any placeholder parameters are replaced with supplied args.
func (tx *Tx) QueryxContext(ctx context.Context, query string, args ...interface{}) (*Rows, error) {
	r, err := tx.Tx.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return &Rows{Rows: r, unsafe: tx.unsafe, Mapper: tx.Mapper}, err
}

// SelectContext within a transaction and context.
// Any placeholder parameters are replaced with supplied args.
func (tx *Tx) SelectContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return SelectContext(ctx, tx, dest, query, args...)
}

// GetContext within a transaction and context.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func (tx *Tx) GetContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return GetContext(ctx, tx, dest, query, args...)
}

// QueryRowxContext within a transaction and context.
// Any placeholder parameters are replaced with supplied args.
func (tx *Tx) QueryRowxContext(ctx context.Context, query string, args ...interface{}) *Row {
	rows, err := tx.Tx.QueryContext(ctx, query, args...)
	return &Row{rows: rows, err: err, unsafe: tx.unsafe, Mapper: tx.Mapper}
}

// NamedExecContext using this Tx.
// Any named placeholder parameters are replaced with fields from arg.
func (tx *Tx) NamedExecContext(ctx context.Context, query string, arg interface{}) (sql.Result, error) {
	return NamedExecContext(ctx, tx, query, arg)
}

// SelectContext using the prepared statement.
// Any placeholder parameters are replaced with supplied args.
func (s *Stmt) SelectContext(ctx context.Context, dest interface{}, args ...interface{}) error {
	return SelectContext(ctx, &qStmt{s}, dest, "", args...)
}

// GetContext using the prepared statement.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func (s *Stmt) GetContext(ctx context.Context, dest interface{}, args ...interface{}) error {
	return GetContext(ctx, &qStmt{s}, dest, "", args...)
}

// MustExecContext (panic) using this statement.  Note that the query portion of
// the error output will be blank, as Stmt does not expose its query.
// Any placeholder parameters are replaced with supplied args.
func (s *Stmt) MustExecContext(ctx context.Context, args ...interface{}) sql.Result {
	return MustExecContext(ctx, &qStmt{s}, "", args...)
}

// QueryRowxContext using this statement.
// Any placeholder parameters are replaced with supplied args.
func (s *Stmt) QueryRowxContext(ctx context.Context, args ...interface{}) *Row {
	qs := &qStmt{s}
	return qs.QueryRowxContext(ctx, "", args...)
}

// QueryxContext using this statement.
// Any placeholder parameters are replaced with supplied args.
func (s *Stmt) QueryxContext(ctx context.Context, args ...interface{}) (*Rows, error) {
	qs := &qStmt{s}
	return qs.QueryxContext(ctx, "", args...)
}

func (q *qStmt) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return q.Stmt.QueryContext(ctx, args...)
}

func (q *qStmt) QueryxContext(ctx context.Context, query string, args ...interface{}) (*Rows, error) {
	r, err := q.Stmt.QueryContext(ctx, args...)
	if err != nil {
		return nil, err
	}
	return &Rows{Rows: r, unsafe: q.Stmt.unsafe, Mapper: q.Stmt.Mapper}, err
}

func (q *qStmt) QueryRowxContext(ctx context.Context, query string, args ...interface{}) *Row {
	rows, err := q.Stmt.QueryContext(ctx, args...)
	return &Row{rows: rows, err: err, unsafe: q.Stmt.unsafe, Mapper: q.Stmt.Mapper}
}

func (q *qStmt) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	return q.Stmt.ExecContext(ctx, args...)
}
