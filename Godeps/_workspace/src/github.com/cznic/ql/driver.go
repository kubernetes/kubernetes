// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// database/sql/driver

package ql

import (
	"bytes"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	_ driver.Conn    = (*driverConn)(nil)
	_ driver.Driver  = (*sqlDriver)(nil)
	_ driver.Execer  = (*driverConn)(nil)
	_ driver.Queryer = (*driverConn)(nil)
	_ driver.Result  = (*driverResult)(nil)
	_ driver.Rows    = (*driverRows)(nil)
	_ driver.Stmt    = (*driverStmt)(nil)
	_ driver.Tx      = (*driverConn)(nil)

	txBegin    = MustCompile("BEGIN TRANSACTION;")
	txCommit   = MustCompile("COMMIT;")
	txRollback = MustCompile("ROLLBACK;")

	errNoResult = errors.New("query statement does not produce a result set (no top level SELECT)")
)

type errList []error

func (e *errList) append(err error) {
	if err != nil {
		*e = append(*e, err)
	}
}

func (e errList) error() error {
	if len(e) == 0 {
		return nil
	}

	return e
}

func (e errList) Error() string {
	a := make([]string, len(e))
	for i, v := range e {
		a[i] = v.Error()
	}
	return strings.Join(a, "\n")
}

func params(args []driver.Value) []interface{} {
	r := make([]interface{}, len(args))
	for i, v := range args {
		r[i] = interface{}(v)
	}
	return r
}

var (
	fileDriver     = &sqlDriver{dbs: map[string]*driverDB{}}
	fileDriverOnce sync.Once
	memDriver      = &sqlDriver{isMem: true, dbs: map[string]*driverDB{}}
	memDriverOnce  sync.Once
)

// RegisterDriver registers a QL database/sql/driver[0] named "ql". The name
// parameter of
//
//	sql.Open("ql", name)
//
// is interpreted as a path name to a named DB file which will be created if
// not present. The underlying QL database data are persisted on db.Close().
// RegisterDriver can be safely called multiple times, it'll register the
// driver only once.
//
// The name argument can be optionally prefixed by "file://". In that case the
// prefix is stripped before interpreting it as a file name.
//
// The name argument can be optionally prefixed by "memory://". In that case
// the prefix is stripped before interpreting it as a name of a memory-only,
// volatile DB.
//
//  [0]: http://golang.org/pkg/database/sql/driver/
func RegisterDriver() {
	fileDriverOnce.Do(func() { sql.Register("ql", fileDriver) })
}

// RegisterMemDriver registers a QL memory database/sql/driver[0] named
// "ql-mem".  The name parameter of
//
//	sql.Open("ql-mem", name)
//
// is interpreted as an unique memory DB name which will be created if not
// present. The underlying QL memory database data are not persisted on
// db.Close(). RegisterMemDriver can be safely called multiple times, it'll
// register the driver only once.
//
//  [0]: http://golang.org/pkg/database/sql/driver/
func RegisterMemDriver() {
	memDriverOnce.Do(func() { sql.Register("ql-mem", memDriver) })
}

type driverDB struct {
	db       *DB
	name     string
	refcount int
}

func newDriverDB(db *DB, name string) *driverDB {
	return &driverDB{db: db, name: name, refcount: 1}
}

// sqlDriver implements the interface required by database/sql/driver.
type sqlDriver struct {
	dbs   map[string]*driverDB
	isMem bool
	mu    sync.Mutex
}

func (d *sqlDriver) lock() func() {
	d.mu.Lock()
	return d.mu.Unlock
}

// Open returns a new connection to the database.  The name is a string in a
// driver-specific format.
//
// Open may return a cached connection (one previously closed), but doing so is
// unnecessary; the sql package maintains a pool of idle connections for
// efficient re-use.
//
// The returned connection is only used by one goroutine at a time.
func (d *sqlDriver) Open(name string) (driver.Conn, error) {
	if d != fileDriver && d != memDriver {
		return nil, fmt.Errorf("open: unexpected/unsupported instance of driver.Driver: %p", d)
	}

	switch {
	case d == fileDriver && strings.HasPrefix(name, "file://"):
		name = name[len("file://"):]
	case d == fileDriver && strings.HasPrefix(name, "memory://"):
		d = memDriver
		name = name[len("memory://"):]
	}
	name = filepath.Clean(name)
	if name == "" || name == "." || name == string(os.PathSeparator) {
		return nil, fmt.Errorf("invalid DB name %q", name)
	}

	defer d.lock()()
	db := d.dbs[name]
	if db == nil {
		var err error
		var db0 *DB
		switch d.isMem {
		case true:
			db0, err = OpenMem()
		default:
			db0, err = OpenFile(name, &Options{CanCreate: true})
		}
		if err != nil {
			return nil, err
		}

		db = newDriverDB(db0, name)
		d.dbs[name] = db
		return newDriverConn(d, db), nil
	}

	db.refcount++
	return newDriverConn(d, db), nil
}

// driverConn is a connection to a database. It is not used concurrently by
// multiple goroutines.
//
// Conn is assumed to be stateful.
type driverConn struct {
	ctx    *TCtx
	db     *driverDB
	driver *sqlDriver
	stop   map[*driverStmt]struct{}
	tnl    int
}

func newDriverConn(d *sqlDriver, ddb *driverDB) driver.Conn {
	r := &driverConn{
		db:     ddb,
		driver: d,
		stop:   map[*driverStmt]struct{}{},
	}
	return r
}

// Prepare returns a prepared statement, bound to this connection.
func (c *driverConn) Prepare(query string) (driver.Stmt, error) {
	list, err := Compile(query)
	if err != nil {
		return nil, err
	}

	s := &driverStmt{conn: c, stmt: list}
	c.stop[s] = struct{}{}
	return s, nil
}

// Close invalidates and potentially stops any current prepared statements and
// transactions, marking this connection as no longer in use.
//
// Because the sql package maintains a free pool of connections and only calls
// Close when there's a surplus of idle connections, it shouldn't be necessary
// for drivers to do their own connection caching.
func (c *driverConn) Close() error {
	var err errList
	for s := range c.stop {
		err.append(s.Close())
	}
	defer c.driver.lock()()
	dbs, name := c.driver.dbs, c.db.name
	v := dbs[name]
	v.refcount--
	if v.refcount == 0 {
		err.append(c.db.db.Close())
		delete(dbs, name)
	}
	return err.error()
}

// Begin starts and returns a new transaction.
func (c *driverConn) Begin() (driver.Tx, error) {
	if c.ctx == nil {
		c.ctx = NewRWCtx()
	}

	if _, _, err := c.db.db.Execute(c.ctx, txBegin); err != nil {
		return nil, err
	}

	c.tnl++
	return c, nil
}

func (c *driverConn) Commit() error {
	if c.tnl == 0 || c.ctx == nil {
		return errCommitNotInTransaction
	}

	if _, _, err := c.db.db.Execute(c.ctx, txCommit); err != nil {
		return err
	}

	c.tnl--
	if c.tnl == 0 {
		c.ctx = nil
	}
	return nil
}

func (c *driverConn) Rollback() error {
	if c.tnl == 0 || c.ctx == nil {
		return errRollbackNotInTransaction
	}

	if _, _, err := c.db.db.Execute(c.ctx, txRollback); err != nil {
		return err
	}

	c.tnl--
	if c.tnl == 0 {
		c.ctx = nil
	}
	return nil
}

// Execer is an optional interface that may be implemented by a Conn.
//
// If a Conn does not implement Execer, the sql package's DB.Exec will first
// prepare a query, execute the statement, and then close the statement.
//
// Exec may return driver.ErrSkip.
func (c *driverConn) Exec(query string, args []driver.Value) (driver.Result, error) {
	list, err := Compile(query)
	if err != nil {
		return nil, err
	}

	return driverExec(c.db, c.ctx, list, args)
}

func driverExec(db *driverDB, ctx *TCtx, list List, args []driver.Value) (driver.Result, error) {
	if _, _, err := db.db.Execute(ctx, list, params(args)...); err != nil {
		return nil, err
	}

	if len(list.l) == 1 {
		switch list.l[0].(type) {
		case *createTableStmt, *dropTableStmt, *alterTableAddStmt,
			*alterTableDropColumnStmt, *truncateTableStmt:
			return driver.ResultNoRows, nil
		}
	}

	r := &driverResult{}
	if ctx != nil {
		r.lastInsertID, r.rowsAffected = ctx.LastInsertID, ctx.RowsAffected
	}
	return r, nil
}

// Queryer is an optional interface that may be implemented by a Conn.
//
// If a Conn does not implement Queryer, the sql package's DB.Query will first
// prepare a query, execute the statement, and then close the statement.
//
// Query may return driver.ErrSkip.
func (c *driverConn) Query(query string, args []driver.Value) (driver.Rows, error) {
	list, err := Compile(query)
	if err != nil {
		return nil, err
	}

	return driverQuery(c.db, c.ctx, list, args)
}

func driverQuery(db *driverDB, ctx *TCtx, list List, args []driver.Value) (driver.Rows, error) {
	rss, _, err := db.db.Execute(ctx, list, params(args)...)
	if err != nil {
		return nil, err
	}

	switch n := len(rss); n {
	case 0:
		return nil, errNoResult
	case 1:
		rs := rss[n-1]
		if x, ok := rss[n-1].(recordset); ok {
			x.tx = ctx
			rs = x
		}
		return newdriverRows(rs), nil
	default:
		return nil, fmt.Errorf("query produced %d result sets, expected only one", n)
	}
}

// driverResult is the result of a query execution.
type driverResult struct {
	lastInsertID int64
	rowsAffected int64
}

// LastInsertId returns the database's auto-generated ID after, for example, an
// INSERT into a table with primary key.
func (r *driverResult) LastInsertId() (int64, error) { // -golint
	return r.lastInsertID, nil
}

// RowsAffected returns the number of rows affected by the query.
func (r *driverResult) RowsAffected() (int64, error) {
	return r.rowsAffected, nil
}

// driverRows is an iterator over an executed query's results.
type driverRows struct {
	rs   Recordset
	done chan int
	rows chan interface{}
}

func newdriverRows(rs Recordset) *driverRows {
	r := &driverRows{
		rs:   rs,
		done: make(chan int),
		rows: make(chan interface{}, 500),
	}
	go func() {
		err := io.EOF
		if e := r.rs.Do(false, func(data []interface{}) (bool, error) {
			select {
			case r.rows <- data:
				return true, nil
			case <-r.done:
				return false, nil
			}
		}); e != nil {
			err = e
		}

		select {
		case r.rows <- err:
		case <-r.done:
		}
	}()
	return r
}

// Columns returns the names of the columns. The number of columns of the
// result is inferred from the length of the slice.  If a particular column
// name isn't known, an empty string should be returned for that entry.
func (r *driverRows) Columns() []string {
	f, _ := r.rs.Fields()
	return f
}

// Close closes the rows iterator.
func (r *driverRows) Close() error {
	close(r.done)
	return nil
}

// Next is called to populate the next row of data into the provided slice. The
// provided slice will be the same size as the Columns() are wide.
//
// The dest slice may be populated only with a driver Value type, but excluding
// string.  All string values must be converted to []byte.
//
// Next should return io.EOF when there are no more rows.
func (r *driverRows) Next(dest []driver.Value) error {
	select {
	case rx := <-r.rows:
		switch x := rx.(type) {
		case error:
			return x
		case []interface{}:
			if g, e := len(x), len(dest); g != e {
				return fmt.Errorf("field count mismatch: got %d, need %d", g, e)
			}

			for i, xi := range x {
				switch v := xi.(type) {
				case nil, int64, float64, bool, []byte, time.Time:
					dest[i] = v
				case complex64, complex128, *big.Int, *big.Rat:
					var buf bytes.Buffer
					fmt.Fprintf(&buf, "%v", v)
					dest[i] = buf.Bytes()
				case int8:
					dest[i] = int64(v)
				case int16:
					dest[i] = int64(v)
				case int32:
					dest[i] = int64(v)
				case int:
					dest[i] = int64(v)
				case uint8:
					dest[i] = int64(v)
				case uint16:
					dest[i] = int64(v)
				case uint32:
					dest[i] = int64(v)
				case uint64:
					dest[i] = int64(v)
				case uint:
					dest[i] = int64(v)
				case time.Duration:
					dest[i] = int64(v)
				case string:
					dest[i] = []byte(v)
				default:
					return fmt.Errorf("internal error 004")
				}
			}
			return nil
		default:
			return fmt.Errorf("internal error 005")
		}
	case <-r.done:
		return io.EOF
	}
}

// driverStmt is a prepared statement. It is bound to a driverConn and not used
// by multiple goroutines concurrently.
type driverStmt struct {
	conn *driverConn
	stmt List
}

// Close closes the statement.
//
// As of Go 1.1, a Stmt will not be closed if it's in use by any queries.
func (s *driverStmt) Close() error {
	delete(s.conn.stop, s)
	return nil
}

// NumInput returns the number of placeholder parameters.
//
// If NumInput returns >= 0, the sql package will sanity check argument counts
// from callers and return errors to the caller before the statement's Exec or
// Query methods are called.
//
// NumInput may also return -1, if the driver doesn't know its number of
// placeholders. In that case, the sql package will not sanity check Exec or
// Query argument counts.
func (s *driverStmt) NumInput() int {
	if x := s.stmt; len(x.l) == 1 {
		return x.params
	}

	return -1
}

// Exec executes a query that doesn't return rows, such as an INSERT or UPDATE.
func (s *driverStmt) Exec(args []driver.Value) (driver.Result, error) {
	c := s.conn
	return driverExec(c.db, c.ctx, s.stmt, args)
}

// Exec executes a query that may return rows, such as a SELECT.
func (s *driverStmt) Query(args []driver.Value) (driver.Rows, error) {
	c := s.conn
	return driverQuery(c.db, c.ctx, s.stmt, args)
}
