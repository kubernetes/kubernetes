// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"crypto/tls"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var (
	user      string
	pass      string
	prot      string
	addr      string
	dbname    string
	dsn       string
	netAddr   string
	available bool
)

var (
	tDate      = time.Date(2012, 6, 14, 0, 0, 0, 0, time.UTC)
	sDate      = "2012-06-14"
	tDateTime  = time.Date(2011, 11, 20, 21, 27, 37, 0, time.UTC)
	sDateTime  = "2011-11-20 21:27:37"
	tDate0     = time.Time{}
	sDate0     = "0000-00-00"
	sDateTime0 = "0000-00-00 00:00:00"
)

// See https://github.com/go-sql-driver/mysql/wiki/Testing
func init() {
	// get environment variables
	env := func(key, defaultValue string) string {
		if value := os.Getenv(key); value != "" {
			return value
		}
		return defaultValue
	}
	user = env("MYSQL_TEST_USER", "root")
	pass = env("MYSQL_TEST_PASS", "")
	prot = env("MYSQL_TEST_PROT", "tcp")
	addr = env("MYSQL_TEST_ADDR", "localhost:3306")
	dbname = env("MYSQL_TEST_DBNAME", "gotest")
	netAddr = fmt.Sprintf("%s(%s)", prot, addr)
	dsn = fmt.Sprintf("%s:%s@%s/%s?timeout=30s&strict=true", user, pass, netAddr, dbname)
	c, err := net.Dial(prot, addr)
	if err == nil {
		available = true
		c.Close()
	}
}

type DBTest struct {
	*testing.T
	db *sql.DB
}

func runTests(t *testing.T, dsn string, tests ...func(dbt *DBTest)) {
	if !available {
		t.Skipf("MySQL-Server not running on %s", netAddr)
	}

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		t.Fatalf("Error connecting: %s", err.Error())
	}
	defer db.Close()

	db.Exec("DROP TABLE IF EXISTS test")

	dsn2 := dsn + "&interpolateParams=true"
	var db2 *sql.DB
	if _, err := parseDSN(dsn2); err != errInvalidDSNUnsafeCollation {
		db2, err = sql.Open("mysql", dsn2)
		if err != nil {
			t.Fatalf("Error connecting: %s", err.Error())
		}
		defer db2.Close()
	}

	dbt := &DBTest{t, db}
	dbt2 := &DBTest{t, db2}
	for _, test := range tests {
		test(dbt)
		dbt.db.Exec("DROP TABLE IF EXISTS test")
		if db2 != nil {
			test(dbt2)
			dbt2.db.Exec("DROP TABLE IF EXISTS test")
		}
	}
}

func (dbt *DBTest) fail(method, query string, err error) {
	if len(query) > 300 {
		query = "[query too large to print]"
	}
	dbt.Fatalf("Error on %s %s: %s", method, query, err.Error())
}

func (dbt *DBTest) mustExec(query string, args ...interface{}) (res sql.Result) {
	res, err := dbt.db.Exec(query, args...)
	if err != nil {
		dbt.fail("Exec", query, err)
	}
	return res
}

func (dbt *DBTest) mustQuery(query string, args ...interface{}) (rows *sql.Rows) {
	rows, err := dbt.db.Query(query, args...)
	if err != nil {
		dbt.fail("Query", query, err)
	}
	return rows
}

func TestEmptyQuery(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		// just a comment, no query
		rows := dbt.mustQuery("--")
		// will hang before #255
		if rows.Next() {
			dbt.Errorf("Next on rows must be false")
		}
	})
}

func TestCRUD(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		// Create Table
		dbt.mustExec("CREATE TABLE test (value BOOL)")

		// Test for unexpected data
		var out bool
		rows := dbt.mustQuery("SELECT * FROM test")
		if rows.Next() {
			dbt.Error("unexpected data in empty table")
		}

		// Create Data
		res := dbt.mustExec("INSERT INTO test VALUES (1)")
		count, err := res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 1 {
			dbt.Fatalf("Expected 1 affected row, got %d", count)
		}

		id, err := res.LastInsertId()
		if err != nil {
			dbt.Fatalf("res.LastInsertId() returned error: %s", err.Error())
		}
		if id != 0 {
			dbt.Fatalf("Expected InsertID 0, got %d", id)
		}

		// Read
		rows = dbt.mustQuery("SELECT value FROM test")
		if rows.Next() {
			rows.Scan(&out)
			if true != out {
				dbt.Errorf("true != %t", out)
			}

			if rows.Next() {
				dbt.Error("unexpected data")
			}
		} else {
			dbt.Error("no data")
		}

		// Update
		res = dbt.mustExec("UPDATE test SET value = ? WHERE value = ?", false, true)
		count, err = res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 1 {
			dbt.Fatalf("Expected 1 affected row, got %d", count)
		}

		// Check Update
		rows = dbt.mustQuery("SELECT value FROM test")
		if rows.Next() {
			rows.Scan(&out)
			if false != out {
				dbt.Errorf("false != %t", out)
			}

			if rows.Next() {
				dbt.Error("unexpected data")
			}
		} else {
			dbt.Error("no data")
		}

		// Delete
		res = dbt.mustExec("DELETE FROM test WHERE value = ?", false)
		count, err = res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 1 {
			dbt.Fatalf("Expected 1 affected row, got %d", count)
		}

		// Check for unexpected rows
		res = dbt.mustExec("DELETE FROM test")
		count, err = res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 0 {
			dbt.Fatalf("Expected 0 affected row, got %d", count)
		}
	})
}

func TestInt(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		types := [5]string{"TINYINT", "SMALLINT", "MEDIUMINT", "INT", "BIGINT"}
		in := int64(42)
		var out int64
		var rows *sql.Rows

		// SIGNED
		for _, v := range types {
			dbt.mustExec("CREATE TABLE test (value " + v + ")")

			dbt.mustExec("INSERT INTO test VALUES (?)", in)

			rows = dbt.mustQuery("SELECT value FROM test")
			if rows.Next() {
				rows.Scan(&out)
				if in != out {
					dbt.Errorf("%s: %d != %d", v, in, out)
				}
			} else {
				dbt.Errorf("%s: no data", v)
			}

			dbt.mustExec("DROP TABLE IF EXISTS test")
		}

		// UNSIGNED ZEROFILL
		for _, v := range types {
			dbt.mustExec("CREATE TABLE test (value " + v + " ZEROFILL)")

			dbt.mustExec("INSERT INTO test VALUES (?)", in)

			rows = dbt.mustQuery("SELECT value FROM test")
			if rows.Next() {
				rows.Scan(&out)
				if in != out {
					dbt.Errorf("%s ZEROFILL: %d != %d", v, in, out)
				}
			} else {
				dbt.Errorf("%s ZEROFILL: no data", v)
			}

			dbt.mustExec("DROP TABLE IF EXISTS test")
		}
	})
}

func TestFloat(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		types := [2]string{"FLOAT", "DOUBLE"}
		in := float32(42.23)
		var out float32
		var rows *sql.Rows
		for _, v := range types {
			dbt.mustExec("CREATE TABLE test (value " + v + ")")
			dbt.mustExec("INSERT INTO test VALUES (?)", in)
			rows = dbt.mustQuery("SELECT value FROM test")
			if rows.Next() {
				rows.Scan(&out)
				if in != out {
					dbt.Errorf("%s: %g != %g", v, in, out)
				}
			} else {
				dbt.Errorf("%s: no data", v)
			}
			dbt.mustExec("DROP TABLE IF EXISTS test")
		}
	})
}

func TestString(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		types := [6]string{"CHAR(255)", "VARCHAR(255)", "TINYTEXT", "TEXT", "MEDIUMTEXT", "LONGTEXT"}
		in := "κόσμε üöäßñóùéàâÿœ'îë Árvíztűrő いろはにほへとちりぬるを イロハニホヘト דג סקרן чащах  น่าฟังเอย"
		var out string
		var rows *sql.Rows

		for _, v := range types {
			dbt.mustExec("CREATE TABLE test (value " + v + ") CHARACTER SET utf8")

			dbt.mustExec("INSERT INTO test VALUES (?)", in)

			rows = dbt.mustQuery("SELECT value FROM test")
			if rows.Next() {
				rows.Scan(&out)
				if in != out {
					dbt.Errorf("%s: %s != %s", v, in, out)
				}
			} else {
				dbt.Errorf("%s: no data", v)
			}

			dbt.mustExec("DROP TABLE IF EXISTS test")
		}

		// BLOB
		dbt.mustExec("CREATE TABLE test (id int, value BLOB) CHARACTER SET utf8")

		id := 2
		in = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, " +
			"sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, " +
			"sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. " +
			"Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. " +
			"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, " +
			"sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, " +
			"sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. " +
			"Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
		dbt.mustExec("INSERT INTO test VALUES (?, ?)", id, in)

		err := dbt.db.QueryRow("SELECT value FROM test WHERE id = ?", id).Scan(&out)
		if err != nil {
			dbt.Fatalf("Error on BLOB-Query: %s", err.Error())
		} else if out != in {
			dbt.Errorf("BLOB: %s != %s", in, out)
		}
	})
}

type timeTests struct {
	dbtype  string
	tlayout string
	tests   []timeTest
}

type timeTest struct {
	s string // leading "!": do not use t as value in queries
	t time.Time
}

type timeMode byte

func (t timeMode) String() string {
	switch t {
	case binaryString:
		return "binary:string"
	case binaryTime:
		return "binary:time.Time"
	case textString:
		return "text:string"
	}
	panic("unsupported timeMode")
}

func (t timeMode) Binary() bool {
	switch t {
	case binaryString, binaryTime:
		return true
	}
	return false
}

const (
	binaryString timeMode = iota
	binaryTime
	textString
)

func (t timeTest) genQuery(dbtype string, mode timeMode) string {
	var inner string
	if mode.Binary() {
		inner = "?"
	} else {
		inner = `"%s"`
	}
	return `SELECT cast(` + inner + ` as ` + dbtype + `)`
}

func (t timeTest) run(dbt *DBTest, dbtype, tlayout string, mode timeMode) {
	var rows *sql.Rows
	query := t.genQuery(dbtype, mode)
	switch mode {
	case binaryString:
		rows = dbt.mustQuery(query, t.s)
	case binaryTime:
		rows = dbt.mustQuery(query, t.t)
	case textString:
		query = fmt.Sprintf(query, t.s)
		rows = dbt.mustQuery(query)
	default:
		panic("unsupported mode")
	}
	defer rows.Close()
	var err error
	if !rows.Next() {
		err = rows.Err()
		if err == nil {
			err = fmt.Errorf("no data")
		}
		dbt.Errorf("%s [%s]: %s", dbtype, mode, err)
		return
	}
	var dst interface{}
	err = rows.Scan(&dst)
	if err != nil {
		dbt.Errorf("%s [%s]: %s", dbtype, mode, err)
		return
	}
	switch val := dst.(type) {
	case []uint8:
		str := string(val)
		if str == t.s {
			return
		}
		if mode.Binary() && dbtype == "DATETIME" && len(str) == 26 && str[:19] == t.s {
			// a fix mainly for TravisCI:
			// accept full microsecond resolution in result for DATETIME columns
			// where the binary protocol was used
			return
		}
		dbt.Errorf("%s [%s] to string: expected %q, got %q",
			dbtype, mode,
			t.s, str,
		)
	case time.Time:
		if val == t.t {
			return
		}
		dbt.Errorf("%s [%s] to string: expected %q, got %q",
			dbtype, mode,
			t.s, val.Format(tlayout),
		)
	default:
		fmt.Printf("%#v\n", []interface{}{dbtype, tlayout, mode, t.s, t.t})
		dbt.Errorf("%s [%s]: unhandled type %T (is '%v')",
			dbtype, mode,
			val, val,
		)
	}
}

func TestDateTime(t *testing.T) {
	afterTime := func(t time.Time, d string) time.Time {
		dur, err := time.ParseDuration(d)
		if err != nil {
			panic(err)
		}
		return t.Add(dur)
	}
	// NOTE: MySQL rounds DATETIME(x) up - but that's not included in the tests
	format := "2006-01-02 15:04:05.999999"
	t0 := time.Time{}
	tstr0 := "0000-00-00 00:00:00.000000"
	testcases := []timeTests{
		{"DATE", format[:10], []timeTest{
			{t: time.Date(2011, 11, 20, 0, 0, 0, 0, time.UTC)},
			{t: t0, s: tstr0[:10]},
		}},
		{"DATETIME", format[:19], []timeTest{
			{t: time.Date(2011, 11, 20, 21, 27, 37, 0, time.UTC)},
			{t: t0, s: tstr0[:19]},
		}},
		{"DATETIME(0)", format[:21], []timeTest{
			{t: time.Date(2011, 11, 20, 21, 27, 37, 0, time.UTC)},
			{t: t0, s: tstr0[:19]},
		}},
		{"DATETIME(1)", format[:21], []timeTest{
			{t: time.Date(2011, 11, 20, 21, 27, 37, 100000000, time.UTC)},
			{t: t0, s: tstr0[:21]},
		}},
		{"DATETIME(6)", format, []timeTest{
			{t: time.Date(2011, 11, 20, 21, 27, 37, 123456000, time.UTC)},
			{t: t0, s: tstr0},
		}},
		{"TIME", format[11:19], []timeTest{
			{t: afterTime(t0, "12345s")},
			{s: "!-12:34:56"},
			{s: "!-838:59:59"},
			{s: "!838:59:59"},
			{t: t0, s: tstr0[11:19]},
		}},
		{"TIME(0)", format[11:19], []timeTest{
			{t: afterTime(t0, "12345s")},
			{s: "!-12:34:56"},
			{s: "!-838:59:59"},
			{s: "!838:59:59"},
			{t: t0, s: tstr0[11:19]},
		}},
		{"TIME(1)", format[11:21], []timeTest{
			{t: afterTime(t0, "12345600ms")},
			{s: "!-12:34:56.7"},
			{s: "!-838:59:58.9"},
			{s: "!838:59:58.9"},
			{t: t0, s: tstr0[11:21]},
		}},
		{"TIME(6)", format[11:], []timeTest{
			{t: afterTime(t0, "1234567890123000ns")},
			{s: "!-12:34:56.789012"},
			{s: "!-838:59:58.999999"},
			{s: "!838:59:58.999999"},
			{t: t0, s: tstr0[11:]},
		}},
	}
	dsns := []string{
		dsn + "&parseTime=true",
		dsn + "&parseTime=false",
	}
	for _, testdsn := range dsns {
		runTests(t, testdsn, func(dbt *DBTest) {
			microsecsSupported := false
			zeroDateSupported := false
			var rows *sql.Rows
			var err error
			rows, err = dbt.db.Query(`SELECT cast("00:00:00.1" as TIME(1)) = "00:00:00.1"`)
			if err == nil {
				rows.Scan(&microsecsSupported)
				rows.Close()
			}
			rows, err = dbt.db.Query(`SELECT cast("0000-00-00" as DATE) = "0000-00-00"`)
			if err == nil {
				rows.Scan(&zeroDateSupported)
				rows.Close()
			}
			for _, setups := range testcases {
				if t := setups.dbtype; !microsecsSupported && t[len(t)-1:] == ")" {
					// skip fractional second tests if unsupported by server
					continue
				}
				for _, setup := range setups.tests {
					allowBinTime := true
					if setup.s == "" {
						// fill time string whereever Go can reliable produce it
						setup.s = setup.t.Format(setups.tlayout)
					} else if setup.s[0] == '!' {
						// skip tests using setup.t as source in queries
						allowBinTime = false
						// fix setup.s - remove the "!"
						setup.s = setup.s[1:]
					}
					if !zeroDateSupported && setup.s == tstr0[:len(setup.s)] {
						// skip disallowed 0000-00-00 date
						continue
					}
					setup.run(dbt, setups.dbtype, setups.tlayout, textString)
					setup.run(dbt, setups.dbtype, setups.tlayout, binaryString)
					if allowBinTime {
						setup.run(dbt, setups.dbtype, setups.tlayout, binaryTime)
					}
				}
			}
		})
	}
}

func TestTimestampMicros(t *testing.T) {
	format := "2006-01-02 15:04:05.999999"
	f0 := format[:19]
	f1 := format[:21]
	f6 := format[:26]
	runTests(t, dsn, func(dbt *DBTest) {
		// check if microseconds are supported.
		// Do not use timestamp(x) for that check - before 5.5.6, x would mean display width
		// and not precision.
		// Se last paragraph at http://dev.mysql.com/doc/refman/5.6/en/fractional-seconds.html
		microsecsSupported := false
		if rows, err := dbt.db.Query(`SELECT cast("00:00:00.1" as TIME(1)) = "00:00:00.1"`); err == nil {
			rows.Scan(&microsecsSupported)
			rows.Close()
		}
		if !microsecsSupported {
			// skip test
			return
		}
		_, err := dbt.db.Exec(`
			CREATE TABLE test (
				value0 TIMESTAMP NOT NULL DEFAULT '` + f0 + `',
				value1 TIMESTAMP(1) NOT NULL DEFAULT '` + f1 + `',
				value6 TIMESTAMP(6) NOT NULL DEFAULT '` + f6 + `'
			)`,
		)
		if err != nil {
			dbt.Error(err)
		}
		defer dbt.mustExec("DROP TABLE IF EXISTS test")
		dbt.mustExec("INSERT INTO test SET value0=?, value1=?, value6=?", f0, f1, f6)
		var res0, res1, res6 string
		rows := dbt.mustQuery("SELECT * FROM test")
		if !rows.Next() {
			dbt.Errorf("test contained no selectable values")
		}
		err = rows.Scan(&res0, &res1, &res6)
		if err != nil {
			dbt.Error(err)
		}
		if res0 != f0 {
			dbt.Errorf("expected %q, got %q", f0, res0)
		}
		if res1 != f1 {
			dbt.Errorf("expected %q, got %q", f1, res1)
		}
		if res6 != f6 {
			dbt.Errorf("expected %q, got %q", f6, res6)
		}
	})
}

func TestNULL(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		nullStmt, err := dbt.db.Prepare("SELECT NULL")
		if err != nil {
			dbt.Fatal(err)
		}
		defer nullStmt.Close()

		nonNullStmt, err := dbt.db.Prepare("SELECT 1")
		if err != nil {
			dbt.Fatal(err)
		}
		defer nonNullStmt.Close()

		// NullBool
		var nb sql.NullBool
		// Invalid
		if err = nullStmt.QueryRow().Scan(&nb); err != nil {
			dbt.Fatal(err)
		}
		if nb.Valid {
			dbt.Error("Valid NullBool which should be invalid")
		}
		// Valid
		if err = nonNullStmt.QueryRow().Scan(&nb); err != nil {
			dbt.Fatal(err)
		}
		if !nb.Valid {
			dbt.Error("Invalid NullBool which should be valid")
		} else if nb.Bool != true {
			dbt.Errorf("Unexpected NullBool value: %t (should be true)", nb.Bool)
		}

		// NullFloat64
		var nf sql.NullFloat64
		// Invalid
		if err = nullStmt.QueryRow().Scan(&nf); err != nil {
			dbt.Fatal(err)
		}
		if nf.Valid {
			dbt.Error("Valid NullFloat64 which should be invalid")
		}
		// Valid
		if err = nonNullStmt.QueryRow().Scan(&nf); err != nil {
			dbt.Fatal(err)
		}
		if !nf.Valid {
			dbt.Error("Invalid NullFloat64 which should be valid")
		} else if nf.Float64 != float64(1) {
			dbt.Errorf("Unexpected NullFloat64 value: %f (should be 1.0)", nf.Float64)
		}

		// NullInt64
		var ni sql.NullInt64
		// Invalid
		if err = nullStmt.QueryRow().Scan(&ni); err != nil {
			dbt.Fatal(err)
		}
		if ni.Valid {
			dbt.Error("Valid NullInt64 which should be invalid")
		}
		// Valid
		if err = nonNullStmt.QueryRow().Scan(&ni); err != nil {
			dbt.Fatal(err)
		}
		if !ni.Valid {
			dbt.Error("Invalid NullInt64 which should be valid")
		} else if ni.Int64 != int64(1) {
			dbt.Errorf("Unexpected NullInt64 value: %d (should be 1)", ni.Int64)
		}

		// NullString
		var ns sql.NullString
		// Invalid
		if err = nullStmt.QueryRow().Scan(&ns); err != nil {
			dbt.Fatal(err)
		}
		if ns.Valid {
			dbt.Error("Valid NullString which should be invalid")
		}
		// Valid
		if err = nonNullStmt.QueryRow().Scan(&ns); err != nil {
			dbt.Fatal(err)
		}
		if !ns.Valid {
			dbt.Error("Invalid NullString which should be valid")
		} else if ns.String != `1` {
			dbt.Error("Unexpected NullString value:" + ns.String + " (should be `1`)")
		}

		// nil-bytes
		var b []byte
		// Read nil
		if err = nullStmt.QueryRow().Scan(&b); err != nil {
			dbt.Fatal(err)
		}
		if b != nil {
			dbt.Error("Non-nil []byte wich should be nil")
		}
		// Read non-nil
		if err = nonNullStmt.QueryRow().Scan(&b); err != nil {
			dbt.Fatal(err)
		}
		if b == nil {
			dbt.Error("Nil []byte wich should be non-nil")
		}
		// Insert nil
		b = nil
		success := false
		if err = dbt.db.QueryRow("SELECT ? IS NULL", b).Scan(&success); err != nil {
			dbt.Fatal(err)
		}
		if !success {
			dbt.Error("Inserting []byte(nil) as NULL failed")
		}
		// Check input==output with input==nil
		b = nil
		if err = dbt.db.QueryRow("SELECT ?", b).Scan(&b); err != nil {
			dbt.Fatal(err)
		}
		if b != nil {
			dbt.Error("Non-nil echo from nil input")
		}
		// Check input==output with input!=nil
		b = []byte("")
		if err = dbt.db.QueryRow("SELECT ?", b).Scan(&b); err != nil {
			dbt.Fatal(err)
		}
		if b == nil {
			dbt.Error("nil echo from non-nil input")
		}

		// Insert NULL
		dbt.mustExec("CREATE TABLE test (dummmy1 int, value int, dummy2 int)")

		dbt.mustExec("INSERT INTO test VALUES (?, ?, ?)", 1, nil, 2)

		var out interface{}
		rows := dbt.mustQuery("SELECT * FROM test")
		if rows.Next() {
			rows.Scan(&out)
			if out != nil {
				dbt.Errorf("%v != nil", out)
			}
		} else {
			dbt.Error("no data")
		}
	})
}

func TestUint64(t *testing.T) {
	const (
		u0    = uint64(0)
		uall  = ^u0
		uhigh = uall >> 1
		utop  = ^uhigh
		s0    = int64(0)
		sall  = ^s0
		shigh = int64(uhigh)
		stop  = ^shigh
	)
	runTests(t, dsn, func(dbt *DBTest) {
		stmt, err := dbt.db.Prepare(`SELECT ?, ?, ? ,?, ?, ?, ?, ?`)
		if err != nil {
			dbt.Fatal(err)
		}
		defer stmt.Close()
		row := stmt.QueryRow(
			u0, uhigh, utop, uall,
			s0, shigh, stop, sall,
		)

		var ua, ub, uc, ud uint64
		var sa, sb, sc, sd int64

		err = row.Scan(&ua, &ub, &uc, &ud, &sa, &sb, &sc, &sd)
		if err != nil {
			dbt.Fatal(err)
		}
		switch {
		case ua != u0,
			ub != uhigh,
			uc != utop,
			ud != uall,
			sa != s0,
			sb != shigh,
			sc != stop,
			sd != sall:
			dbt.Fatal("Unexpected result value")
		}
	})
}

func TestLongData(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		var maxAllowedPacketSize int
		err := dbt.db.QueryRow("select @@max_allowed_packet").Scan(&maxAllowedPacketSize)
		if err != nil {
			dbt.Fatal(err)
		}
		maxAllowedPacketSize--

		// don't get too ambitious
		if maxAllowedPacketSize > 1<<25 {
			maxAllowedPacketSize = 1 << 25
		}

		dbt.mustExec("CREATE TABLE test (value LONGBLOB)")

		in := strings.Repeat(`a`, maxAllowedPacketSize+1)
		var out string
		var rows *sql.Rows

		// Long text data
		const nonDataQueryLen = 28 // length query w/o value
		inS := in[:maxAllowedPacketSize-nonDataQueryLen]
		dbt.mustExec("INSERT INTO test VALUES('" + inS + "')")
		rows = dbt.mustQuery("SELECT value FROM test")
		if rows.Next() {
			rows.Scan(&out)
			if inS != out {
				dbt.Fatalf("LONGBLOB: length in: %d, length out: %d", len(inS), len(out))
			}
			if rows.Next() {
				dbt.Error("LONGBLOB: unexpexted row")
			}
		} else {
			dbt.Fatalf("LONGBLOB: no data")
		}

		// Empty table
		dbt.mustExec("TRUNCATE TABLE test")

		// Long binary data
		dbt.mustExec("INSERT INTO test VALUES(?)", in)
		rows = dbt.mustQuery("SELECT value FROM test WHERE 1=?", 1)
		if rows.Next() {
			rows.Scan(&out)
			if in != out {
				dbt.Fatalf("LONGBLOB: length in: %d, length out: %d", len(in), len(out))
			}
			if rows.Next() {
				dbt.Error("LONGBLOB: unexpexted row")
			}
		} else {
			if err = rows.Err(); err != nil {
				dbt.Fatalf("LONGBLOB: no data (err: %s)", err.Error())
			} else {
				dbt.Fatal("LONGBLOB: no data (err: <nil>)")
			}
		}
	})
}

func TestLoadData(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		verifyLoadDataResult := func() {
			rows, err := dbt.db.Query("SELECT * FROM test")
			if err != nil {
				dbt.Fatal(err.Error())
			}

			i := 0
			values := [4]string{
				"a string",
				"a string containing a \t",
				"a string containing a \n",
				"a string containing both \t\n",
			}

			var id int
			var value string

			for rows.Next() {
				i++
				err = rows.Scan(&id, &value)
				if err != nil {
					dbt.Fatal(err.Error())
				}
				if i != id {
					dbt.Fatalf("%d != %d", i, id)
				}
				if values[i-1] != value {
					dbt.Fatalf("%q != %q", values[i-1], value)
				}
			}
			err = rows.Err()
			if err != nil {
				dbt.Fatal(err.Error())
			}

			if i != 4 {
				dbt.Fatalf("Rows count mismatch. Got %d, want 4", i)
			}
		}
		file, err := ioutil.TempFile("", "gotest")
		defer os.Remove(file.Name())
		if err != nil {
			dbt.Fatal(err)
		}
		file.WriteString("1\ta string\n2\ta string containing a \\t\n3\ta string containing a \\n\n4\ta string containing both \\t\\n\n")
		file.Close()

		dbt.db.Exec("DROP TABLE IF EXISTS test")
		dbt.mustExec("CREATE TABLE test (id INT NOT NULL PRIMARY KEY, value TEXT NOT NULL) CHARACTER SET utf8")

		// Local File
		RegisterLocalFile(file.Name())
		dbt.mustExec(fmt.Sprintf("LOAD DATA LOCAL INFILE %q INTO TABLE test", file.Name()))
		verifyLoadDataResult()
		// negative test
		_, err = dbt.db.Exec("LOAD DATA LOCAL INFILE 'doesnotexist' INTO TABLE test")
		if err == nil {
			dbt.Fatal("Load non-existent file didn't fail")
		} else if err.Error() != "Local File 'doesnotexist' is not registered. Use the DSN parameter 'allowAllFiles=true' to allow all files" {
			dbt.Fatal(err.Error())
		}

		// Empty table
		dbt.mustExec("TRUNCATE TABLE test")

		// Reader
		RegisterReaderHandler("test", func() io.Reader {
			file, err = os.Open(file.Name())
			if err != nil {
				dbt.Fatal(err)
			}
			return file
		})
		dbt.mustExec("LOAD DATA LOCAL INFILE 'Reader::test' INTO TABLE test")
		verifyLoadDataResult()
		// negative test
		_, err = dbt.db.Exec("LOAD DATA LOCAL INFILE 'Reader::doesnotexist' INTO TABLE test")
		if err == nil {
			dbt.Fatal("Load non-existent Reader didn't fail")
		} else if err.Error() != "Reader 'doesnotexist' is not registered" {
			dbt.Fatal(err.Error())
		}
	})
}

func TestFoundRows(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		dbt.mustExec("CREATE TABLE test (id INT NOT NULL ,data INT NOT NULL)")
		dbt.mustExec("INSERT INTO test (id, data) VALUES (0, 0),(0, 0),(1, 0),(1, 0),(1, 1)")

		res := dbt.mustExec("UPDATE test SET data = 1 WHERE id = 0")
		count, err := res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 2 {
			dbt.Fatalf("Expected 2 affected rows, got %d", count)
		}
		res = dbt.mustExec("UPDATE test SET data = 1 WHERE id = 1")
		count, err = res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 2 {
			dbt.Fatalf("Expected 2 affected rows, got %d", count)
		}
	})
	runTests(t, dsn+"&clientFoundRows=true", func(dbt *DBTest) {
		dbt.mustExec("CREATE TABLE test (id INT NOT NULL ,data INT NOT NULL)")
		dbt.mustExec("INSERT INTO test (id, data) VALUES (0, 0),(0, 0),(1, 0),(1, 0),(1, 1)")

		res := dbt.mustExec("UPDATE test SET data = 1 WHERE id = 0")
		count, err := res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 2 {
			dbt.Fatalf("Expected 2 matched rows, got %d", count)
		}
		res = dbt.mustExec("UPDATE test SET data = 1 WHERE id = 1")
		count, err = res.RowsAffected()
		if err != nil {
			dbt.Fatalf("res.RowsAffected() returned error: %s", err.Error())
		}
		if count != 3 {
			dbt.Fatalf("Expected 3 matched rows, got %d", count)
		}
	})
}

func TestStrict(t *testing.T) {
	// ALLOW_INVALID_DATES to get rid of stricter modes - we want to test for warnings, not errors
	relaxedDsn := dsn + "&sql_mode=ALLOW_INVALID_DATES"
	// make sure the MySQL version is recent enough with a separate connection
	// before running the test
	conn, err := MySQLDriver{}.Open(relaxedDsn)
	if conn != nil {
		conn.Close()
	}
	if me, ok := err.(*MySQLError); ok && me.Number == 1231 {
		// Error 1231: Variable 'sql_mode' can't be set to the value of 'ALLOW_INVALID_DATES'
		// => skip test, MySQL server version is too old
		return
	}
	runTests(t, relaxedDsn, func(dbt *DBTest) {
		dbt.mustExec("CREATE TABLE test (a TINYINT NOT NULL, b CHAR(4))")

		var queries = [...]struct {
			in    string
			codes []string
		}{
			{"DROP TABLE IF EXISTS no_such_table", []string{"1051"}},
			{"INSERT INTO test VALUES(10,'mysql'),(NULL,'test'),(300,'Open Source')", []string{"1265", "1048", "1264", "1265"}},
		}
		var err error

		var checkWarnings = func(err error, mode string, idx int) {
			if err == nil {
				dbt.Errorf("Expected STRICT error on query [%s] %s", mode, queries[idx].in)
			}

			if warnings, ok := err.(MySQLWarnings); ok {
				var codes = make([]string, len(warnings))
				for i := range warnings {
					codes[i] = warnings[i].Code
				}
				if len(codes) != len(queries[idx].codes) {
					dbt.Errorf("Unexpected STRICT error count on query [%s] %s: Wanted %v, Got %v", mode, queries[idx].in, queries[idx].codes, codes)
				}

				for i := range warnings {
					if codes[i] != queries[idx].codes[i] {
						dbt.Errorf("Unexpected STRICT error codes on query [%s] %s: Wanted %v, Got %v", mode, queries[idx].in, queries[idx].codes, codes)
						return
					}
				}

			} else {
				dbt.Errorf("Unexpected error on query [%s] %s: %s", mode, queries[idx].in, err.Error())
			}
		}

		// text protocol
		for i := range queries {
			_, err = dbt.db.Exec(queries[i].in)
			checkWarnings(err, "text", i)
		}

		var stmt *sql.Stmt

		// binary protocol
		for i := range queries {
			stmt, err = dbt.db.Prepare(queries[i].in)
			if err != nil {
				dbt.Errorf("Error on preparing query %s: %s", queries[i].in, err.Error())
			}

			_, err = stmt.Exec()
			checkWarnings(err, "binary", i)

			err = stmt.Close()
			if err != nil {
				dbt.Errorf("Error on closing stmt for query %s: %s", queries[i].in, err.Error())
			}
		}
	})
}

func TestTLS(t *testing.T) {
	tlsTest := func(dbt *DBTest) {
		if err := dbt.db.Ping(); err != nil {
			if err == ErrNoTLS {
				dbt.Skip("Server does not support TLS")
			} else {
				dbt.Fatalf("Error on Ping: %s", err.Error())
			}
		}

		rows := dbt.mustQuery("SHOW STATUS LIKE 'Ssl_cipher'")

		var variable, value *sql.RawBytes
		for rows.Next() {
			if err := rows.Scan(&variable, &value); err != nil {
				dbt.Fatal(err.Error())
			}

			if value == nil {
				dbt.Fatal("No Cipher")
			}
		}
	}

	runTests(t, dsn+"&tls=skip-verify", tlsTest)

	// Verify that registering / using a custom cfg works
	RegisterTLSConfig("custom-skip-verify", &tls.Config{
		InsecureSkipVerify: true,
	})
	runTests(t, dsn+"&tls=custom-skip-verify", tlsTest)
}

func TestReuseClosedConnection(t *testing.T) {
	// this test does not use sql.database, it uses the driver directly
	if !available {
		t.Skipf("MySQL-Server not running on %s", netAddr)
	}

	md := &MySQLDriver{}
	conn, err := md.Open(dsn)
	if err != nil {
		t.Fatalf("Error connecting: %s", err.Error())
	}
	stmt, err := conn.Prepare("DO 1")
	if err != nil {
		t.Fatalf("Error preparing statement: %s", err.Error())
	}
	_, err = stmt.Exec(nil)
	if err != nil {
		t.Fatalf("Error executing statement: %s", err.Error())
	}
	err = conn.Close()
	if err != nil {
		t.Fatalf("Error closing connection: %s", err.Error())
	}

	defer func() {
		if err := recover(); err != nil {
			t.Errorf("Panic after reusing a closed connection: %v", err)
		}
	}()
	_, err = stmt.Exec(nil)
	if err != nil && err != driver.ErrBadConn {
		t.Errorf("Unexpected error '%s', expected '%s'",
			err.Error(), driver.ErrBadConn.Error())
	}
}

func TestCharset(t *testing.T) {
	if !available {
		t.Skipf("MySQL-Server not running on %s", netAddr)
	}

	mustSetCharset := func(charsetParam, expected string) {
		runTests(t, dsn+"&"+charsetParam, func(dbt *DBTest) {
			rows := dbt.mustQuery("SELECT @@character_set_connection")
			defer rows.Close()

			if !rows.Next() {
				dbt.Fatalf("Error getting connection charset: %s", rows.Err())
			}

			var got string
			rows.Scan(&got)

			if got != expected {
				dbt.Fatalf("Expected connection charset %s but got %s", expected, got)
			}
		})
	}

	// non utf8 test
	mustSetCharset("charset=ascii", "ascii")

	// when the first charset is invalid, use the second
	mustSetCharset("charset=none,utf8", "utf8")

	// when the first charset is valid, use it
	mustSetCharset("charset=ascii,utf8", "ascii")
	mustSetCharset("charset=utf8,ascii", "utf8")
}

func TestFailingCharset(t *testing.T) {
	runTests(t, dsn+"&charset=none", func(dbt *DBTest) {
		// run query to really establish connection...
		_, err := dbt.db.Exec("SELECT 1")
		if err == nil {
			dbt.db.Close()
			t.Fatalf("Connection must not succeed without a valid charset")
		}
	})
}

func TestCollation(t *testing.T) {
	if !available {
		t.Skipf("MySQL-Server not running on %s", netAddr)
	}

	defaultCollation := "utf8_general_ci"
	testCollations := []string{
		"",               // do not set
		defaultCollation, // driver default
		"latin1_general_ci",
		"binary",
		"utf8_unicode_ci",
		"cp1257_bin",
	}

	for _, collation := range testCollations {
		var expected, tdsn string
		if collation != "" {
			tdsn = dsn + "&collation=" + collation
			expected = collation
		} else {
			tdsn = dsn
			expected = defaultCollation
		}

		runTests(t, tdsn, func(dbt *DBTest) {
			var got string
			if err := dbt.db.QueryRow("SELECT @@collation_connection").Scan(&got); err != nil {
				dbt.Fatal(err)
			}

			if got != expected {
				dbt.Fatalf("Expected connection collation %s but got %s", expected, got)
			}
		})
	}
}

func TestColumnsWithAlias(t *testing.T) {
	runTests(t, dsn+"&columnsWithAlias=true", func(dbt *DBTest) {
		rows := dbt.mustQuery("SELECT 1 AS A")
		defer rows.Close()
		cols, _ := rows.Columns()
		if len(cols) != 1 {
			t.Fatalf("expected 1 column, got %d", len(cols))
		}
		if cols[0] != "A" {
			t.Fatalf("expected column name \"A\", got \"%s\"", cols[0])
		}
		rows.Close()

		rows = dbt.mustQuery("SELECT * FROM (SELECT 1 AS one) AS A")
		cols, _ = rows.Columns()
		if len(cols) != 1 {
			t.Fatalf("expected 1 column, got %d", len(cols))
		}
		if cols[0] != "A.one" {
			t.Fatalf("expected column name \"A.one\", got \"%s\"", cols[0])
		}
	})
}

func TestRawBytesResultExceedsBuffer(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		// defaultBufSize from buffer.go
		expected := strings.Repeat("abc", defaultBufSize)

		rows := dbt.mustQuery("SELECT '" + expected + "'")
		defer rows.Close()
		if !rows.Next() {
			dbt.Error("expected result, got none")
		}
		var result sql.RawBytes
		rows.Scan(&result)
		if expected != string(result) {
			dbt.Error("result did not match expected value")
		}
	})
}

func TestTimezoneConversion(t *testing.T) {
	zones := []string{"UTC", "US/Central", "US/Pacific", "Local"}

	// Regression test for timezone handling
	tzTest := func(dbt *DBTest) {

		// Create table
		dbt.mustExec("CREATE TABLE test (ts TIMESTAMP)")

		// Insert local time into database (should be converted)
		usCentral, _ := time.LoadLocation("US/Central")
		reftime := time.Date(2014, 05, 30, 18, 03, 17, 0, time.UTC).In(usCentral)
		dbt.mustExec("INSERT INTO test VALUE (?)", reftime)

		// Retrieve time from DB
		rows := dbt.mustQuery("SELECT ts FROM test")
		if !rows.Next() {
			dbt.Fatal("Didn't get any rows out")
		}

		var dbTime time.Time
		err := rows.Scan(&dbTime)
		if err != nil {
			dbt.Fatal("Err", err)
		}

		// Check that dates match
		if reftime.Unix() != dbTime.Unix() {
			dbt.Errorf("Times don't match.\n")
			dbt.Errorf(" Now(%v)=%v\n", usCentral, reftime)
			dbt.Errorf(" Now(UTC)=%v\n", dbTime)
		}
	}

	for _, tz := range zones {
		runTests(t, dsn+"&parseTime=true&loc="+url.QueryEscape(tz), tzTest)
	}
}

// Special cases

func TestRowsClose(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		rows, err := dbt.db.Query("SELECT 1")
		if err != nil {
			dbt.Fatal(err)
		}

		err = rows.Close()
		if err != nil {
			dbt.Fatal(err)
		}

		if rows.Next() {
			dbt.Fatal("Unexpected row after rows.Close()")
		}

		err = rows.Err()
		if err != nil {
			dbt.Fatal(err)
		}
	})
}

// dangling statements
// http://code.google.com/p/go/issues/detail?id=3865
func TestCloseStmtBeforeRows(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		stmt, err := dbt.db.Prepare("SELECT 1")
		if err != nil {
			dbt.Fatal(err)
		}

		rows, err := stmt.Query()
		if err != nil {
			stmt.Close()
			dbt.Fatal(err)
		}
		defer rows.Close()

		err = stmt.Close()
		if err != nil {
			dbt.Fatal(err)
		}

		if !rows.Next() {
			dbt.Fatal("Getting row failed")
		} else {
			err = rows.Err()
			if err != nil {
				dbt.Fatal(err)
			}

			var out bool
			err = rows.Scan(&out)
			if err != nil {
				dbt.Fatalf("Error on rows.Scan(): %s", err.Error())
			}
			if out != true {
				dbt.Errorf("true != %t", out)
			}
		}
	})
}

// It is valid to have multiple Rows for the same Stmt
// http://code.google.com/p/go/issues/detail?id=3734
func TestStmtMultiRows(t *testing.T) {
	runTests(t, dsn, func(dbt *DBTest) {
		stmt, err := dbt.db.Prepare("SELECT 1 UNION SELECT 0")
		if err != nil {
			dbt.Fatal(err)
		}

		rows1, err := stmt.Query()
		if err != nil {
			stmt.Close()
			dbt.Fatal(err)
		}
		defer rows1.Close()

		rows2, err := stmt.Query()
		if err != nil {
			stmt.Close()
			dbt.Fatal(err)
		}
		defer rows2.Close()

		var out bool

		// 1
		if !rows1.Next() {
			dbt.Fatal("1st rows1.Next failed")
		} else {
			err = rows1.Err()
			if err != nil {
				dbt.Fatal(err)
			}

			err = rows1.Scan(&out)
			if err != nil {
				dbt.Fatalf("Error on rows.Scan(): %s", err.Error())
			}
			if out != true {
				dbt.Errorf("true != %t", out)
			}
		}

		if !rows2.Next() {
			dbt.Fatal("1st rows2.Next failed")
		} else {
			err = rows2.Err()
			if err != nil {
				dbt.Fatal(err)
			}

			err = rows2.Scan(&out)
			if err != nil {
				dbt.Fatalf("Error on rows.Scan(): %s", err.Error())
			}
			if out != true {
				dbt.Errorf("true != %t", out)
			}
		}

		// 2
		if !rows1.Next() {
			dbt.Fatal("2nd rows1.Next failed")
		} else {
			err = rows1.Err()
			if err != nil {
				dbt.Fatal(err)
			}

			err = rows1.Scan(&out)
			if err != nil {
				dbt.Fatalf("Error on rows.Scan(): %s", err.Error())
			}
			if out != false {
				dbt.Errorf("false != %t", out)
			}

			if rows1.Next() {
				dbt.Fatal("Unexpected row on rows1")
			}
			err = rows1.Close()
			if err != nil {
				dbt.Fatal(err)
			}
		}

		if !rows2.Next() {
			dbt.Fatal("2nd rows2.Next failed")
		} else {
			err = rows2.Err()
			if err != nil {
				dbt.Fatal(err)
			}

			err = rows2.Scan(&out)
			if err != nil {
				dbt.Fatalf("Error on rows.Scan(): %s", err.Error())
			}
			if out != false {
				dbt.Errorf("false != %t", out)
			}

			if rows2.Next() {
				dbt.Fatal("Unexpected row on rows2")
			}
			err = rows2.Close()
			if err != nil {
				dbt.Fatal(err)
			}
		}
	})
}

// Regression test for
// * more than 32 NULL parameters (issue 209)
// * more parameters than fit into the buffer (issue 201)
func TestPreparedManyCols(t *testing.T) {
	const numParams = defaultBufSize
	runTests(t, dsn, func(dbt *DBTest) {
		query := "SELECT ?" + strings.Repeat(",?", numParams-1)
		stmt, err := dbt.db.Prepare(query)
		if err != nil {
			dbt.Fatal(err)
		}
		defer stmt.Close()
		// create more parameters than fit into the buffer
		// which will take nil-values
		params := make([]interface{}, numParams)
		rows, err := stmt.Query(params...)
		if err != nil {
			stmt.Close()
			dbt.Fatal(err)
		}
		defer rows.Close()
	})
}

func TestConcurrent(t *testing.T) {
	if enabled, _ := readBool(os.Getenv("MYSQL_TEST_CONCURRENT")); !enabled {
		t.Skip("MYSQL_TEST_CONCURRENT env var not set")
	}

	runTests(t, dsn, func(dbt *DBTest) {
		var max int
		err := dbt.db.QueryRow("SELECT @@max_connections").Scan(&max)
		if err != nil {
			dbt.Fatalf("%s", err.Error())
		}
		dbt.Logf("Testing up to %d concurrent connections \r\n", max)

		var remaining, succeeded int32 = int32(max), 0

		var wg sync.WaitGroup
		wg.Add(max)

		var fatalError string
		var once sync.Once
		fatalf := func(s string, vals ...interface{}) {
			once.Do(func() {
				fatalError = fmt.Sprintf(s, vals...)
			})
		}

		for i := 0; i < max; i++ {
			go func(id int) {
				defer wg.Done()

				tx, err := dbt.db.Begin()
				atomic.AddInt32(&remaining, -1)

				if err != nil {
					if err.Error() != "Error 1040: Too many connections" {
						fatalf("Error on Conn %d: %s", id, err.Error())
					}
					return
				}

				// keep the connection busy until all connections are open
				for remaining > 0 {
					if _, err = tx.Exec("DO 1"); err != nil {
						fatalf("Error on Conn %d: %s", id, err.Error())
						return
					}
				}

				if err = tx.Commit(); err != nil {
					fatalf("Error on Conn %d: %s", id, err.Error())
					return
				}

				// everything went fine with this connection
				atomic.AddInt32(&succeeded, 1)
			}(i)
		}

		// wait until all conections are open
		wg.Wait()

		if fatalError != "" {
			dbt.Fatal(fatalError)
		}

		dbt.Logf("Reached %d concurrent connections\r\n", succeeded)
	})
}

// Tests custom dial functions
func TestCustomDial(t *testing.T) {
	if !available {
		t.Skipf("MySQL-Server not running on %s", netAddr)
	}

	// our custom dial function which justs wraps net.Dial here
	RegisterDial("mydial", func(addr string) (net.Conn, error) {
		return net.Dial(prot, addr)
	})

	db, err := sql.Open("mysql", fmt.Sprintf("%s:%s@mydial(%s)/%s?timeout=30s&strict=true", user, pass, addr, dbname))
	if err != nil {
		t.Fatalf("Error connecting: %s", err.Error())
	}
	defer db.Close()

	if _, err = db.Exec("DO 1"); err != nil {
		t.Fatalf("Connection failed: %s", err.Error())
	}
}

func TestSqlInjection(t *testing.T) {
	createTest := func(arg string) func(dbt *DBTest) {
		return func(dbt *DBTest) {
			dbt.mustExec("CREATE TABLE test (v INTEGER)")
			dbt.mustExec("INSERT INTO test VALUES (?)", 1)

			var v int
			// NULL can't be equal to anything, the idea here is to inject query so it returns row
			// This test verifies that escapeQuotes and escapeBackslash are working properly
			err := dbt.db.QueryRow("SELECT v FROM test WHERE NULL = ?", arg).Scan(&v)
			if err == sql.ErrNoRows {
				return // success, sql injection failed
			} else if err == nil {
				dbt.Errorf("Sql injection successful with arg: %s", arg)
			} else {
				dbt.Errorf("Error running query with arg: %s; err: %s", arg, err.Error())
			}
		}
	}

	dsns := []string{
		dsn,
		dsn + "&sql_mode=NO_BACKSLASH_ESCAPES",
	}
	for _, testdsn := range dsns {
		runTests(t, testdsn, createTest("1 OR 1=1"))
		runTests(t, testdsn, createTest("' OR '1'='1"))
	}
}

// Test if inserted data is correctly retrieved after being escaped
func TestInsertRetrieveEscapedData(t *testing.T) {
	testData := func(dbt *DBTest) {
		dbt.mustExec("CREATE TABLE test (v VARCHAR(255))")

		// All sequences that are escaped by escapeQuotes and escapeBackslash
		v := "foo \x00\n\r\x1a\"'\\"
		dbt.mustExec("INSERT INTO test VALUES (?)", v)

		var out string
		err := dbt.db.QueryRow("SELECT v FROM test").Scan(&out)
		if err != nil {
			dbt.Fatalf("%s", err.Error())
		}

		if out != v {
			dbt.Errorf("%q != %q", out, v)
		}
	}

	dsns := []string{
		dsn,
		dsn + "&sql_mode=NO_BACKSLASH_ESCAPES",
	}
	for _, testdsn := range dsns {
		runTests(t, testdsn, testData)
	}
}
