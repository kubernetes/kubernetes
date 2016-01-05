// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"bytes"
	"database/sql"
	"database/sql/driver"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type TB testing.B

func (tb *TB) check(err error) {
	if err != nil {
		tb.Fatal(err)
	}
}

func (tb *TB) checkDB(db *sql.DB, err error) *sql.DB {
	tb.check(err)
	return db
}

func (tb *TB) checkRows(rows *sql.Rows, err error) *sql.Rows {
	tb.check(err)
	return rows
}

func (tb *TB) checkStmt(stmt *sql.Stmt, err error) *sql.Stmt {
	tb.check(err)
	return stmt
}

func initDB(b *testing.B, queries ...string) *sql.DB {
	tb := (*TB)(b)
	db := tb.checkDB(sql.Open("mysql", dsn))
	for _, query := range queries {
		if _, err := db.Exec(query); err != nil {
			if w, ok := err.(MySQLWarnings); ok {
				b.Logf("Warning on %q: %v", query, w)
			} else {
				b.Fatalf("Error on %q: %v", query, err)
			}
		}
	}
	return db
}

const concurrencyLevel = 10

func BenchmarkQuery(b *testing.B) {
	tb := (*TB)(b)
	b.StopTimer()
	b.ReportAllocs()
	db := initDB(b,
		"DROP TABLE IF EXISTS foo",
		"CREATE TABLE foo (id INT PRIMARY KEY, val CHAR(50))",
		`INSERT INTO foo VALUES (1, "one")`,
		`INSERT INTO foo VALUES (2, "two")`,
	)
	db.SetMaxIdleConns(concurrencyLevel)
	defer db.Close()

	stmt := tb.checkStmt(db.Prepare("SELECT val FROM foo WHERE id=?"))
	defer stmt.Close()

	remain := int64(b.N)
	var wg sync.WaitGroup
	wg.Add(concurrencyLevel)
	defer wg.Wait()
	b.StartTimer()

	for i := 0; i < concurrencyLevel; i++ {
		go func() {
			for {
				if atomic.AddInt64(&remain, -1) < 0 {
					wg.Done()
					return
				}

				var got string
				tb.check(stmt.QueryRow(1).Scan(&got))
				if got != "one" {
					b.Errorf("query = %q; want one", got)
					wg.Done()
					return
				}
			}
		}()
	}
}

func BenchmarkExec(b *testing.B) {
	tb := (*TB)(b)
	b.StopTimer()
	b.ReportAllocs()
	db := tb.checkDB(sql.Open("mysql", dsn))
	db.SetMaxIdleConns(concurrencyLevel)
	defer db.Close()

	stmt := tb.checkStmt(db.Prepare("DO 1"))
	defer stmt.Close()

	remain := int64(b.N)
	var wg sync.WaitGroup
	wg.Add(concurrencyLevel)
	defer wg.Wait()
	b.StartTimer()

	for i := 0; i < concurrencyLevel; i++ {
		go func() {
			for {
				if atomic.AddInt64(&remain, -1) < 0 {
					wg.Done()
					return
				}

				if _, err := stmt.Exec(); err != nil {
					b.Fatal(err.Error())
				}
			}
		}()
	}
}

// data, but no db writes
var roundtripSample []byte

func initRoundtripBenchmarks() ([]byte, int, int) {
	if roundtripSample == nil {
		roundtripSample = []byte(strings.Repeat("0123456789abcdef", 1024*1024))
	}
	return roundtripSample, 16, len(roundtripSample)
}

func BenchmarkRoundtripTxt(b *testing.B) {
	b.StopTimer()
	sample, min, max := initRoundtripBenchmarks()
	sampleString := string(sample)
	b.ReportAllocs()
	tb := (*TB)(b)
	db := tb.checkDB(sql.Open("mysql", dsn))
	defer db.Close()
	b.StartTimer()
	var result string
	for i := 0; i < b.N; i++ {
		length := min + i
		if length > max {
			length = max
		}
		test := sampleString[0:length]
		rows := tb.checkRows(db.Query(`SELECT "` + test + `"`))
		if !rows.Next() {
			rows.Close()
			b.Fatalf("crashed")
		}
		err := rows.Scan(&result)
		if err != nil {
			rows.Close()
			b.Fatalf("crashed")
		}
		if result != test {
			rows.Close()
			b.Errorf("mismatch")
		}
		rows.Close()
	}
}

func BenchmarkRoundtripBin(b *testing.B) {
	b.StopTimer()
	sample, min, max := initRoundtripBenchmarks()
	b.ReportAllocs()
	tb := (*TB)(b)
	db := tb.checkDB(sql.Open("mysql", dsn))
	defer db.Close()
	stmt := tb.checkStmt(db.Prepare("SELECT ?"))
	defer stmt.Close()
	b.StartTimer()
	var result sql.RawBytes
	for i := 0; i < b.N; i++ {
		length := min + i
		if length > max {
			length = max
		}
		test := sample[0:length]
		rows := tb.checkRows(stmt.Query(test))
		if !rows.Next() {
			rows.Close()
			b.Fatalf("crashed")
		}
		err := rows.Scan(&result)
		if err != nil {
			rows.Close()
			b.Fatalf("crashed")
		}
		if !bytes.Equal(result, test) {
			rows.Close()
			b.Errorf("mismatch")
		}
		rows.Close()
	}
}

func BenchmarkInterpolation(b *testing.B) {
	mc := &mysqlConn{
		cfg: &config{
			interpolateParams: true,
			loc:               time.UTC,
		},
		maxPacketAllowed: maxPacketSize,
		maxWriteSize:     maxPacketSize - 1,
		buf:              newBuffer(nil),
	}

	args := []driver.Value{
		int64(42424242),
		float64(math.Pi),
		false,
		time.Unix(1423411542, 807015000),
		[]byte("bytes containing special chars ' \" \a \x00"),
		"string containing special chars ' \" \a \x00",
	}
	q := "SELECT ?, ?, ?, ?, ?, ?"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := mc.interpolateParams(q, args)
		if err != nil {
			b.Fatal(err)
		}
	}
}
