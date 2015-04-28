// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"crypto/md5"
	"database/sql"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/big"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cznic/strutil"
)

// Note: All benchmarks report MB/s equal to record/s.
const benchScale = 1e6

func init() {
	log.SetFlags(log.Flags() | log.Lshortfile)
	isTesting = true
}

func dbg(s string, va ...interface{}) {
	if s == "" {
		s = strings.Repeat("%v ", len(va))
	}
	_, fn, fl, _ := runtime.Caller(1)
	fmt.Printf("dbg %s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
}

func caller(s string, va ...interface{}) {
	_, fn, fl, _ := runtime.Caller(2)
	fmt.Printf("caller: %s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
	_, fn, fl, _ = runtime.Caller(1)
	fmt.Printf("\tcallee: %s:%d: ", path.Base(fn), fl)
	fmt.Println()
}

func use(...interface{}) {}

func dumpTables3(r *root) {
	fmt.Printf("---- r.head %d, r.thead %p\n", r.head, r.thead)
	for k, v := range r.tables {
		fmt.Printf("%p: %s->%+v\n", v, k, v)
	}
	fmt.Println("<exit>")
}

func dumpTables2(s storage) {
	fmt.Println("****")
	h := int64(1)
	for h != 0 {
		d, err := s.Read(nil, h)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("%d: %v\n", h, d)
		h = d[0].(int64)
	}
	fmt.Println("<exit>")
}

func (db *DB) dumpTables() string {
	var buf bytes.Buffer
	for k, v := range db.root.tables {
		buf.WriteString(fmt.Sprintf("%s->%v, %v\n", k, v.h, v.next))
	}
	return buf.String()
}

func fldsString(f []*fld) string {
	a := []string{}
	for _, v := range f {
		a = append(a, v.name)
	}
	return strings.Join(a, " ")
}

type testDB interface {
	setup() (db *DB, err error)
	mark() (err error)
	teardown() (err error)
}

var (
	_ testDB = (*fileTestDB)(nil)
	_ testDB = (*memTestDB)(nil)
)

type memTestDB struct {
	db *DB
	m0 int64
}

func (m *memTestDB) setup() (db *DB, err error) {
	if m.db, err = OpenMem(); err != nil {
		return
	}

	return m.db, nil
}

func (m *memTestDB) mark() (err error) {
	m.m0, err = m.db.store.Verify()
	if err != nil {
		m.m0 = -1
	}
	return
}

func (m *memTestDB) teardown() (err error) {
	if m.m0 < 0 {
		return
	}

	n, err := m.db.store.Verify()
	if err != nil {
		return
	}

	if g, e := n, m.m0; g != e {
		return fmt.Errorf("STORAGE LEAK: allocs: got %d, exp %d", g, e)
	}

	return
}

type fileTestDB struct {
	db   *DB
	gmp0 int
	m0   int64
}

func (m *fileTestDB) setup() (db *DB, err error) {
	m.gmp0 = runtime.GOMAXPROCS(0)
	f, err := ioutil.TempFile("", "ql-test-")
	if err != nil {
		return
	}

	if m.db, err = OpenFile(f.Name(), &Options{}); err != nil {
		return
	}

	return m.db, nil
}

func (m *fileTestDB) mark() (err error) {
	m.m0, err = m.db.store.Verify()
	if err != nil {
		m.m0 = -1
	}
	return
}

func (m *fileTestDB) teardown() (err error) {
	runtime.GOMAXPROCS(m.gmp0)
	defer func() {
		f := m.db.store.(*file)
		errSet(&err, m.db.Close())
		os.Remove(f.f0.Name())
		if f.wal != nil {
			os.Remove(f.wal.Name())
		}
	}()

	if m.m0 < 0 {
		return
	}

	n, err := m.db.store.Verify()
	if err != nil {
		return
	}

	if g, e := n, m.m0; g != e {
		return fmt.Errorf("STORAGE LEAK: allocs: got %d, exp %d", g, e)
	}
	return
}

type osFileTestDB struct {
	db   *DB
	gmp0 int
	m0   int64
}

func (m *osFileTestDB) setup() (db *DB, err error) {
	m.gmp0 = runtime.GOMAXPROCS(0)
	f, err := ioutil.TempFile("", "ql-test-osfile")
	if err != nil {
		return
	}

	if m.db, err = OpenFile("", &Options{OSFile: f}); err != nil {
		return
	}

	return m.db, nil
}

func (m *osFileTestDB) mark() (err error) {
	m.m0, err = m.db.store.Verify()
	if err != nil {
		m.m0 = -1
	}
	return
}

func (m *osFileTestDB) teardown() (err error) {
	runtime.GOMAXPROCS(m.gmp0)
	defer func() {
		f := m.db.store.(*file)
		errSet(&err, m.db.Close())
		os.Remove(f.f0.Name())
		if f.wal != nil {
			os.Remove(f.wal.Name())
		}
	}()

	if m.m0 < 0 {
		return
	}

	n, err := m.db.store.Verify()
	if err != nil {
		return
	}

	if g, e := n, m.m0; g != e {
		return fmt.Errorf("STORAGE LEAK: allocs: got %d, exp %d", g, e)
	}
	return
}

func TestMemStorage(t *testing.T) {
	test(t, &memTestDB{})
}

func TestFileStorage(t *testing.T) {
	test(t, &fileTestDB{})
}

func TestOSFileStorage(t *testing.T) {
	test(t, &osFileTestDB{})
}

var (
	compiledCommit        = MustCompile("COMMIT;")
	compiledCreate        = MustCompile("BEGIN TRANSACTION; CREATE TABLE t (i16 int16, s16 string, s string);")
	compiledCreate2       = MustCompile("BEGIN TRANSACTION; CREATE TABLE t (i16 int16, s16 string, s string); COMMIT;")
	compiledIns           = MustCompile("INSERT INTO t VALUES($1, $2, $3);")
	compiledSelect        = MustCompile("SELECT * FROM t;")
	compiledSelectOrderBy = MustCompile("SELECT * FROM t ORDER BY i16, s16;")
	compiledTrunc         = MustCompile("BEGIN TRANSACTION; TRUNCATE TABLE t; COMMIT;")
)

func rnds16(rng *rand.Rand, n int) string {
	a := make([]string, n)
	for i := range a {
		a[i] = fmt.Sprintf("%016x", rng.Int63())
	}
	return strings.Join(a, "")
}

var (
	benchmarkScaleOnce  sync.Once
	benchmarkSelectOnce = map[string]sync.Once{}
)

func benchProlog(b *testing.B) {
	benchmarkScaleOnce.Do(func() {
		b.Logf(`
=============================================================
NOTE: All benchmarks report records/s as %d bytes/s.
=============================================================`, int64(benchScale))
	})
}

func benchmarkSelect(b *testing.B, n int, sel List, ts testDB) {
	if testing.Verbose() {
		benchProlog(b)
		id := fmt.Sprintf("%T|%d", ts, n)
		once := benchmarkSelectOnce[id]
		once.Do(func() {
			b.Logf(`Having a table of %d records, each of size 1kB, measure the performance of
%s
`, n, sel)
		})
		benchmarkSelectOnce[id] = once
	}

	db, err := ts.setup()
	if err != nil {
		b.Error(err)
		return
	}

	defer ts.teardown()

	ctx := NewRWCtx()
	if _, i, err := db.Execute(ctx, compiledCreate); err != nil {
		b.Error(i, err)
		return
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		if _, _, err := db.Execute(ctx, compiledIns, int16(rng.Int()), rnds16(rng, 1), rnds16(rng, 63)); err != nil {
			b.Error(err)
			return
		}
	}

	if _, i, err := db.Execute(ctx, compiledCommit); err != nil {
		b.Error(i, err)
		return
	}

	b.SetBytes(int64(n) * benchScale)
	runtime.GC()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rs, index, err := db.Execute(nil, sel)
		if err != nil {
			b.Error(index, err)
			return
		}

		if err = rs[0].Do(false, func(record []interface{}) (bool, error) { return true, nil }); err != nil {
			b.Errorf("%v %T(%#v)", err, err, err)
			return
		}
	}
	b.StopTimer()

}

func BenchmarkSelectMem1kBx1e2(b *testing.B) {
	benchmarkSelect(b, 1e2, compiledSelect, &memTestDB{})
}

func BenchmarkSelectMem1kBx1e3(b *testing.B) {
	benchmarkSelect(b, 1e3, compiledSelect, &memTestDB{})
}

func BenchmarkSelectMem1kBx1e4(b *testing.B) {
	benchmarkSelect(b, 1e4, compiledSelect, &memTestDB{})
}

func BenchmarkSelectMem1kBx1e5(b *testing.B) {
	benchmarkSelect(b, 1e5, compiledSelect, &memTestDB{})
}

func BenchmarkSelectFile1kBx1e2(b *testing.B) {
	benchmarkSelect(b, 1e2, compiledSelect, &fileTestDB{})
}

func BenchmarkSelectFile1kBx1e3(b *testing.B) {
	benchmarkSelect(b, 1e3, compiledSelect, &fileTestDB{})
}

func BenchmarkSelectFile1kBx1e4(b *testing.B) {
	benchmarkSelect(b, 1e4, compiledSelect, &fileTestDB{})
}

func BenchmarkSelectFile1kBx1e5(b *testing.B) {
	benchmarkSelect(b, 1e5, compiledSelect, &fileTestDB{})
}

func BenchmarkSelectOrderedMem1kBx1e2(b *testing.B) {
	benchmarkSelect(b, 1e2, compiledSelectOrderBy, &memTestDB{})
}

func BenchmarkSelectOrderedMem1kBx1e3(b *testing.B) {
	benchmarkSelect(b, 1e3, compiledSelectOrderBy, &memTestDB{})
}

func BenchmarkSelectOrderedMem1kBx1e4(b *testing.B) {
	benchmarkSelect(b, 1e4, compiledSelectOrderBy, &memTestDB{})
}

func BenchmarkSelectOrderedFile1kBx1e2(b *testing.B) {
	benchmarkSelect(b, 1e2, compiledSelectOrderBy, &fileTestDB{})
}

func BenchmarkSelectOrderedFile1kBx1e3(b *testing.B) {
	benchmarkSelect(b, 1e3, compiledSelectOrderBy, &fileTestDB{})
}

func BenchmarkSelectOrderedFile1kBx1e4(b *testing.B) {
	benchmarkSelect(b, 1e4, compiledSelectOrderBy, &fileTestDB{})
}

func TestString(t *testing.T) {
	for _, v := range testdata {
		a := strings.Split(v, "\n|")
		v = a[0]
		v = strings.Replace(v, "&or;", "|", -1)
		v = strings.Replace(v, "&oror;", "||", -1)
		l, err := Compile(v)
		if err != nil {
			continue
		}

		if s := l.String(); len(s) == 0 {
			t.Fatal("List.String() returned empty string")
		}
	}
}

var benchmarkInsertOnce = map[string]sync.Once{}

func benchmarkInsert(b *testing.B, batch, total int, ts testDB) {
	if testing.Verbose() {
		benchProlog(b)
		id := fmt.Sprintf("%T|%d|%d", ts, batch, total)
		once := benchmarkInsertOnce[id]
		once.Do(func() {
			b.Logf(`In batches of %d record(s), insert a total of %d records, each of size 1kB, into a table.

`, batch, total)
		})
		benchmarkInsertOnce[id] = once
	}

	if total%batch != 0 {
		b.Fatal("internal error 001")
	}

	db, err := ts.setup()
	if err != nil {
		b.Error(err)
		return
	}

	defer ts.teardown()

	ctx := NewRWCtx()
	if _, i, err := db.Execute(ctx, compiledCreate2); err != nil {
		b.Error(i, err)
		return
	}

	s := fmt.Sprintf(`(0, "0123456789abcdef", "%s"),`, rnds16(rand.New(rand.NewSource(42)), 63))
	var buf bytes.Buffer
	buf.WriteString("BEGIN TRANSACTION; INSERT INTO t VALUES\n")
	for i := 0; i < batch; i++ {
		buf.WriteString(s)
	}
	buf.WriteString("; COMMIT;")
	ins, err := Compile(buf.String())
	if err != nil {
		b.Error(err)
		return
	}

	b.SetBytes(int64(total) * benchScale)
	runtime.GC()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for n := 0; n != total; n += batch {
			if _, _, err = db.Execute(ctx, ins); err != nil {
				b.Error(err)
				return
			}
		}
		b.StopTimer()
		if _, _, err = db.Execute(ctx, compiledTrunc); err != nil {
			b.Error(err)
		}
		b.StartTimer()
	}
	b.StopTimer()
}

func BenchmarkInsertMem1kBn1e0t1e2(b *testing.B) {
	benchmarkInsert(b, 1e0, 1e2, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e0t1e2(b *testing.B) {
	benchmarkInsert(b, 1e0, 1e2, &fileTestDB{})
}

//=============================================================================

func BenchmarkInsertMem1kBn1e1t1e2(b *testing.B) {
	benchmarkInsert(b, 1e1, 1e2, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e1t1e2(b *testing.B) {
	benchmarkInsert(b, 1e1, 1e2, &fileTestDB{})
}

func BenchmarkInsertMem1kBn1e1t1e3(b *testing.B) {
	benchmarkInsert(b, 1e1, 1e3, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e1t1e3(b *testing.B) {
	benchmarkInsert(b, 1e1, 1e3, &fileTestDB{})
}

//=============================================================================

func BenchmarkInsertMem1kBn1e2t1e2(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e2, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e2t1e2(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e2, &fileTestDB{})
}

func BenchmarkInsertMem1kBn1e2t1e3(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e3, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e2t1e3(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e3, &fileTestDB{})
}

func BenchmarkInsertMem1kBn1e2t1e4(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e4, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e2t1e4(b *testing.B) {
	benchmarkInsert(b, 1e2, 1e4, &fileTestDB{})
}

//=============================================================================

func BenchmarkInsertMem1kBn1e3t1e3(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e3, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e3t1e3(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e3, &fileTestDB{})
}

func BenchmarkInsertMem1kBn1e3t1e4(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e4, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e3t1e4(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e4, &fileTestDB{})
}

func BenchmarkInsertMem1kBn1e3t1e5(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e5, &memTestDB{})
}

func BenchmarkInsertFile1kBn1e3t1e5(b *testing.B) {
	benchmarkInsert(b, 1e3, 1e5, &fileTestDB{})
}

func TestReopen(t *testing.T) {
	f, err := ioutil.TempFile("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	nm := f.Name()
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if nm != "" {
			os.Remove(nm)
		}
	}()

	db, err := OpenFile(nm, &Options{})
	if err != nil {
		t.Error(err)
		return
	}

	for _, tn := range "abc" {
		ql := fmt.Sprintf(`
BEGIN TRANSACTION;
	CREATE TABLE %c (i int, s string);
	INSERT INTO %c VALUES (%d, "<%c>");
COMMIT;
		`, tn, tn, tn-'a'+42, tn)
		_, i, err := db.Run(NewRWCtx(), ql)
		if err != nil {
			db.Close()
			t.Error(i, err)
			return
		}
	}

	if err = db.Close(); err != nil {
		t.Error(err)
		return
	}

	db, err = OpenFile(nm, &Options{})
	if err != nil {
		t.Error(err)
		return
	}

	defer func() {
		if err = db.Close(); err != nil {
			t.Error(err)
		}
	}()

	if _, _, err = db.Run(NewRWCtx(), "BEGIN TRANSACTION; DROP TABLE b; COMMIT;"); err != nil {
		t.Error(err)
		return
	}

	for _, tn := range "ac" {
		ql := fmt.Sprintf("SELECT * FROM %c;", tn)
		rs, i, err := db.Run(NewRWCtx(), ql)
		if err != nil {
			t.Error(i, err)
			return
		}

		if rs == nil {
			t.Error(rs)
			return
		}

		rid := 0
		if err = rs[0].Do(false, func(r []interface{}) (bool, error) {
			rid++
			if rid != 1 {
				return false, fmt.Errorf("rid %d", rid)
			}

			if g, e := recStr(r), fmt.Sprintf(`%d, "<%c>"`, tn-'a'+42, tn); g != e {
				return false, fmt.Errorf("g `%s`, e `%s`", g, e)
			}

			return true, nil
		}); err != nil {
			t.Error(err)
			return
		}
	}
}

func recStr(data []interface{}) string {
	a := make([]string, len(data))
	for i, v := range data {
		switch x := v.(type) {
		case string:
			a[i] = fmt.Sprintf("%q", x)
		default:
			a[i] = fmt.Sprint(x)
		}
	}
	return strings.Join(a, ", ")
}

//TODO +test long blob types with multiple inner chunks.

func TestLastInsertID(t *testing.T) {
	table := []struct {
		ql string
		id int
	}{
		// 0
		{`BEGIN TRANSACTION; CREATE TABLE t (c int); COMMIT`, 0},
		{`BEGIN TRANSACTION; INSERT INTO t VALUES (41); COMMIT`, 1},
		{`BEGIN TRANSACTION; INSERT INTO t VALUES (42);`, 2},
		{`INSERT INTO t VALUES (43)`, 3},
		{`ROLLBACK; BEGIN TRANSACTION; INSERT INTO t VALUES (44); COMMIT`, 4},

		//5
		{`BEGIN TRANSACTION; INSERT INTO t VALUES (45); COMMIT`, 5},
		{`
		BEGIN TRANSACTION;
			INSERT INTO t VALUES (46); // 6
			BEGIN TRANSACTION;
				INSERT INTO t VALUES (47); // 7
				INSERT INTO t VALUES (48); // 8
			ROLLBACK;
			INSERT INTO t VALUES (49); // 9
		COMMIT`, 9},
		{`
		BEGIN TRANSACTION;
			INSERT INTO t VALUES (50); // 10
			BEGIN TRANSACTION;
				INSERT INTO t VALUES (51); // 11
				INSERT INTO t VALUES (52); // 12
			ROLLBACK;
		COMMIT;`, 10},
		{`BEGIN TRANSACTION; INSERT INTO t VALUES (53); ROLLBACK`, 10},
		{`BEGIN TRANSACTION; INSERT INTO t VALUES (54); COMMIT`, 14},
		// 10
		{`BEGIN TRANSACTION; CREATE TABLE u (c int); COMMIT`, 14},
		{`
		BEGIN TRANSACTION;
			INSERT INTO t SELECT * FROM u;
		COMMIT;`, 14},
		{`BEGIN TRANSACTION; INSERT INTO u VALUES (150); COMMIT`, 15},
		{`
		BEGIN TRANSACTION;
			INSERT INTO t SELECT * FROM u;
		COMMIT;`, 16},
	}

	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	ctx := NewRWCtx()
	for i, test := range table {
		l, err := Compile(test.ql)
		if err != nil {
			t.Fatal(i, err)
		}

		if _, _, err = db.Execute(ctx, l); err != nil {
			t.Fatal(i, err)
		}

		if g, e := ctx.LastInsertID, int64(test.id); g != e {
			t.Fatal(i, g, e)
		}
	}
}

func ExampleTCtx_lastInsertID() {
	ins := MustCompile("BEGIN TRANSACTION; INSERT INTO t VALUES ($1); COMMIT;")

	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	ctx := NewRWCtx()
	if _, _, err = db.Run(ctx, `
		BEGIN TRANSACTION;
			CREATE TABLE t (c int);
			INSERT INTO t VALUES (1), (2), (3);
		COMMIT;
	`); err != nil {
		panic(err)
	}

	if _, _, err = db.Execute(ctx, ins, int64(42)); err != nil {
		panic(err)
	}

	id := ctx.LastInsertID
	rs, _, err := db.Run(ctx, `SELECT * FROM t WHERE id() == $1`, id)
	if err != nil {
		panic(err)
	}

	if err = rs[0].Do(false, func(data []interface{}) (more bool, err error) {
		fmt.Println(data)
		return true, nil
	}); err != nil {
		panic(err)
	}
	// Output:
	// [42]
}

func Example_recordsetFields() {
	// See RecordSet.Fields documentation

	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	ctx := NewRWCtx()
	rs, _, err := db.Run(ctx, `
		BEGIN TRANSACTION;
			CREATE TABLE t (s string, i int);
			CREATE TABLE u (s string, i int);
			INSERT INTO t VALUES
				("a", 1),
				("a", 2),
				("b", 3),
				("b", 4),
			;
			INSERT INTO u VALUES
				("A", 10),
				("A", 20),
				("B", 30),
				("B", 40),
			;
		COMMIT;
		
		// [0]: Fields are not computable.
		SELECT * FROM noTable;
		
		// [1]: Fields are computable even when Do will fail (table noTable does not exist).
		SELECT X AS Y FROM noTable;
		
		// [2]: Both Fields and Do are okay.
		SELECT t.s+u.s as a, t.i+u.i as b, "noName", "name" as Named FROM t, u;
		
		// [3]: Filds are computable even when Do will fail (uknown column a).
		SELECT DISTINCT s as S, sum(i) as I FROM (
			SELECT t.s+u.s as s, t.i+u.i, 3 as i FROM t, u;
		)
		GROUP BY a
		ORDER BY d;
		
		// [4]: Fields are computable even when Do will fail on missing $1.
		SELECT DISTINCT * FROM (
			SELECT t.s+u.s as S, t.i+u.i, 3 as I FROM t, u;
		)
		WHERE I < $1
		ORDER BY S;
		` /* , 42 */) // <-- $1 missing
	if err != nil {
		panic(err)
	}

	for i, v := range rs {
		fields, err := v.Fields()
		switch {
		case err != nil:
			fmt.Printf("Fields[%d]: error: %s\n", i, err)
		default:
			fmt.Printf("Fields[%d]: %#v\n", i, fields)
		}
		if err = v.Do(
			true,
			func(data []interface{}) (more bool, err error) {
				fmt.Printf("    Do[%d]: %#v\n", i, data)
				return false, nil
			},
		); err != nil {
			fmt.Printf("    Do[%d]: error: %s\n", i, err)
		}
	}
	// Output:
	// Fields[0]: error: table noTable does not exist
	//     Do[0]: error: table noTable does not exist
	// Fields[1]: []string{"Y"}
	//     Do[1]: error: table noTable does not exist
	// Fields[2]: []string{"a", "b", "", "Named"}
	//     Do[2]: []interface {}{"a", "b", "", "Named"}
	// Fields[3]: []string{"S", "I"}
	//     Do[3]: error: unknown column a
	// Fields[4]: []string{"S", "", "I"}
	//     Do[4]: error: missing $1
}

func TestRowsAffected(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	ctx := NewRWCtx()
	ctx.LastInsertID, ctx.RowsAffected = -1, -1
	if _, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		CREATE TABLE t (i int);
	COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	if g, e := ctx.LastInsertID, int64(0); g != e {
		t.Fatal(g, e)
	}

	if g, e := ctx.RowsAffected, int64(0); g != e {
		t.Fatal(g, e)
	}

	if _, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		INSERT INTO t VALUES (1), (2), (3);
	COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	if g, e := ctx.LastInsertID, int64(3); g != e {
		t.Fatal(g, e)
	}

	if g, e := ctx.RowsAffected, int64(3); g != e {
		t.Fatal(g, e)
	}

	if _, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		INSERT INTO t
		SELECT * FROM t WHERE i == 2;
	COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	if g, e := ctx.LastInsertID, int64(4); g != e {
		t.Fatal(g, e)
	}

	if g, e := ctx.RowsAffected, int64(1); g != e {
		t.Fatal(g, e)
	}

	if _, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		DELETE FROM t WHERE i == 2;
	COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	if g, e := ctx.LastInsertID, int64(4); g != e {
		t.Fatal(g, e)
	}

	if g, e := ctx.RowsAffected, int64(2); g != e {
		t.Fatal(g, e)
	}

	if _, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		UPDATE t i = 42 WHERE i == 3;
	COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	if g, e := ctx.LastInsertID, int64(4); g != e {
		t.Fatal(g, e)
	}

	if g, e := ctx.RowsAffected, int64(1); g != e {
		t.Fatal(g, e)
	}
}

func dumpDB(db *DB, tag string) (string, error) {
	var buf bytes.Buffer
	f := strutil.IndentFormatter(&buf, "\t")
	f.Format("---- %s%i\n", tag)
	for nm, tab := range db.root.tables {
		h := tab.head
		f.Format("%u%q: head %d, scols0 %q, scols %q%i\n", nm, h, cols2meta(tab.cols0), cols2meta(tab.cols))
		for h != 0 {
			rec, err := db.store.Read(nil, h, tab.cols...)
			if err != nil {
				return "", err
			}

			f.Format("record @%d: %v\n", h, rec)
			h = rec[0].(int64)
		}
		f.Format("%u")
		for i, v := range tab.indices {
			if v == nil {
				continue
			}

			xname := v.name
			cname := "id()"
			if i != 0 {
				cname = tab.cols0[i-1].name
			}
			f.Format("index %s on %s%i\n", xname, cname)
			it, _, err := v.x.Seek(nil)
			if err != nil {
				return "", err
			}

			for {
				k, v, err := it.Next()
				if err != nil {
					if err == io.EOF {
						break
					}

					return "", err
				}

				f.Format("%v: %v\n", k, v)
			}
			f.Format("%u")
		}
	}

	return buf.String(), nil
}

func testIndices(db *DB, t *testing.T) {
	ctx := NewRWCtx()
	var err error
	e := func(q string) {
		if _, _, err = db.Run(ctx, q); err != nil {
			t.Fatal(err)
		}

		s, err := dumpDB(db, "post\n\t"+q)
		if err != nil {
			t.Fatal(err)
		}

		t.Logf("%s\n\n", s)
		if db.isMem {
			return
		}

		nm := db.Name()
		if err = db.Close(); err != nil {
			t.Fatal(err)
		}

		if db, err = OpenFile(nm, &Options{}); err != nil {
			t.Fatal(err)
		}

		if s, err = dumpDB(db, "reopened"); err != nil {
			t.Fatal(err)
		}

		t.Logf("%s\n\n", s)
	}

	e(`	BEGIN TRANSACTION;
			CREATE TABLE t (i int);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			CREATE INDEX x ON t (id());
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			INSERT INTO t VALUES(42);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			INSERT INTO t VALUES(24);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			CREATE INDEX xi ON t (i);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			INSERT INTO t VALUES(1);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			INSERT INTO t VALUES(999);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			UPDATE t i = 240 WHERE i == 24;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DELETE FROM t WHERE i == 240;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			TRUNCATE TABLE t;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int, s string);
			CREATE INDEX xi ON t (i);
			INSERT INTO t VALUES (42, "foo");
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			ALTER TABLE t DROP COLUMN i;
		COMMIT;`)

	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES (42);
			INSERT INTO t SELECT 10*i FROM t;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES (42);
		COMMIT;
		BEGIN TRANSACTION;
			INSERT INTO t SELECT 10*i FROM t;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES (42);
			DROP INDEX x;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES (42);
		COMMIT;
		BEGIN TRANSACTION;
			DROP INDEX x;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES (42);
		COMMIT;`)
	e(`
		BEGIN TRANSACTION;
			DROP INDEX x;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
			ALTER TABLE t ADD s string;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i int);
			CREATE INDEX x ON t (i);
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			ALTER TABLE t ADD s string;
		COMMIT;`)
	e(`	BEGIN TRANSACTION;
			DROP TABLE IF EXISTS t;
			CREATE TABLE t (i bigint);
			CREATE INDEX x ON t (i);
			INSERT INTO t VALUES(42);
		COMMIT;`)

	if err = db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestIndices(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	testIndices(db, t)
	dir, err := ioutil.TempDir("", "ql-test")

	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)

	nm := filepath.Join(dir, "ql.db")
	db, err = OpenFile(nm, &Options{CanCreate: true})
	if err != nil {
		t.Fatal(err)
	}

	testIndices(db, t)
}

var benchmarkInsertBoolOnce = map[string]sync.Once{}

func benchmarkInsertBool(b *testing.B, db *DB, size int, selectivity float64, index bool, teardown func()) {
	if testing.Verbose() {
		benchProlog(b)
		id := fmt.Sprintf("%t|%d|%g|%t", db.isMem, size, selectivity, index)
		once := benchmarkInsertBoolOnce[id]
		once.Do(func() {
			s := "INDEXED"
			if !index {
				s = "NON " + s
			}
			b.Logf(`Insert %d records into a table having a single bool %s column. Batch size: 1 record.

`, size, s)
		})
		benchmarkInsertBoolOnce[id] = once
	}

	if teardown != nil {
		defer teardown()
	}

	ctx := NewRWCtx()
	if _, _, err := db.Run(ctx, `
		BEGIN TRANSACTION;
			CREATE TABLE t (b bool);
	`); err != nil {
		b.Fatal(err)
	}

	if index {
		if _, _, err := db.Run(ctx, `
			CREATE INDEX x ON t (b);
		`); err != nil {
			b.Fatal(err)
		}
	}

	ins, err := Compile("INSERT INTO t VALUES($1);")
	if err != nil {
		b.Fatal(err)
	}

	trunc, err := Compile("TRUNCATE TABLE t;")
	if err != nil {
		b.Fatal(err)
	}

	begin, err := Compile("BEGIN TRANSACTION;")
	if err != nil {
		b.Fatal(err)
	}

	commit, err := Compile("COMMIT;")
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	debug.FreeOSMemory()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		if i != 0 {
			if _, _, err = db.Execute(ctx, begin); err != nil {
				b.Fatal(err)
			}
		}

		if _, _, err = db.Execute(ctx, trunc); err != nil {
			b.Fatal(err)
		}

		b.StartTimer()
		for j := 0; j < size; j++ {
			if _, _, err = db.Execute(ctx, ins, rng.Float64() < selectivity); err != nil {
				b.Fatal(err)
			}
		}
		if _, _, err = db.Execute(ctx, commit); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.SetBytes(int64(size) * benchScale)
}

func benchmarkInsertBoolMem(b *testing.B, size int, sel float64, index bool) {
	db, err := OpenMem()
	if err != nil {
		b.Fatal(err)
	}

	benchmarkInsertBool(b, db, size, sel, index, nil)
}

func BenchmarkInsertBoolMemNoX1e1(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e1, 0.5, false)
}

func BenchmarkInsertBoolMemX1e1(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e1, 0.5, true)
}

func BenchmarkInsertBoolMemNoX1e2(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e2, 0.5, false)
}

func BenchmarkInsertBoolMemX1e2(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e2, 0.5, true)
}

func BenchmarkInsertBoolMemNoX1e3(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e3, 0.5, false)
}

func BenchmarkInsertBoolMemX1e3(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e3, 0.5, true)
}

func BenchmarkInsertBoolMemNoX1e4(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e4, 0.5, false)
}

func BenchmarkInsertBoolMemX1e4(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e4, 0.5, true)
}

func BenchmarkInsertBoolMemNoX1e5(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e5, 0.5, false)
}

func BenchmarkInsertBoolMemX1e5(b *testing.B) {
	benchmarkInsertBoolMem(b, 1e5, 0.5, true)
}

func benchmarkInsertBoolFile(b *testing.B, size int, sel float64, index bool) {
	dir, err := ioutil.TempDir("", "ql-bench-")
	if err != nil {
		b.Fatal(err)
	}

	n := runtime.GOMAXPROCS(0)
	db, err := OpenFile(filepath.Join(dir, "ql.db"), &Options{CanCreate: true})
	if err != nil {
		b.Fatal(err)
	}

	benchmarkInsertBool(b, db, size, sel, index, func() {
		runtime.GOMAXPROCS(n)
		db.Close()
		os.RemoveAll(dir)
	})
}

func BenchmarkInsertBoolFileNoX1e1(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e1, 0.5, false)
}

func BenchmarkInsertBoolFileX1e1(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e1, 0.5, true)
}

func BenchmarkInsertBoolFileNoX1e2(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e2, 0.5, false)
}

func BenchmarkInsertBoolFileX1e2(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e2, 0.5, true)
}

func BenchmarkInsertBoolFileNoX1e3(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e3, 0.5, false)
}

func BenchmarkInsertBoolFileX1e3(b *testing.B) {
	benchmarkInsertBoolFile(b, 1e3, 0.5, true)
}

var benchmarkSelectBoolOnce = map[string]sync.Once{}

func benchmarkSelectBool(b *testing.B, db *DB, size int, selectivity float64, index bool, teardown func()) {
	sel, err := Compile("SELECT * FROM t WHERE b;")
	if err != nil {
		b.Fatal(err)
	}

	if testing.Verbose() {
		benchProlog(b)
		id := fmt.Sprintf("%t|%d|%g|%t", db.isMem, size, selectivity, index)
		once := benchmarkSelectBoolOnce[id]
		once.Do(func() {
			s := "INDEXED"
			if !index {
				s = "NON " + s
			}
			b.Logf(`A table has a single %s bool column b. Insert %d records with a random bool value,
%.0f%% of them are true. Measure the performance of
%s
`, s, size, 100*selectivity, sel)
		})
		benchmarkSelectBoolOnce[id] = once
	}

	if teardown != nil {
		defer teardown()
	}

	ctx := NewRWCtx()
	if _, _, err := db.Run(ctx, `
		BEGIN TRANSACTION;
			CREATE TABLE t (b bool);
	`); err != nil {
		b.Fatal(err)
	}

	if index {
		if _, _, err := db.Run(ctx, `
			CREATE INDEX x ON t (b);
		`); err != nil {
			b.Fatal(err)
		}
	}

	ins, err := Compile("INSERT INTO t VALUES($1);")
	if err != nil {
		b.Fatal(err)
	}

	var n int64
	rng := rand.New(rand.NewSource(42))
	for j := 0; j < size; j++ {
		v := rng.Float64() < selectivity
		if v {
			n++
		}
		if _, _, err = db.Execute(ctx, ins, v); err != nil {
			b.Fatal(err)
		}
	}

	if _, _, err := db.Run(ctx, "COMMIT;"); err != nil {
		b.Fatal(err)
	}

	debug.FreeOSMemory()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var m int64
		rss, _, err := db.Execute(nil, sel)
		if err != nil {
			b.Fatal(err)
		}

		if err = rss[0].Do(false, func([]interface{}) (bool, error) {
			m++
			return true, nil
		}); err != nil {
			b.Fatal(err)
		}
		if g, e := n, m; g != e {
			b.Fatal(g, e)
		}
	}
	b.StopTimer()
	b.SetBytes(n * benchScale)
}

func benchmarkSelectBoolMem(b *testing.B, size int, sel float64, index bool) {
	db, err := OpenMem()
	if err != nil {
		b.Fatal(err)
	}

	benchmarkSelectBool(b, db, size, sel, index, nil)
}

// ----

func BenchmarkSelectBoolMemNoX1e1Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e1, 0.5, false)
}

func BenchmarkSelectBoolMemX1e1Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e1, 0.5, true)
}

func BenchmarkSelectBoolMemNoX1e2Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e2, 0.5, false)
}

func BenchmarkSelectBoolMemX1e2Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e2, 0.5, true)
}

func BenchmarkSelectBoolMemNoX1e3Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e3, 0.5, false)
}

func BenchmarkSelectBoolMemX1e3Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e3, 0.5, true)
}

func BenchmarkSelectBoolMemNoX1e4Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e4, 0.5, false)
}

func BenchmarkSelectBoolMemX1e4Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e4, 0.5, true)
}

func BenchmarkSelectBoolMemNoX1e5Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e5, 0.5, false)
}

func BenchmarkSelectBoolMemX1e5Perc50(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e5, 0.5, true)
}

// ----

func BenchmarkSelectBoolMemNoX1e1Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e1, 0.05, false)
}

func BenchmarkSelectBoolMemX1e1Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e1, 0.05, true)
}

func BenchmarkSelectBoolMemNoX1e2Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e2, 0.05, false)
}

func BenchmarkSelectBoolMemX1e2Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e2, 0.05, true)
}

func BenchmarkSelectBoolMemNoX1e3Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e3, 0.05, false)
}

func BenchmarkSelectBoolMemX1e3Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e3, 0.05, true)
}

func BenchmarkSelectBoolMemNoX1e4Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e4, 0.05, false)
}

func BenchmarkSelectBoolMemX1e4Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e4, 0.05, true)
}

func BenchmarkSelectBoolMemNoX1e5Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e5, 0.05, false)
}

func BenchmarkSelectBoolMemX1e5Perc5(b *testing.B) {
	benchmarkSelectBoolMem(b, 1e5, 0.05, true)
}

func benchmarkSelectBoolFile(b *testing.B, size int, sel float64, index bool) {
	dir, err := ioutil.TempDir("", "ql-bench-")
	if err != nil {
		b.Fatal(err)
	}

	n := runtime.GOMAXPROCS(0)
	db, err := OpenFile(filepath.Join(dir, "ql.db"), &Options{CanCreate: true})
	if err != nil {
		b.Fatal(err)
	}

	benchmarkSelectBool(b, db, size, sel, index, func() {
		runtime.GOMAXPROCS(n)
		db.Close()
		os.RemoveAll(dir)
	})
}

// ----

func BenchmarkSelectBoolFileNoX1e1Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e1, 0.5, false)
}

func BenchmarkSelectBoolFileX1e1Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e1, 0.5, true)
}

func BenchmarkSelectBoolFileNoX1e2Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e2, 0.5, false)
}

func BenchmarkSelectBoolFileX1e2Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e2, 0.5, true)
}

func BenchmarkSelectBoolFileNoX1e3Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e3, 0.5, false)
}

func BenchmarkSelectBoolFileX1e3Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e3, 0.5, true)
}

func BenchmarkSelectBoolFileNoX1e4Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e4, 0.5, false)
}

func BenchmarkSelectBoolFileX1e4Perc50(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e4, 0.5, true)
}

// ----

func BenchmarkSelectBoolFileNoX1e1Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e1, 0.05, false)
}

func BenchmarkSelectBoolFileX1e1Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e1, 0.05, true)
}

func BenchmarkSelectBoolFileNoX1e2Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e2, 0.05, false)
}

func BenchmarkSelectBoolFileX1e2Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e2, 0.05, true)
}

func BenchmarkSelectBoolFileNoX1e3Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e3, 0.05, false)
}

func BenchmarkSelectBoolFileX1e3Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e3, 0.05, true)
}

func BenchmarkSelectBoolFileNoX1e4Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e4, 0.05, false)
}

func BenchmarkSelectBoolFileX1e4Perc5(b *testing.B) {
	benchmarkSelectBoolFile(b, 1e4, 0.05, true)
}

func TestIndex(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	ctx := NewRWCtx()
	if _, _, err := db.Run(ctx, `
		BEGIN TRANSACTION;
			CREATE TABLE t (b bool);
	`); err != nil {
		t.Fatal(err)
	}

	if _, _, err := db.Run(ctx, `
			CREATE INDEX x ON t (b);
		`); err != nil {
		t.Fatal(err)
	}

	ins, err := Compile("INSERT INTO t VALUES($1);")
	if err != nil {
		t.Fatal(err)
	}

	size, selectivity := int(1e1), 0.5
	rng := rand.New(rand.NewSource(42))
	var n int64
	for j := 0; j < size; j++ {
		v := rng.Float64() < selectivity
		if v {
			n++
			t.Logf("id %d <- true", j+1)
		}
		if _, _, err = db.Execute(ctx, ins, v); err != nil {
			t.Fatal(err)
		}
	}

	if _, _, err := db.Run(ctx, "COMMIT;"); err != nil {
		t.Fatal(err)
	}

	s, err := dumpDB(db, "")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("n: %d\n%s", n, s)
	sel, err := Compile("SELECT id(), b FROM t WHERE b;")
	if err != nil {
		t.Fatal(err)
	}

	var m int64
	rss, _, err := db.Execute(nil, sel)
	if err != nil {
		t.Fatal(err)
	}

	if err = rss[0].Do(false, func(rec []interface{}) (bool, error) {
		t.Logf("%v", rec)
		m++
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	if g, e := n, m; g != e {
		t.Fatal(g, e)
	}
}

var benchmarkCrossJoinOnce = map[string]sync.Once{}

func benchmarkCrossJoin(b *testing.B, db *DB, create, sel List, size1, size2 int, index bool, teardown func()) {
	if testing.Verbose() {
		benchProlog(b)
		id := fmt.Sprintf("%t|%d|%d|%t", db.isMem, size1, size2, index)
		once := benchmarkCrossJoinOnce[id]
		once.Do(func() {
			s := "INDEXED "
			if !index {
				s = "NON " + s
			}
			b.Logf(`Fill two %stables with %d and %d records of random numbers [0, 1). Measure the performance of
%s
`, s, size1, size2, sel)
		})
		benchmarkCrossJoinOnce[id] = once
	}

	if teardown != nil {
		defer teardown()
	}

	ctx := NewRWCtx()
	if _, _, err := db.Execute(ctx, create); err != nil {
		b.Fatal(err)
	}

	if index {
		if _, _, err := db.Execute(ctx, xjoinX); err != nil {
			b.Fatal(err)
		}
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < size1; i++ {
		if _, _, err := db.Execute(ctx, xjoinT, rng.Float64()); err != nil {
			b.Fatal(err)
		}
	}
	for i := 0; i < size2; i++ {
		if _, _, err := db.Execute(ctx, xjoinU, rng.Float64()); err != nil {
			b.Fatal(err)
		}
	}

	if _, _, err := db.Run(ctx, "COMMIT"); err != nil {
		b.Fatal(err)
	}

	rs, _, err := db.Execute(nil, sel)
	if err != nil {
		b.Fatal(err)
	}

	var n int
	debug.FreeOSMemory()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n = 0
		if err := rs[0].Do(false, func(rec []interface{}) (bool, error) {
			n++
			return true, nil
		}); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.SetBytes(int64(n) * benchScale)
}

var (
	xjoinCreate = MustCompile(`BEGIN TRANSACTION;
	CREATE TABLE t (f float);
	CREATE TABLE u (f float);`)
	xjoinSel = MustCompile(`SELECT *  FROM (SELECT f FROM t WHERE f < 0.1), (SELECT f FROM u where f < 0.1);`)
	xjoinT   = MustCompile("INSERT INTO t VALUES($1);")
	xjoinU   = MustCompile("INSERT INTO u VALUES($1);")
	xjoinX   = MustCompile(`CREATE INDEX x ON t (f); CREATE INDEX y ON u (f);`)
)

func benchmarkCrossJoinMem(b *testing.B, size1, size2 int, index bool) {
	db, err := OpenMem()
	if err != nil {
		b.Fatal(err)
	}

	benchmarkCrossJoin(b, db, xjoinCreate, xjoinSel, size1, size2, index, nil)
}

func benchmarkCrossJoinFile(b *testing.B, size1, size2 int, index bool) {
	dir, err := ioutil.TempDir("", "ql-bench-")
	if err != nil {
		b.Fatal(err)
	}

	n := runtime.GOMAXPROCS(0)
	db, err := OpenFile(filepath.Join(dir, "ql.db"), &Options{CanCreate: true})
	if err != nil {
		b.Fatal(err)
	}

	benchmarkCrossJoin(b, db, xjoinCreate, xjoinSel, size1, size2, index, func() {
		runtime.GOMAXPROCS(n)
		db.Close()
		os.RemoveAll(dir)
	})
}

func BenchmarkCrossJoinMem1e1NoX1e2(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e1, 1e2, false)
}

func BenchmarkCrossJoinMem1e1X1e2(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e1, 1e2, true)
}

func BenchmarkCrossJoinMem1e2NoX1e3(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e2, 1e3, false)
}

func BenchmarkCrossJoinMem1e2X1e3(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e2, 1e3, true)
}

func BenchmarkCrossJoinMem1e3NoX1e4(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e3, 1e4, false)
}

func BenchmarkCrossJoinMem1e3X1e4(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e3, 1e4, true)
}

func BenchmarkCrossJoinMem1e2NoX1e1(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e2, 1e1, false)
}

func BenchmarkCrossJoinMem1e2X1e1(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e2, 1e1, true)
}

func BenchmarkCrossJoinMem1e3NoX1e2(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e3, 1e2, false)
}

func BenchmarkCrossJoinMem1e3X1e2(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e3, 1e2, true)
}

func BenchmarkCrossJoinMem1e4NoX1e3(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e4, 1e3, false)
}

func BenchmarkCrossJoinMem1e4X1e3(b *testing.B) {
	benchmarkCrossJoinMem(b, 1e4, 1e3, true)
}

// ----

func BenchmarkCrossJoinFile1e1NoX1e2(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e1, 1e2, false)
}

func BenchmarkCrossJoinFile1e1X1e2(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e1, 1e2, true)
}

func BenchmarkCrossJoinFile1e2NoX1e3(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e2, 1e3, false)
}

func BenchmarkCrossJoinFile1e2X1e3(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e2, 1e3, true)
}

func BenchmarkCrossJoinFile1e3NoX1e4(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e3, 1e4, false)
}

func BenchmarkCrossJoinFile1e3X1e4(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e3, 1e4, true)
}

func BenchmarkCrossJoinFile1e2NoX1e1(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e2, 1e1, false)
}

func BenchmarkCrossJoinFile1e2X1e1(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e2, 1e1, true)
}

func BenchmarkCrossJoinFile1e3NoX1e2(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e3, 1e2, false)
}

func BenchmarkCrossJoinFile1e3X1e2(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e3, 1e2, true)
}

func BenchmarkCrossJoinFile1e4NoX1e3(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e4, 1e3, false)
}

func BenchmarkCrossJoinFile1e4X1e3(b *testing.B) {
	benchmarkCrossJoinFile(b, 1e4, 1e3, true)
}

func TestIssue35(t *testing.T) {
	var bigInt big.Int
	var bigRat big.Rat
	bigInt.SetInt64(42)
	bigRat.SetInt64(24)
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	ctx := NewRWCtx()
	_, _, err = db.Run(ctx, `
	BEGIN TRANSACTION;
		CREATE TABLE t (i bigint, r bigrat);
		INSERT INTO t VALUES ($1, $2);
	COMMIT;
	`, bigInt, bigRat)
	if err != nil {
		t.Fatal(err)
	}

	bigInt.SetInt64(420)
	bigRat.SetInt64(240)

	rs, _, err := db.Run(nil, "SELECT * FROM t;")
	if err != nil {
		t.Fatal(err)
	}

	n := 0
	if err := rs[0].Do(false, func(rec []interface{}) (bool, error) {
		switch n {
		case 0:
			n++
			if g, e := fmt.Sprint(rec), "[42 24/1]"; g != e {
				t.Fatal(g, e)
			}

			return true, nil
		default:
			t.Fatal(n)
			panic("unreachable")
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func TestIssue28(t *testing.T) {
	RegisterDriver()
	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)
	pth := filepath.Join(dir, "ql.db")
	sdb, err := sql.Open("ql", "file://"+pth)
	if err != nil {
		t.Fatal(err)
	}

	defer sdb.Close()
	tx, err := sdb.Begin()
	if err != nil {
		t.Fatal(err)
	}

	if _, err = tx.Exec("CREATE TABLE t (i int);"); err != nil {
		t.Fatal(err)
	}

	if err = tx.Commit(); err != nil {
		t.Fatal(err)
	}

	if _, err = os.Stat(pth); err != nil {
		t.Fatal(err)
	}

	pth = filepath.Join(dir, "mem.db")
	mdb, err := sql.Open("ql", "memory://"+pth)
	if err != nil {
		t.Fatal(err)
	}

	defer mdb.Close()
	if tx, err = mdb.Begin(); err != nil {
		t.Fatal(err)
	}

	if _, err = tx.Exec("CREATE TABLE t (i int);"); err != nil {
		t.Fatal(err)
	}

	if err = tx.Commit(); err != nil {
		t.Fatal(err)
	}

	if _, err = os.Stat(pth); err == nil {
		t.Fatal(err)
	}
}

func TestIsPossiblyRewriteableCrossJoinWhereExpression(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	table := []struct {
		q     string
		e     bool
		slist string
	}{
		// 0
		{"SELECT * FROM t WHERE !c", false, ""},
		{"SELECT * FROM t WHERE !t.c && 4 < !u.c", true, "!c|4<!c"},
		{"SELECT * FROM t WHERE !t.c && 4 < u.c", true, "!c|4<c"},
		{"SELECT * FROM t WHERE !t.c", true, "!c"},
		{"SELECT * FROM t WHERE 3 < c", false, ""},
		// 5
		{"SELECT * FROM t WHERE 3 < t.c", true, "3<c"},
		{"SELECT * FROM t WHERE c && c", false, ""},
		{"SELECT * FROM t WHERE c && u.c", false, ""},
		{"SELECT * FROM t WHERE c == 42", false, ""},
		{"SELECT * FROM t WHERE c > 3", false, ""},
		// 10
		{"SELECT * FROM t WHERE c", false, ""},
		{"SELECT * FROM t WHERE false == !t.c", true, "false==!c"}, //TODO(indices) support !c relOp fixedValue (rewrite false==!c -> !c, true==c -> c, true != c -> !c, etc.)
		{"SELECT * FROM t WHERE false == ^t.c", false, ""},
		{"SELECT * FROM t WHERE false == t.c", true, "false==c"},
		{"SELECT * FROM t WHERE t.c && 4 < u.c", true, "c|4<c"},
		// 15
		{"SELECT * FROM t WHERE t.c && c", false, ""},
		{"SELECT * FROM t WHERE t.c && u.c && v.c > 0", true, "c|c|c>0"},
		{"SELECT * FROM t WHERE t.c && u.c && v.c", true, "c|c|c"},
		{"SELECT * FROM t WHERE t.c && u.c > 0 && v.c > 0", true, "c|c>0|c>0"},
		{"SELECT * FROM t WHERE t.c && u.c", true, "c|c"},
		// 20
		{"SELECT * FROM t WHERE t.c < 3 && u.c > 2 && v.c != 42", true, "c<3|c>2|c!=42"},
		{"SELECT * FROM t WHERE t.c > 0 && u.c && v.c", true, "c>0|c|c"},
		{"SELECT * FROM t WHERE t.c > 0 && u.c > 0 && v.c > 0", true, "c>0|c>0|c>0"},
		{"SELECT * FROM t WHERE t.c > 3 && 4 < u.c", true, "c>3|4<c"},
		{"SELECT * FROM t WHERE t.c > 3 && u.c", true, "c>3|c"},
		// 25
		{"SELECT * FROM t WHERE t.c > 3", true, "c>3"},
		{"SELECT * FROM t WHERE t.c", true, "c"},
		{"SELECT * FROM t WHERE u.c == !t.c", false, ""},
		{"SELECT * FROM t WHERE u.c == 42", true, "c==42"},
		{"SELECT * FROM t WHERE u.c == ^t.c", false, ""},
	}

	for i, test := range table {
		q, e, list := test.q, test.e, strings.Split(test.slist, "|")
		sort.Strings(list)
		l, err := Compile(q)
		if err != nil {
			t.Fatalf("%s\n%v", q, err)
		}

		rs, _, err := db.Execute(nil, l)
		if err != nil {
			t.Fatalf("%s\n%v", q, err)
		}

		r := rs[0].(recordset)
		sel := r.rset.(*selectRset)
		where := sel.src.(*whereRset)
		g, glist := isPossiblyRewriteableCrossJoinWhereExpression(where.expr)
		if g != e {
			t.Fatalf("%d: %sg: %v e: %v", i, l, g, e)
		}

		if !g {
			continue
		}

		a := []string{}
		for _, v := range glist {
			a = append(a, v.expr.String())
		}
		sort.Strings(a)
		if g, e := len(glist), len(list); g != e {
			t.Fatalf("%d: g: %v, e: %v", i, glist, list)
		}

		for j, g := range a {
			if e := list[j]; g != e {
				t.Fatalf("%d[%d]: g: %v e: %v", i, j, g, e)
			}
		}
	}
}

func dumpFields(f []*fld) string {
	a := []string{}
	for _, v := range f {
		a = append(a, fmt.Sprintf("%p: %q", v, v.name))
	}
	return strings.Join(a, ", ")
}

func rndBytes(n int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Int())
	}
	return b
}

func TestIssue50(t *testing.T) { // https://github.com/cznic/ql/issues/50
	const dbFileName = "scans.qldb"

	type Scan struct {
		ID        int
		Jobname   string
		Timestamp time.Time
		Data      []byte

		X, Y, Z            float64
		Alpha, Beta, Gamma float64
	}

	// querys
	const dbCreateTables = `
CREATE TABLE IF NOT EXISTS Scans (
	X float,
	Y float,
	Z float,
	Alpha float,
	Beta float,
	Gamma float,
	Timestamp time,
	Jobname string,
	Data blob
);
CREATE INDEX IF NOT EXISTS ScansId on Scans (id());
`

	const dbInsertScan = `
INSERT INTO Scans (Timestamp,Jobname,X,Y,Z,Alpha,Beta,Gamma,Data) VALUES(
$1,
$2,
$3,$4,$5,
$6,$7,$8,
$9
);
`

	const dbSelectOverview = `SELECT id() as ID, Jobname, Timestamp, Data, Y,Z, Gamma From Scans;`

	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// create db
	t.Log("Opening db.")
	RegisterDriver()
	db, err := sql.Open("ql", filepath.Join(dir, dbFileName))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		return
	}
	_, err = tx.Exec(dbCreateTables)
	if err != nil {
		t.Fatal("could not create Table.", err)
	}

	err = tx.Commit()
	if err != nil {
		t.Fatal("could not commit transaction.", err)
	}

	// insert some data
	tx, err = db.Begin()
	if err != nil {
		t.Fatalf("db.Begin() Error - %v", err)
	}

	stmt, err := tx.Prepare(dbInsertScan)
	if err != nil {
		t.Fatalf("tx.Prepare(dbInsertScan) Error - %v", err)
	}
	defer stmt.Close()

	scanFnames := []string{"1.xyz", "2.xyz", "3.xyz"}
	for _, fname := range scanFnames {
		scanData, err := ioutil.ReadFile(filepath.Join("_testdata", fname))
		if err != nil {
			t.Fatalf("ioutil.ReadFile(%s) Error - %v", fname, err)
		}

		// hash before insert
		h := md5.New()
		h.Write(scanData)

		t.Logf("md5 of %s: %x", fname, h.Sum(nil))

		_, err = stmt.Exec(time.Now(), "Job-0815", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, scanData)
		if err != nil {
			t.Fatalf("stmt.Exec() Error - %v", err)
			return
		}
	}

	err = tx.Commit()
	if err != nil {
		t.Fatalf("tx.Commit() Error - %v", err)
	}

	// select the inserted data
	rows, err := db.Query(dbSelectOverview)
	if err != nil {
		t.Fatalf("db.Query(dbSelectOverview) Error - %v", err)
	}
	defer rows.Close()

	var scans []Scan
	for rows.Next() {
		var s Scan
		var data []byte

		err = rows.Scan(&s.ID, &s.Jobname, &s.Timestamp, &data, &s.Y, &s.Z, &s.Gamma)
		if err != nil {
			t.Fatalf("rows.Scan(&s..) Error - %v", err)
		}
		scans = append(scans, s)

		// hash after select
		h := md5.New()
		h.Write(data)

		t.Logf("md5 of %d: %x", s.ID, h.Sum(nil))
	}

	err = rows.Err()
	if err != nil {
		t.Fatalf("rows.Err() Error - %v", err)
	}

	t.Log("Done:", scans)
}

func TestIssue56(t *testing.T) {
	var schema = `
CREATE TABLE IF NOT EXISTS Test (
	A string,
	B string,
	Suppressed bool,
);
CREATE INDEX IF NOT EXISTS aIdx ON Test (A);
CREATE INDEX IF NOT EXISTS bIdx ON Test (B);
`

	RegisterDriver()
	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)
	pth := filepath.Join(dir, "test.db")
	db, err := sql.Open("ql", "file://"+pth)
	if err != nil {
		t.Fatal(err)
	}

	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	_, err = tx.Exec(schema)
	if err != nil {
		t.Fatal(err)
	}

	err = tx.Commit()
	if err != nil {
		t.Fatal(err)
	}

	// Open a new transaction and do a SELECT

	tx, err = db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	var id int64
	row := tx.QueryRow("SELECT * FROM Test")
	err = row.Scan(&id) // <-- Blocks here

	switch err {
	case sql.ErrNoRows:
		break
	case nil:
		break
	default:
		t.Fatal(err)
	}

	tx.Rollback()
	return
}

func TestRecordSetRows(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	rss, _, err := db.Run(NewRWCtx(), `
		BEGIN TRANSACTION;
			CREATE TABLE t (i int);
			INSERT INTO t VALUES (1), (2), (3), (4), (5);
		COMMIT;
		SELECT * FROM t ORDER BY i;
	`)
	if err != nil {
		t.Fatal(err)
	}

	tab := []struct {
		limit, offset int
		result        []int64
	}{
		// 0
		{-1, 0, []int64{1, 2, 3, 4, 5}},
		{0, 0, nil},
		{1, 0, []int64{1}},
		{2, 0, []int64{1, 2}},
		{3, 0, []int64{1, 2, 3}},
		// 5
		{4, 0, []int64{1, 2, 3, 4}},
		{5, 0, []int64{1, 2, 3, 4, 5}},
		{6, 0, []int64{1, 2, 3, 4, 5}},
		{-1, 0, []int64{1, 2, 3, 4, 5}},
		{-1, 1, []int64{2, 3, 4, 5}},
		// 10
		{-1, 2, []int64{3, 4, 5}},
		{-1, 3, []int64{4, 5}},
		{-1, 4, []int64{5}},
		{-1, 5, nil},
		{3, 0, []int64{1, 2, 3}},
		// 15
		{3, 1, []int64{2, 3, 4}},
		{3, 2, []int64{3, 4, 5}},
		{3, 3, []int64{4, 5}},
		{3, 4, []int64{5}},
		{3, 5, nil},
		// 20
		{-1, 2, []int64{3, 4, 5}},
		{0, 2, nil},
		{1, 2, []int64{3}},
		{2, 2, []int64{3, 4}},
		{3, 2, []int64{3, 4, 5}},
		// 25
		{4, 2, []int64{3, 4, 5}},
	}

	rs := rss[0]
	for iTest, test := range tab {
		t.Log(iTest)
		rows, err := rs.Rows(test.limit, test.offset)
		if err != nil {
			t.Fatal(err)
		}

		if g, e := len(rows), len(test.result); g != e {
			t.Log(rows, test.result)
			t.Fatal(g, e)
		}

		for i, row := range rows {
			if g, e := len(row), 1; g != e {
				t.Fatal(i, g, i)
			}

			if g, e := row[0], test.result[i]; g != e {
				t.Fatal(i, g, e)
			}
		}
	}
}

func TestRecordFirst(t *testing.T) {
	q := MustCompile("SELECT * FROM t WHERE i > $1 ORDER BY i;")
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err = db.Run(NewRWCtx(), `
		BEGIN TRANSACTION;
			CREATE TABLE t (i int);
			INSERT INTO t VALUES (1), (2), (3), (4), (5);
		COMMIT;
	`); err != nil {
		t.Fatal(err)
	}

	tab := []struct {
		par    int64
		result int64
	}{
		{-1, 1},
		{0, 1},
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
		{5, -1},
	}

	for iTest, test := range tab {
		t.Log(iTest)
		rss, _, err := db.Execute(nil, q, test.par)
		if err != nil {
			t.Fatal(err)
		}

		row, err := rss[0].FirstRow()
		if err != nil {
			t.Fatal(err)
		}

		switch {
		case test.result < 0:
			if row != nil {
				t.Fatal(row)
			}
		default:
			if row == nil {
				t.Fatal(row)
			}

			if g, e := len(row), 1; g != e {
				t.Fatal(g, e)
			}

			if g, e := row[0], test.result; g != e {
				t.Fatal(g, e)
			}
		}
	}
}

var issue63 = MustCompile(`
BEGIN TRANSACTION;
	CREATE TABLE Forecast (WeatherProvider string, Timestamp time, MinTemp int32, MaxTemp int32);
	INSERT INTO Forecast VALUES ("dwd.de", now(), 20, 22);
COMMIT;
SELECT * FROM Forecast WHERE Timestamp > 0;`)

func TestIssue63(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	rs, _, err := db.Execute(NewRWCtx(), issue63)
	if err != nil {
		t.Fatal(err)
	}

	if _, err = rs[0].Rows(-1, 0); err == nil {
		t.Fatal(err)
	}

	t.Log(err)
	if g, e := strings.Contains(err.Error(), "invalid operation"), true; g != e {
		t.Fatal(g, e)
	}

	if g, e := strings.Contains(err.Error(), "mismatched types time.Time and int64"), true; g != e {
		t.Fatal(g, e)
	}
}

const issue66Src = `
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE UNIQUE INDEX x ON t (i);
	INSERT INTO t VALUES (1), (1);
COMMIT;`

var issue66 = MustCompile(issue66Src)

func TestIssue66Mem(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = db.Execute(NewRWCtx(), issue66)
	if err == nil {
		t.Fatal(err)
	}

	t.Log(err)
}

func TestIssue66File(t *testing.T) {
	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)

	db, err := OpenFile(filepath.Join(dir, "test.db"), &Options{CanCreate: true})
	if err != nil {
		t.Fatal(err)
	}

	defer db.Close()

	_, _, err = db.Execute(NewRWCtx(), issue66)
	if err == nil {
		t.Fatal(err)
	}

	t.Log(err)
}

func TestIssue66MemDriver(t *testing.T) {
	RegisterMemDriver()
	db, err := sql.Open("ql-mem", "TestIssue66MemDriver-"+fmt.Sprintf("%d", time.Now().UnixNano()))
	if err != nil {
		t.Fatal(err)
	}

	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	if _, err = tx.Exec(issue66Src); err == nil {
		t.Fatal(err)
	}

	t.Log(err)
}

func TestIssue66FileDriver(t *testing.T) {
	RegisterDriver()
	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)

	db, err := sql.Open("ql", filepath.Join(dir, "TestIssue66MemDriver"))
	if err != nil {
		t.Fatal(err)
	}

	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	if _, err = tx.Exec(issue66Src); err == nil {
		t.Fatal(err)
	}

	t.Log(err)
}

func Example_lIKE() {
	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	rss, _, err := db.Run(NewRWCtx(), `
        BEGIN TRANSACTION;
            CREATE TABLE t (i int, s string);
            INSERT INTO t VALUES
            	(1, "seafood"),
            	(2, "A fool on the hill"),
            	(3, NULL),
            	(4, "barbaz"),
            	(5, "foobar"),
            ;
        COMMIT;
        
        SELECT * FROM t WHERE s LIKE "foo" ORDER BY i;
        SELECT * FROM t WHERE s LIKE "^bar" ORDER BY i;
        SELECT * FROM t WHERE s LIKE "bar$" ORDER BY i;
        SELECT * FROM t WHERE !(s LIKE "foo") ORDER BY i;`,
	)
	if err != nil {
		panic(err)
	}

	for _, rs := range rss {
		if err := rs.Do(false, func(data []interface{}) (bool, error) {
			fmt.Println(data)
			return true, nil
		}); err != nil {
			panic(err)
		}
		fmt.Println("----")
	}
	// Output:
	// [1 seafood]
	// [2 A fool on the hill]
	// [5 foobar]
	// ----
	// [4 barbaz]
	// ----
	// [5 foobar]
	// ----
	// [4 barbaz]
	// ----
}

func TestIssue73(t *testing.T) {
	RegisterDriver()
	dir, err := ioutil.TempDir("", "ql-test-")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir)
	pth := filepath.Join(dir, "test.db")

	for i := 0; i < 10; i++ {
		var db *sql.DB
		var tx *sql.Tx
		var err error
		var row *sql.Row
		var name string

		if db, err = sql.Open("ql", pth); err != nil {
			t.Fatal("sql.Open: ", err)
		}

		t.Log("Call to db.Begin()")
		if tx, err = db.Begin(); err != nil {
			t.Fatal("db.Begin: ", err)
		}

		t.Log("Call to tx.QueryRow()")
		row = tx.QueryRow(`SELECT Name FROM __Table`)
		t.Log("Call to tx.Commit()")
		if err = tx.Commit(); err != nil {
			t.Fatal("tx.Commit: ", err)
		}

		row.Scan(&name)
		t.Log("name: ", name)
	}
}

func Example_id() {
	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	rss, _, err := db.Run(NewRWCtx(), `
	BEGIN TRANSACTION;
		CREATE TABLE foo (i int);
		INSERT INTO foo VALUES (10), (20);
		CREATE TABLE bar (fooID int, s string);
		INSERT INTO bar SELECT id(), "ten" FROM foo WHERE i == 10;
		INSERT INTO bar SELECT id(), "twenty" FROM foo WHERE i == 20;
	COMMIT;
	SELECT *
	FROM foo, bar
	WHERE bar.fooID == id(foo)
	ORDER BY id(foo);`,
	)
	if err != nil {
		panic(err)
	}

	for _, rs := range rss {
		if err := rs.Do(false, func(data []interface{}) (bool, error) {
			fmt.Println(data)
			return true, nil
		}); err != nil {
			panic(err)
		}
		fmt.Println("----")
	}
	// Output:
	// [10 1 ten]
	// [20 2 twenty]
	// ----
}
