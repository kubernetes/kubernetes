package bolt_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/boltdb/bolt"
)

var statsFlag = flag.Bool("stats", false, "show performance stats")

// version is the data file format version.
const version = 2

// magic is the marker value to indicate that a file is a Bolt DB.
const magic uint32 = 0xED0CDAED

// pageSize is the size of one page in the data file.
const pageSize = 4096

// pageHeaderSize is the size of a page header.
const pageHeaderSize = 16

// meta represents a simplified version of a database meta page for testing.
type meta struct {
	magic    uint32
	version  uint32
	_        uint32
	_        uint32
	_        [16]byte
	_        uint64
	pgid     uint64
	_        uint64
	checksum uint64
}

// Ensure that a database can be opened without error.
func TestOpen(t *testing.T) {
	path := tempfile()
	db, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	} else if db == nil {
		t.Fatal("expected db")
	}

	if s := db.Path(); s != path {
		t.Fatalf("unexpected path: %s", s)
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that opening a database with a blank path returns an error.
func TestOpen_ErrPathRequired(t *testing.T) {
	_, err := bolt.Open("", 0666, nil)
	if err == nil {
		t.Fatalf("expected error")
	}
}

// Ensure that opening a database with a bad path returns an error.
func TestOpen_ErrNotExists(t *testing.T) {
	_, err := bolt.Open(filepath.Join(tempfile(), "bad-path"), 0666, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

// Ensure that opening a file that is not a Bolt database returns ErrInvalid.
func TestOpen_ErrInvalid(t *testing.T) {
	path := tempfile()

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintln(f, "this is not a bolt database"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(path)

	if _, err := bolt.Open(path, 0666, nil); err != bolt.ErrInvalid {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that opening a file with two invalid versions returns ErrVersionMismatch.
func TestOpen_ErrVersionMismatch(t *testing.T) {
	if pageSize != os.Getpagesize() {
		t.Skip("page size mismatch")
	}

	// Create empty database.
	db := MustOpenDB()
	path := db.Path()
	defer db.MustClose()

	// Close database.
	if err := db.DB.Close(); err != nil {
		t.Fatal(err)
	}

	// Read data file.
	buf, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Rewrite meta pages.
	meta0 := (*meta)(unsafe.Pointer(&buf[pageHeaderSize]))
	meta0.version++
	meta1 := (*meta)(unsafe.Pointer(&buf[pageSize+pageHeaderSize]))
	meta1.version++
	if err := ioutil.WriteFile(path, buf, 0666); err != nil {
		t.Fatal(err)
	}

	// Reopen data file.
	if _, err := bolt.Open(path, 0666, nil); err != bolt.ErrVersionMismatch {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that opening a file with two invalid checksums returns ErrChecksum.
func TestOpen_ErrChecksum(t *testing.T) {
	if pageSize != os.Getpagesize() {
		t.Skip("page size mismatch")
	}

	// Create empty database.
	db := MustOpenDB()
	path := db.Path()
	defer db.MustClose()

	// Close database.
	if err := db.DB.Close(); err != nil {
		t.Fatal(err)
	}

	// Read data file.
	buf, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Rewrite meta pages.
	meta0 := (*meta)(unsafe.Pointer(&buf[pageHeaderSize]))
	meta0.pgid++
	meta1 := (*meta)(unsafe.Pointer(&buf[pageSize+pageHeaderSize]))
	meta1.pgid++
	if err := ioutil.WriteFile(path, buf, 0666); err != nil {
		t.Fatal(err)
	}

	// Reopen data file.
	if _, err := bolt.Open(path, 0666, nil); err != bolt.ErrChecksum {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that opening an already open database file will timeout.
func TestOpen_Timeout(t *testing.T) {
	if runtime.GOOS == "solaris" {
		t.Skip("solaris fcntl locks don't support intra-process locking")
	}

	path := tempfile()

	// Open a data file.
	db0, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	} else if db0 == nil {
		t.Fatal("expected database")
	}

	// Attempt to open the database again.
	start := time.Now()
	db1, err := bolt.Open(path, 0666, &bolt.Options{Timeout: 100 * time.Millisecond})
	if err != bolt.ErrTimeout {
		t.Fatalf("unexpected timeout: %s", err)
	} else if db1 != nil {
		t.Fatal("unexpected database")
	} else if time.Since(start) <= 100*time.Millisecond {
		t.Fatal("expected to wait at least timeout duration")
	}

	if err := db0.Close(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that opening an already open database file will wait until its closed.
func TestOpen_Wait(t *testing.T) {
	if runtime.GOOS == "solaris" {
		t.Skip("solaris fcntl locks don't support intra-process locking")
	}

	path := tempfile()

	// Open a data file.
	db0, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Close it in just a bit.
	time.AfterFunc(100*time.Millisecond, func() { _ = db0.Close() })

	// Attempt to open the database again.
	start := time.Now()
	db1, err := bolt.Open(path, 0666, &bolt.Options{Timeout: 200 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	} else if time.Since(start) <= 100*time.Millisecond {
		t.Fatal("expected to wait at least timeout duration")
	}

	if err := db1.Close(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that opening a database does not increase its size.
// https://github.com/boltdb/bolt/issues/291
func TestOpen_Size(t *testing.T) {
	// Open a data file.
	db := MustOpenDB()
	path := db.Path()
	defer db.MustClose()

	pagesize := db.Info().PageSize

	// Insert until we get above the minimum 4MB size.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, _ := tx.CreateBucketIfNotExists([]byte("data"))
		for i := 0; i < 10000; i++ {
			if err := b.Put([]byte(fmt.Sprintf("%04d", i)), make([]byte, 1000)); err != nil {
				t.Fatal(err)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Close database and grab the size.
	if err := db.DB.Close(); err != nil {
		t.Fatal(err)
	}
	sz := fileSize(path)
	if sz == 0 {
		t.Fatalf("unexpected new file size: %d", sz)
	}

	// Reopen database, update, and check size again.
	db0, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db0.Update(func(tx *bolt.Tx) error {
		if err := tx.Bucket([]byte("data")).Put([]byte{0}, []byte{0}); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := db0.Close(); err != nil {
		t.Fatal(err)
	}
	newSz := fileSize(path)
	if newSz == 0 {
		t.Fatalf("unexpected new file size: %d", newSz)
	}

	// Compare the original size with the new size.
	// db size might increase by a few page sizes due to the new small update.
	if sz < newSz-5*int64(pagesize) {
		t.Fatalf("unexpected file growth: %d => %d", sz, newSz)
	}
}

// Ensure that opening a database beyond the max step size does not increase its size.
// https://github.com/boltdb/bolt/issues/303
func TestOpen_Size_Large(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}

	// Open a data file.
	db := MustOpenDB()
	path := db.Path()
	defer db.MustClose()

	pagesize := db.Info().PageSize

	// Insert until we get above the minimum 4MB size.
	var index uint64
	for i := 0; i < 10000; i++ {
		if err := db.Update(func(tx *bolt.Tx) error {
			b, _ := tx.CreateBucketIfNotExists([]byte("data"))
			for j := 0; j < 1000; j++ {
				if err := b.Put(u64tob(index), make([]byte, 50)); err != nil {
					t.Fatal(err)
				}
				index++
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	// Close database and grab the size.
	if err := db.DB.Close(); err != nil {
		t.Fatal(err)
	}
	sz := fileSize(path)
	if sz == 0 {
		t.Fatalf("unexpected new file size: %d", sz)
	} else if sz < (1 << 30) {
		t.Fatalf("expected larger initial size: %d", sz)
	}

	// Reopen database, update, and check size again.
	db0, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db0.Update(func(tx *bolt.Tx) error {
		return tx.Bucket([]byte("data")).Put([]byte{0}, []byte{0})
	}); err != nil {
		t.Fatal(err)
	}
	if err := db0.Close(); err != nil {
		t.Fatal(err)
	}

	newSz := fileSize(path)
	if newSz == 0 {
		t.Fatalf("unexpected new file size: %d", newSz)
	}

	// Compare the original size with the new size.
	// db size might increase by a few page sizes due to the new small update.
	if sz < newSz-5*int64(pagesize) {
		t.Fatalf("unexpected file growth: %d => %d", sz, newSz)
	}
}

// Ensure that a re-opened database is consistent.
func TestOpen_Check(t *testing.T) {
	path := tempfile()

	db, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.View(func(tx *bolt.Tx) error { return <-tx.Check() }); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.View(func(tx *bolt.Tx) error { return <-tx.Check() }); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that write errors to the meta file handler during initialization are returned.
func TestOpen_MetaInitWriteError(t *testing.T) {
	t.Skip("pending")
}

// Ensure that a database that is too small returns an error.
func TestOpen_FileTooSmall(t *testing.T) {
	path := tempfile()

	db, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// corrupt the database
	if err := os.Truncate(path, int64(os.Getpagesize())); err != nil {
		t.Fatal(err)
	}

	db, err = bolt.Open(path, 0666, nil)
	if err == nil || err.Error() != "file size too small" {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that a database can be opened in read-only mode by multiple processes
// and that a database can not be opened in read-write mode and in read-only
// mode at the same time.
func TestOpen_ReadOnly(t *testing.T) {
	if runtime.GOOS == "solaris" {
		t.Skip("solaris fcntl locks don't support intra-process locking")
	}

	bucket, key, value := []byte(`bucket`), []byte(`key`), []byte(`value`)

	path := tempfile()

	// Open in read-write mode.
	db, err := bolt.Open(path, 0666, nil)
	if err != nil {
		t.Fatal(err)
	} else if db.IsReadOnly() {
		t.Fatal("db should not be in read only mode")
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket(bucket)
		if err != nil {
			return err
		}
		if err := b.Put(key, value); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// Open in read-only mode.
	db0, err := bolt.Open(path, 0666, &bolt.Options{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}

	// Opening in read-write mode should return an error.
	if _, err = bolt.Open(path, 0666, &bolt.Options{Timeout: time.Millisecond * 100}); err == nil {
		t.Fatal("expected error")
	}

	// And again (in read-only mode).
	db1, err := bolt.Open(path, 0666, &bolt.Options{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}

	// Verify both read-only databases are accessible.
	for _, db := range []*bolt.DB{db0, db1} {
		// Verify is is in read only mode indeed.
		if !db.IsReadOnly() {
			t.Fatal("expected read only mode")
		}

		// Read-only databases should not allow updates.
		if err := db.Update(func(*bolt.Tx) error {
			panic(`should never get here`)
		}); err != bolt.ErrDatabaseReadOnly {
			t.Fatalf("unexpected error: %s", err)
		}

		// Read-only databases should not allow beginning writable txns.
		if _, err := db.Begin(true); err != bolt.ErrDatabaseReadOnly {
			t.Fatalf("unexpected error: %s", err)
		}

		// Verify the data.
		if err := db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket(bucket)
			if b == nil {
				return fmt.Errorf("expected bucket `%s`", string(bucket))
			}

			got := string(b.Get(key))
			expected := string(value)
			if got != expected {
				return fmt.Errorf("expected `%s`, got `%s`", expected, got)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	if err := db0.Close(); err != nil {
		t.Fatal(err)
	}
	if err := db1.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestDB_Open_InitialMmapSize tests if having InitialMmapSize large enough
// to hold data from concurrent write transaction resolves the issue that
// read transaction blocks the write transaction and causes deadlock.
// This is a very hacky test since the mmap size is not exposed.
func TestDB_Open_InitialMmapSize(t *testing.T) {
	path := tempfile()
	defer os.Remove(path)

	initMmapSize := 1 << 31  // 2GB
	testWriteSize := 1 << 27 // 134MB

	db, err := bolt.Open(path, 0666, &bolt.Options{InitialMmapSize: initMmapSize})
	if err != nil {
		t.Fatal(err)
	}

	// create a long-running read transaction
	// that never gets closed while writing
	rtx, err := db.Begin(false)
	if err != nil {
		t.Fatal(err)
	}

	// create a write transaction
	wtx, err := db.Begin(true)
	if err != nil {
		t.Fatal(err)
	}

	b, err := wtx.CreateBucket([]byte("test"))
	if err != nil {
		t.Fatal(err)
	}

	// and commit a large write
	err = b.Put([]byte("foo"), make([]byte, testWriteSize))
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})

	go func() {
		if err := wtx.Commit(); err != nil {
			t.Fatal(err)
		}
		done <- struct{}{}
	}()

	select {
	case <-time.After(5 * time.Second):
		t.Errorf("unexpected that the reader blocks writer")
	case <-done:
	}

	if err := rtx.Rollback(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a database cannot open a transaction when it's not open.
func TestDB_Begin_ErrDatabaseNotOpen(t *testing.T) {
	var db bolt.DB
	if _, err := db.Begin(false); err != bolt.ErrDatabaseNotOpen {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that a read-write transaction can be retrieved.
func TestDB_BeginRW(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	tx, err := db.Begin(true)
	if err != nil {
		t.Fatal(err)
	} else if tx == nil {
		t.Fatal("expected tx")
	}

	if tx.DB() != db.DB {
		t.Fatal("unexpected tx database")
	} else if !tx.Writable() {
		t.Fatal("expected writable tx")
	}

	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that opening a transaction while the DB is closed returns an error.
func TestDB_BeginRW_Closed(t *testing.T) {
	var db bolt.DB
	if _, err := db.Begin(true); err != bolt.ErrDatabaseNotOpen {
		t.Fatalf("unexpected error: %s", err)
	}
}

func TestDB_Close_PendingTx_RW(t *testing.T) { testDB_Close_PendingTx(t, true) }
func TestDB_Close_PendingTx_RO(t *testing.T) { testDB_Close_PendingTx(t, false) }

// Ensure that a database cannot close while transactions are open.
func testDB_Close_PendingTx(t *testing.T, writable bool) {
	db := MustOpenDB()
	defer db.MustClose()

	// Start transaction.
	tx, err := db.Begin(true)
	if err != nil {
		t.Fatal(err)
	}

	// Open update in separate goroutine.
	done := make(chan struct{})
	go func() {
		if err := db.Close(); err != nil {
			t.Fatal(err)
		}
		close(done)
	}()

	// Ensure database hasn't closed.
	time.Sleep(100 * time.Millisecond)
	select {
	case <-done:
		t.Fatal("database closed too early")
	default:
	}

	// Commit transaction.
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	// Ensure database closed now.
	time.Sleep(100 * time.Millisecond)
	select {
	case <-done:
	default:
		t.Fatal("database did not close")
	}
}

// Ensure a database can provide a transactional block.
func TestDB_Update(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("foo"), []byte("bar")); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("baz"), []byte("bat")); err != nil {
			t.Fatal(err)
		}
		if err := b.Delete([]byte("foo")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		if v := b.Get([]byte("foo")); v != nil {
			t.Fatalf("expected nil value, got: %v", v)
		}
		if v := b.Get([]byte("baz")); !bytes.Equal(v, []byte("bat")) {
			t.Fatalf("unexpected value: %v", v)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure a closed database returns an error while running a transaction block
func TestDB_Update_Closed(t *testing.T) {
	var db bolt.DB
	if err := db.Update(func(tx *bolt.Tx) error {
		if _, err := tx.CreateBucket([]byte("widgets")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != bolt.ErrDatabaseNotOpen {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure a panic occurs while trying to commit a managed transaction.
func TestDB_Update_ManualCommit(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var panicked bool
	if err := db.Update(func(tx *bolt.Tx) error {
		func() {
			defer func() {
				if r := recover(); r != nil {
					panicked = true
				}
			}()

			if err := tx.Commit(); err != nil {
				t.Fatal(err)
			}
		}()
		return nil
	}); err != nil {
		t.Fatal(err)
	} else if !panicked {
		t.Fatal("expected panic")
	}
}

// Ensure a panic occurs while trying to rollback a managed transaction.
func TestDB_Update_ManualRollback(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var panicked bool
	if err := db.Update(func(tx *bolt.Tx) error {
		func() {
			defer func() {
				if r := recover(); r != nil {
					panicked = true
				}
			}()

			if err := tx.Rollback(); err != nil {
				t.Fatal(err)
			}
		}()
		return nil
	}); err != nil {
		t.Fatal(err)
	} else if !panicked {
		t.Fatal("expected panic")
	}
}

// Ensure a panic occurs while trying to commit a managed transaction.
func TestDB_View_ManualCommit(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var panicked bool
	if err := db.View(func(tx *bolt.Tx) error {
		func() {
			defer func() {
				if r := recover(); r != nil {
					panicked = true
				}
			}()

			if err := tx.Commit(); err != nil {
				t.Fatal(err)
			}
		}()
		return nil
	}); err != nil {
		t.Fatal(err)
	} else if !panicked {
		t.Fatal("expected panic")
	}
}

// Ensure a panic occurs while trying to rollback a managed transaction.
func TestDB_View_ManualRollback(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var panicked bool
	if err := db.View(func(tx *bolt.Tx) error {
		func() {
			defer func() {
				if r := recover(); r != nil {
					panicked = true
				}
			}()

			if err := tx.Rollback(); err != nil {
				t.Fatal(err)
			}
		}()
		return nil
	}); err != nil {
		t.Fatal(err)
	} else if !panicked {
		t.Fatal("expected panic")
	}
}

// Ensure a write transaction that panics does not hold open locks.
func TestDB_Update_Panic(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	// Panic during update but recover.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Log("recover: update", r)
			}
		}()

		if err := db.Update(func(tx *bolt.Tx) error {
			if _, err := tx.CreateBucket([]byte("widgets")); err != nil {
				t.Fatal(err)
			}
			panic("omg")
		}); err != nil {
			t.Fatal(err)
		}
	}()

	// Verify we can update again.
	if err := db.Update(func(tx *bolt.Tx) error {
		if _, err := tx.CreateBucket([]byte("widgets")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Verify that our change persisted.
	if err := db.Update(func(tx *bolt.Tx) error {
		if tx.Bucket([]byte("widgets")) == nil {
			t.Fatal("expected bucket")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure a database can return an error through a read-only transactional block.
func TestDB_View_Error(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.View(func(tx *bolt.Tx) error {
		return errors.New("xxx")
	}); err == nil || err.Error() != "xxx" {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure a read transaction that panics does not hold open locks.
func TestDB_View_Panic(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		if _, err := tx.CreateBucket([]byte("widgets")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Panic during view transaction but recover.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Log("recover: view", r)
			}
		}()

		if err := db.View(func(tx *bolt.Tx) error {
			if tx.Bucket([]byte("widgets")) == nil {
				t.Fatal("expected bucket")
			}
			panic("omg")
		}); err != nil {
			t.Fatal(err)
		}
	}()

	// Verify that we can still use read transactions.
	if err := db.View(func(tx *bolt.Tx) error {
		if tx.Bucket([]byte("widgets")) == nil {
			t.Fatal("expected bucket")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that DB stats can be returned.
func TestDB_Stats(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	stats := db.Stats()
	if stats.TxStats.PageCount != 2 {
		t.Fatalf("unexpected TxStats.PageCount: %d", stats.TxStats.PageCount)
	} else if stats.FreePageN != 0 {
		t.Fatalf("unexpected FreePageN != 0: %d", stats.FreePageN)
	} else if stats.PendingPageN != 2 {
		t.Fatalf("unexpected PendingPageN != 2: %d", stats.PendingPageN)
	}
}

// Ensure that database pages are in expected order and type.
func TestDB_Consistency(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 10; i++ {
		if err := db.Update(func(tx *bolt.Tx) error {
			if err := tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar")); err != nil {
				t.Fatal(err)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		if p, _ := tx.Page(0); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "meta" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(1); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "meta" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(2); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "free" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(3); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "free" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(4); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "leaf" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(5); p == nil {
			t.Fatal("expected page")
		} else if p.Type != "freelist" {
			t.Fatalf("unexpected page type: %s", p.Type)
		}

		if p, _ := tx.Page(6); p != nil {
			t.Fatal("unexpected page")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that DB stats can be subtracted from one another.
func TestDBStats_Sub(t *testing.T) {
	var a, b bolt.Stats
	a.TxStats.PageCount = 3
	a.FreePageN = 4
	b.TxStats.PageCount = 10
	b.FreePageN = 14
	diff := b.Sub(&a)
	if diff.TxStats.PageCount != 7 {
		t.Fatalf("unexpected TxStats.PageCount: %d", diff.TxStats.PageCount)
	}

	// free page stats are copied from the receiver and not subtracted
	if diff.FreePageN != 14 {
		t.Fatalf("unexpected FreePageN: %d", diff.FreePageN)
	}
}

// Ensure two functions can perform updates in a single batch.
func TestDB_Batch(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		if _, err := tx.CreateBucket([]byte("widgets")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Iterate over multiple updates in separate goroutines.
	n := 2
	ch := make(chan error)
	for i := 0; i < n; i++ {
		go func(i int) {
			ch <- db.Batch(func(tx *bolt.Tx) error {
				return tx.Bucket([]byte("widgets")).Put(u64tob(uint64(i)), []byte{})
			})
		}(i)
	}

	// Check all responses to make sure there's no error.
	for i := 0; i < n; i++ {
		if err := <-ch; err != nil {
			t.Fatal(err)
		}
	}

	// Ensure data is correct.
	if err := db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		for i := 0; i < n; i++ {
			if v := b.Get(u64tob(uint64(i))); v == nil {
				t.Errorf("key not found: %d", i)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestDB_Batch_Panic(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var sentinel int
	var bork = &sentinel
	var problem interface{}
	var err error

	// Execute a function inside a batch that panics.
	func() {
		defer func() {
			if p := recover(); p != nil {
				problem = p
			}
		}()
		err = db.Batch(func(tx *bolt.Tx) error {
			panic(bork)
		})
	}()

	// Verify there is no error.
	if g, e := err, error(nil); g != e {
		t.Fatalf("wrong error: %v != %v", g, e)
	}
	// Verify the panic was captured.
	if g, e := problem, bork; g != e {
		t.Fatalf("wrong error: %v != %v", g, e)
	}
}

func TestDB_BatchFull(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	const size = 3
	// buffered so we never leak goroutines
	ch := make(chan error, size)
	put := func(i int) {
		ch <- db.Batch(func(tx *bolt.Tx) error {
			return tx.Bucket([]byte("widgets")).Put(u64tob(uint64(i)), []byte{})
		})
	}

	db.MaxBatchSize = size
	// high enough to never trigger here
	db.MaxBatchDelay = 1 * time.Hour

	go put(1)
	go put(2)

	// Give the batch a chance to exhibit bugs.
	time.Sleep(10 * time.Millisecond)

	// not triggered yet
	select {
	case <-ch:
		t.Fatalf("batch triggered too early")
	default:
	}

	go put(3)

	// Check all responses to make sure there's no error.
	for i := 0; i < size; i++ {
		if err := <-ch; err != nil {
			t.Fatal(err)
		}
	}

	// Ensure data is correct.
	if err := db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		for i := 1; i <= size; i++ {
			if v := b.Get(u64tob(uint64(i))); v == nil {
				t.Errorf("key not found: %d", i)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestDB_BatchTime(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	const size = 1
	// buffered so we never leak goroutines
	ch := make(chan error, size)
	put := func(i int) {
		ch <- db.Batch(func(tx *bolt.Tx) error {
			return tx.Bucket([]byte("widgets")).Put(u64tob(uint64(i)), []byte{})
		})
	}

	db.MaxBatchSize = 1000
	db.MaxBatchDelay = 0

	go put(1)

	// Batch must trigger by time alone.

	// Check all responses to make sure there's no error.
	for i := 0; i < size; i++ {
		if err := <-ch; err != nil {
			t.Fatal(err)
		}
	}

	// Ensure data is correct.
	if err := db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		for i := 1; i <= size; i++ {
			if v := b.Get(u64tob(uint64(i))); v == nil {
				t.Errorf("key not found: %d", i)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func ExampleDB_Update() {
	// Open the database.
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(db.Path())

	// Execute several commands within a read-write transaction.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			return err
		}
		if err := b.Put([]byte("foo"), []byte("bar")); err != nil {
			return err
		}
		return nil
	}); err != nil {
		log.Fatal(err)
	}

	// Read the value back from a separate read-only transaction.
	if err := db.View(func(tx *bolt.Tx) error {
		value := tx.Bucket([]byte("widgets")).Get([]byte("foo"))
		fmt.Printf("The value of 'foo' is: %s\n", value)
		return nil
	}); err != nil {
		log.Fatal(err)
	}

	// Close database to release the file lock.
	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// The value of 'foo' is: bar
}

func ExampleDB_View() {
	// Open the database.
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(db.Path())

	// Insert data into a bucket.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("people"))
		if err != nil {
			return err
		}
		if err := b.Put([]byte("john"), []byte("doe")); err != nil {
			return err
		}
		if err := b.Put([]byte("susy"), []byte("que")); err != nil {
			return err
		}
		return nil
	}); err != nil {
		log.Fatal(err)
	}

	// Access data from within a read-only transactional block.
	if err := db.View(func(tx *bolt.Tx) error {
		v := tx.Bucket([]byte("people")).Get([]byte("john"))
		fmt.Printf("John's last name is %s.\n", v)
		return nil
	}); err != nil {
		log.Fatal(err)
	}

	// Close database to release the file lock.
	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// John's last name is doe.
}

func ExampleDB_Begin_ReadOnly() {
	// Open the database.
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(db.Path())

	// Create a bucket using a read-write transaction.
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		log.Fatal(err)
	}

	// Create several keys in a transaction.
	tx, err := db.Begin(true)
	if err != nil {
		log.Fatal(err)
	}
	b := tx.Bucket([]byte("widgets"))
	if err := b.Put([]byte("john"), []byte("blue")); err != nil {
		log.Fatal(err)
	}
	if err := b.Put([]byte("abby"), []byte("red")); err != nil {
		log.Fatal(err)
	}
	if err := b.Put([]byte("zephyr"), []byte("purple")); err != nil {
		log.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		log.Fatal(err)
	}

	// Iterate over the values in sorted key order.
	tx, err = db.Begin(false)
	if err != nil {
		log.Fatal(err)
	}
	c := tx.Bucket([]byte("widgets")).Cursor()
	for k, v := c.First(); k != nil; k, v = c.Next() {
		fmt.Printf("%s likes %s\n", k, v)
	}

	if err := tx.Rollback(); err != nil {
		log.Fatal(err)
	}

	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// abby likes red
	// john likes blue
	// zephyr likes purple
}

func BenchmarkDBBatchAutomatic(b *testing.B) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("bench"))
		return err
	}); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for round := 0; round < 1000; round++ {
			wg.Add(1)

			go func(id uint32) {
				defer wg.Done()
				<-start

				h := fnv.New32a()
				buf := make([]byte, 4)
				binary.LittleEndian.PutUint32(buf, id)
				_, _ = h.Write(buf[:])
				k := h.Sum(nil)
				insert := func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("bench"))
					return b.Put(k, []byte("filler"))
				}
				if err := db.Batch(insert); err != nil {
					b.Error(err)
					return
				}
			}(uint32(round))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}

func BenchmarkDBBatchSingle(b *testing.B) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("bench"))
		return err
	}); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for round := 0; round < 1000; round++ {
			wg.Add(1)
			go func(id uint32) {
				defer wg.Done()
				<-start

				h := fnv.New32a()
				buf := make([]byte, 4)
				binary.LittleEndian.PutUint32(buf, id)
				_, _ = h.Write(buf[:])
				k := h.Sum(nil)
				insert := func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("bench"))
					return b.Put(k, []byte("filler"))
				}
				if err := db.Update(insert); err != nil {
					b.Error(err)
					return
				}
			}(uint32(round))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}

func BenchmarkDBBatchManual10x100(b *testing.B) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("bench"))
		return err
	}); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for major := 0; major < 10; major++ {
			wg.Add(1)
			go func(id uint32) {
				defer wg.Done()
				<-start

				insert100 := func(tx *bolt.Tx) error {
					h := fnv.New32a()
					buf := make([]byte, 4)
					for minor := uint32(0); minor < 100; minor++ {
						binary.LittleEndian.PutUint32(buf, uint32(id*100+minor))
						h.Reset()
						_, _ = h.Write(buf[:])
						k := h.Sum(nil)
						b := tx.Bucket([]byte("bench"))
						if err := b.Put(k, []byte("filler")); err != nil {
							return err
						}
					}
					return nil
				}
				if err := db.Update(insert100); err != nil {
					b.Fatal(err)
				}
			}(uint32(major))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}

func validateBatchBench(b *testing.B, db *DB) {
	var rollback = errors.New("sentinel error to cause rollback")
	validate := func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("bench"))
		h := fnv.New32a()
		buf := make([]byte, 4)
		for id := uint32(0); id < 1000; id++ {
			binary.LittleEndian.PutUint32(buf, id)
			h.Reset()
			_, _ = h.Write(buf[:])
			k := h.Sum(nil)
			v := bucket.Get(k)
			if v == nil {
				b.Errorf("not found id=%d key=%x", id, k)
				continue
			}
			if g, e := v, []byte("filler"); !bytes.Equal(g, e) {
				b.Errorf("bad value for id=%d key=%x: %s != %q", id, k, g, e)
			}
			if err := bucket.Delete(k); err != nil {
				return err
			}
		}
		// should be empty now
		c := bucket.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			b.Errorf("unexpected key: %x = %q", k, v)
		}
		return rollback
	}
	if err := db.Update(validate); err != nil && err != rollback {
		b.Error(err)
	}
}

// DB is a test wrapper for bolt.DB.
type DB struct {
	*bolt.DB
}

// MustOpenDB returns a new, open DB at a temporary location.
func MustOpenDB() *DB {
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		panic(err)
	}
	return &DB{db}
}

// Close closes the database and deletes the underlying file.
func (db *DB) Close() error {
	// Log statistics.
	if *statsFlag {
		db.PrintStats()
	}

	// Check database consistency after every test.
	db.MustCheck()

	// Close database and remove file.
	defer os.Remove(db.Path())
	return db.DB.Close()
}

// MustClose closes the database and deletes the underlying file. Panic on error.
func (db *DB) MustClose() {
	if err := db.Close(); err != nil {
		panic(err)
	}
}

// PrintStats prints the database stats
func (db *DB) PrintStats() {
	var stats = db.Stats()
	fmt.Printf("[db] %-20s %-20s %-20s\n",
		fmt.Sprintf("pg(%d/%d)", stats.TxStats.PageCount, stats.TxStats.PageAlloc),
		fmt.Sprintf("cur(%d)", stats.TxStats.CursorCount),
		fmt.Sprintf("node(%d/%d)", stats.TxStats.NodeCount, stats.TxStats.NodeDeref),
	)
	fmt.Printf("     %-20s %-20s %-20s\n",
		fmt.Sprintf("rebal(%d/%v)", stats.TxStats.Rebalance, truncDuration(stats.TxStats.RebalanceTime)),
		fmt.Sprintf("spill(%d/%v)", stats.TxStats.Spill, truncDuration(stats.TxStats.SpillTime)),
		fmt.Sprintf("w(%d/%v)", stats.TxStats.Write, truncDuration(stats.TxStats.WriteTime)),
	)
}

// MustCheck runs a consistency check on the database and panics if any errors are found.
func (db *DB) MustCheck() {
	if err := db.Update(func(tx *bolt.Tx) error {
		// Collect all the errors.
		var errors []error
		for err := range tx.Check() {
			errors = append(errors, err)
			if len(errors) > 10 {
				break
			}
		}

		// If errors occurred, copy the DB and print the errors.
		if len(errors) > 0 {
			var path = tempfile()
			if err := tx.CopyFile(path, 0600); err != nil {
				panic(err)
			}

			// Print errors.
			fmt.Print("\n\n")
			fmt.Printf("consistency check failed (%d errors)\n", len(errors))
			for _, err := range errors {
				fmt.Println(err)
			}
			fmt.Println("")
			fmt.Println("db saved to:")
			fmt.Println(path)
			fmt.Print("\n\n")
			os.Exit(-1)
		}

		return nil
	}); err != nil && err != bolt.ErrDatabaseNotOpen {
		panic(err)
	}
}

// CopyTempFile copies a database to a temporary file.
func (db *DB) CopyTempFile() {
	path := tempfile()
	if err := db.View(func(tx *bolt.Tx) error {
		return tx.CopyFile(path, 0600)
	}); err != nil {
		panic(err)
	}
	fmt.Println("db copied to: ", path)
}

// tempfile returns a temporary file path.
func tempfile() string {
	f, err := ioutil.TempFile("", "bolt-")
	if err != nil {
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
	if err := os.Remove(f.Name()); err != nil {
		panic(err)
	}
	return f.Name()
}

// mustContainKeys checks that a bucket contains a given set of keys.
func mustContainKeys(b *bolt.Bucket, m map[string]string) {
	found := make(map[string]string)
	if err := b.ForEach(func(k, _ []byte) error {
		found[string(k)] = ""
		return nil
	}); err != nil {
		panic(err)
	}

	// Check for keys found in bucket that shouldn't be there.
	var keys []string
	for k, _ := range found {
		if _, ok := m[string(k)]; !ok {
			keys = append(keys, k)
		}
	}
	if len(keys) > 0 {
		sort.Strings(keys)
		panic(fmt.Sprintf("keys found(%d): %s", len(keys), strings.Join(keys, ",")))
	}

	// Check for keys not found in bucket that should be there.
	for k, _ := range m {
		if _, ok := found[string(k)]; !ok {
			keys = append(keys, k)
		}
	}
	if len(keys) > 0 {
		sort.Strings(keys)
		panic(fmt.Sprintf("keys not found(%d): %s", len(keys), strings.Join(keys, ",")))
	}
}

func trunc(b []byte, length int) []byte {
	if length < len(b) {
		return b[:length]
	}
	return b
}

func truncDuration(d time.Duration) string {
	return regexp.MustCompile(`^(\d+)(\.\d+)`).ReplaceAllString(d.String(), "$1")
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

func warn(v ...interface{})              { fmt.Fprintln(os.Stderr, v...) }
func warnf(msg string, v ...interface{}) { fmt.Fprintf(os.Stderr, msg+"\n", v...) }

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

// btou64 converts an 8-byte slice into an uint64.
func btou64(b []byte) uint64 { return binary.BigEndian.Uint64(b) }
