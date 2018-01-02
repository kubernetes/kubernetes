package bolt_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"reflect"
	"sort"
	"testing"
	"testing/quick"

	"github.com/boltdb/bolt"
)

// Ensure that a cursor can return a reference to the bucket that created it.
func TestCursor_Bucket(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if cb := b.Cursor().Bucket(); !reflect.DeepEqual(cb, b) {
			t.Fatal("cursor bucket mismatch")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can seek to the appropriate keys.
func TestCursor_Seek(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("foo"), []byte("0001")); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("bar"), []byte("0002")); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("baz"), []byte("0003")); err != nil {
			t.Fatal(err)
		}

		if _, err := b.CreateBucket([]byte("bkt")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()

		// Exact match should go to the key.
		if k, v := c.Seek([]byte("bar")); !bytes.Equal(k, []byte("bar")) {
			t.Fatalf("unexpected key: %v", k)
		} else if !bytes.Equal(v, []byte("0002")) {
			t.Fatalf("unexpected value: %v", v)
		}

		// Inexact match should go to the next key.
		if k, v := c.Seek([]byte("bas")); !bytes.Equal(k, []byte("baz")) {
			t.Fatalf("unexpected key: %v", k)
		} else if !bytes.Equal(v, []byte("0003")) {
			t.Fatalf("unexpected value: %v", v)
		}

		// Low key should go to the first key.
		if k, v := c.Seek([]byte("")); !bytes.Equal(k, []byte("bar")) {
			t.Fatalf("unexpected key: %v", k)
		} else if !bytes.Equal(v, []byte("0002")) {
			t.Fatalf("unexpected value: %v", v)
		}

		// High key should return no key.
		if k, v := c.Seek([]byte("zzz")); k != nil {
			t.Fatalf("expected nil key: %v", k)
		} else if v != nil {
			t.Fatalf("expected nil value: %v", v)
		}

		// Buckets should return their key but no value.
		if k, v := c.Seek([]byte("bkt")); !bytes.Equal(k, []byte("bkt")) {
			t.Fatalf("unexpected key: %v", k)
		} else if v != nil {
			t.Fatalf("expected nil value: %v", v)
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCursor_Delete(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	const count = 1000

	// Insert every other key between 0 and $count.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < count; i += 1 {
			k := make([]byte, 8)
			binary.BigEndian.PutUint64(k, uint64(i))
			if err := b.Put(k, make([]byte, 100)); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := b.CreateBucket([]byte("sub")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		bound := make([]byte, 8)
		binary.BigEndian.PutUint64(bound, uint64(count/2))
		for key, _ := c.First(); bytes.Compare(key, bound) < 0; key, _ = c.Next() {
			if err := c.Delete(); err != nil {
				t.Fatal(err)
			}
		}

		c.Seek([]byte("sub"))
		if err := c.Delete(); err != bolt.ErrIncompatibleValue {
			t.Fatalf("unexpected error: %s", err)
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		stats := tx.Bucket([]byte("widgets")).Stats()
		if stats.KeyN != count/2+1 {
			t.Fatalf("unexpected KeyN: %d", stats.KeyN)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can seek to the appropriate keys when there are a
// large number of keys. This test also checks that seek will always move
// forward to the next key.
//
// Related: https://github.com/boltdb/bolt/pull/187
func TestCursor_Seek_Large(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	var count = 10000

	// Insert every other key between 0 and $count.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}

		for i := 0; i < count; i += 100 {
			for j := i; j < i+100; j += 2 {
				k := make([]byte, 8)
				binary.BigEndian.PutUint64(k, uint64(j))
				if err := b.Put(k, make([]byte, 100)); err != nil {
					t.Fatal(err)
				}
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		for i := 0; i < count; i++ {
			seek := make([]byte, 8)
			binary.BigEndian.PutUint64(seek, uint64(i))

			k, _ := c.Seek(seek)

			// The last seek is beyond the end of the the range so
			// it should return nil.
			if i == count-1 {
				if k != nil {
					t.Fatal("expected nil key")
				}
				continue
			}

			// Otherwise we should seek to the exact key or the next key.
			num := binary.BigEndian.Uint64(k)
			if i%2 == 0 {
				if num != uint64(i) {
					t.Fatalf("unexpected num: %d", num)
				}
			} else {
				if num != uint64(i+1) {
					t.Fatalf("unexpected num: %d", num)
				}
			}
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a cursor can iterate over an empty bucket without error.
func TestCursor_EmptyBucket(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		k, v := c.First()
		if k != nil {
			t.Fatalf("unexpected key: %v", k)
		} else if v != nil {
			t.Fatalf("unexpected value: %v", v)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can reverse iterate over an empty bucket without error.
func TestCursor_EmptyBucketReverse(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if err := db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		k, v := c.Last()
		if k != nil {
			t.Fatalf("unexpected key: %v", k)
		} else if v != nil {
			t.Fatalf("unexpected value: %v", v)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can iterate over a single root with a couple elements.
func TestCursor_Iterate_Leaf(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("baz"), []byte{}); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("foo"), []byte{0}); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("bar"), []byte{1}); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	tx, err := db.Begin(false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()

	c := tx.Bucket([]byte("widgets")).Cursor()

	k, v := c.First()
	if !bytes.Equal(k, []byte("bar")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{1}) {
		t.Fatalf("unexpected value: %v", v)
	}

	k, v = c.Next()
	if !bytes.Equal(k, []byte("baz")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{}) {
		t.Fatalf("unexpected value: %v", v)
	}

	k, v = c.Next()
	if !bytes.Equal(k, []byte("foo")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{0}) {
		t.Fatalf("unexpected value: %v", v)
	}

	k, v = c.Next()
	if k != nil {
		t.Fatalf("expected nil key: %v", k)
	} else if v != nil {
		t.Fatalf("expected nil value: %v", v)
	}

	k, v = c.Next()
	if k != nil {
		t.Fatalf("expected nil key: %v", k)
	} else if v != nil {
		t.Fatalf("expected nil value: %v", v)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can iterate in reverse over a single root with a couple elements.
func TestCursor_LeafRootReverse(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("baz"), []byte{}); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("foo"), []byte{0}); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("bar"), []byte{1}); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	tx, err := db.Begin(false)
	if err != nil {
		t.Fatal(err)
	}
	c := tx.Bucket([]byte("widgets")).Cursor()

	if k, v := c.Last(); !bytes.Equal(k, []byte("foo")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{0}) {
		t.Fatalf("unexpected value: %v", v)
	}

	if k, v := c.Prev(); !bytes.Equal(k, []byte("baz")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{}) {
		t.Fatalf("unexpected value: %v", v)
	}

	if k, v := c.Prev(); !bytes.Equal(k, []byte("bar")) {
		t.Fatalf("unexpected key: %v", k)
	} else if !bytes.Equal(v, []byte{1}) {
		t.Fatalf("unexpected value: %v", v)
	}

	if k, v := c.Prev(); k != nil {
		t.Fatalf("expected nil key: %v", k)
	} else if v != nil {
		t.Fatalf("expected nil value: %v", v)
	}

	if k, v := c.Prev(); k != nil {
		t.Fatalf("expected nil key: %v", k)
	} else if v != nil {
		t.Fatalf("expected nil value: %v", v)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can restart from the beginning.
func TestCursor_Restart(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("bar"), []byte{}); err != nil {
			t.Fatal(err)
		}
		if err := b.Put([]byte("foo"), []byte{}); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	tx, err := db.Begin(false)
	if err != nil {
		t.Fatal(err)
	}
	c := tx.Bucket([]byte("widgets")).Cursor()

	if k, _ := c.First(); !bytes.Equal(k, []byte("bar")) {
		t.Fatalf("unexpected key: %v", k)
	}
	if k, _ := c.Next(); !bytes.Equal(k, []byte("foo")) {
		t.Fatalf("unexpected key: %v", k)
	}

	if k, _ := c.First(); !bytes.Equal(k, []byte("bar")) {
		t.Fatalf("unexpected key: %v", k)
	}
	if k, _ := c.Next(); !bytes.Equal(k, []byte("foo")) {
		t.Fatalf("unexpected key: %v", k)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a cursor can skip over empty pages that have been deleted.
func TestCursor_First_EmptyPages(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	// Create 1000 keys in the "widgets" bucket.
	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}

		for i := 0; i < 1000; i++ {
			if err := b.Put(u64tob(uint64(i)), []byte{}); err != nil {
				t.Fatal(err)
			}
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Delete half the keys and then try to iterate.
	if err := db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		for i := 0; i < 600; i++ {
			if err := b.Delete(u64tob(uint64(i))); err != nil {
				t.Fatal(err)
			}
		}

		c := b.Cursor()
		var n int
		for k, _ := c.First(); k != nil; k, _ = c.Next() {
			n++
		}
		if n != 400 {
			t.Fatalf("unexpected key count: %d", n)
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx can iterate over all elements in a bucket.
func TestCursor_QuickCheck(t *testing.T) {
	f := func(items testdata) bool {
		db := MustOpenDB()
		defer db.MustClose()

		// Bulk insert all values.
		tx, err := db.Begin(true)
		if err != nil {
			t.Fatal(err)
		}
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range items {
			if err := b.Put(item.Key, item.Value); err != nil {
				t.Fatal(err)
			}
		}
		if err := tx.Commit(); err != nil {
			t.Fatal(err)
		}

		// Sort test data.
		sort.Sort(items)

		// Iterate over all items and check consistency.
		var index = 0
		tx, err = db.Begin(false)
		if err != nil {
			t.Fatal(err)
		}

		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.First(); k != nil && index < len(items); k, v = c.Next() {
			if !bytes.Equal(k, items[index].Key) {
				t.Fatalf("unexpected key: %v", k)
			} else if !bytes.Equal(v, items[index].Value) {
				t.Fatalf("unexpected value: %v", v)
			}
			index++
		}
		if len(items) != index {
			t.Fatalf("unexpected item count: %v, expected %v", len(items), index)
		}

		if err := tx.Rollback(); err != nil {
			t.Fatal(err)
		}

		return true
	}
	if err := quick.Check(f, qconfig()); err != nil {
		t.Error(err)
	}
}

// Ensure that a transaction can iterate over all elements in a bucket in reverse.
func TestCursor_QuickCheck_Reverse(t *testing.T) {
	f := func(items testdata) bool {
		db := MustOpenDB()
		defer db.MustClose()

		// Bulk insert all values.
		tx, err := db.Begin(true)
		if err != nil {
			t.Fatal(err)
		}
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range items {
			if err := b.Put(item.Key, item.Value); err != nil {
				t.Fatal(err)
			}
		}
		if err := tx.Commit(); err != nil {
			t.Fatal(err)
		}

		// Sort test data.
		sort.Sort(revtestdata(items))

		// Iterate over all items and check consistency.
		var index = 0
		tx, err = db.Begin(false)
		if err != nil {
			t.Fatal(err)
		}
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.Last(); k != nil && index < len(items); k, v = c.Prev() {
			if !bytes.Equal(k, items[index].Key) {
				t.Fatalf("unexpected key: %v", k)
			} else if !bytes.Equal(v, items[index].Value) {
				t.Fatalf("unexpected value: %v", v)
			}
			index++
		}
		if len(items) != index {
			t.Fatalf("unexpected item count: %v, expected %v", len(items), index)
		}

		if err := tx.Rollback(); err != nil {
			t.Fatal(err)
		}

		return true
	}
	if err := quick.Check(f, qconfig()); err != nil {
		t.Error(err)
	}
}

// Ensure that a Tx cursor can iterate over subbuckets.
func TestCursor_QuickCheck_BucketsOnly(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("foo")); err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("bar")); err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("baz")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		var names []string
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			names = append(names, string(k))
			if v != nil {
				t.Fatalf("unexpected value: %v", v)
			}
		}
		if !reflect.DeepEqual(names, []string{"bar", "baz", "foo"}) {
			t.Fatalf("unexpected names: %+v", names)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// Ensure that a Tx cursor can reverse iterate over subbuckets.
func TestCursor_QuickCheck_BucketsOnly_Reverse(t *testing.T) {
	db := MustOpenDB()
	defer db.MustClose()

	if err := db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("foo")); err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("bar")); err != nil {
			t.Fatal(err)
		}
		if _, err := b.CreateBucket([]byte("baz")); err != nil {
			t.Fatal(err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if err := db.View(func(tx *bolt.Tx) error {
		var names []string
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.Last(); k != nil; k, v = c.Prev() {
			names = append(names, string(k))
			if v != nil {
				t.Fatalf("unexpected value: %v", v)
			}
		}
		if !reflect.DeepEqual(names, []string{"foo", "baz", "bar"}) {
			t.Fatalf("unexpected names: %+v", names)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func ExampleCursor() {
	// Open the database.
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(db.Path())

	// Start a read-write transaction.
	if err := db.Update(func(tx *bolt.Tx) error {
		// Create a new bucket.
		b, err := tx.CreateBucket([]byte("animals"))
		if err != nil {
			return err
		}

		// Insert data into a bucket.
		if err := b.Put([]byte("dog"), []byte("fun")); err != nil {
			log.Fatal(err)
		}
		if err := b.Put([]byte("cat"), []byte("lame")); err != nil {
			log.Fatal(err)
		}
		if err := b.Put([]byte("liger"), []byte("awesome")); err != nil {
			log.Fatal(err)
		}

		// Create a cursor for iteration.
		c := b.Cursor()

		// Iterate over items in sorted key order. This starts from the
		// first key/value pair and updates the k/v variables to the
		// next key/value on each iteration.
		//
		// The loop finishes at the end of the cursor when a nil key is returned.
		for k, v := c.First(); k != nil; k, v = c.Next() {
			fmt.Printf("A %s is %s.\n", k, v)
		}

		return nil
	}); err != nil {
		log.Fatal(err)
	}

	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// A cat is lame.
	// A dog is fun.
	// A liger is awesome.
}

func ExampleCursor_reverse() {
	// Open the database.
	db, err := bolt.Open(tempfile(), 0666, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(db.Path())

	// Start a read-write transaction.
	if err := db.Update(func(tx *bolt.Tx) error {
		// Create a new bucket.
		b, err := tx.CreateBucket([]byte("animals"))
		if err != nil {
			return err
		}

		// Insert data into a bucket.
		if err := b.Put([]byte("dog"), []byte("fun")); err != nil {
			log.Fatal(err)
		}
		if err := b.Put([]byte("cat"), []byte("lame")); err != nil {
			log.Fatal(err)
		}
		if err := b.Put([]byte("liger"), []byte("awesome")); err != nil {
			log.Fatal(err)
		}

		// Create a cursor for iteration.
		c := b.Cursor()

		// Iterate over items in reverse sorted key order. This starts
		// from the last key/value pair and updates the k/v variables to
		// the previous key/value on each iteration.
		//
		// The loop finishes at the beginning of the cursor when a nil key
		// is returned.
		for k, v := c.Last(); k != nil; k, v = c.Prev() {
			fmt.Printf("A %s is %s.\n", k, v)
		}

		return nil
	}); err != nil {
		log.Fatal(err)
	}

	// Close the database to release the file lock.
	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// A liger is awesome.
	// A dog is fun.
	// A cat is lame.
}
