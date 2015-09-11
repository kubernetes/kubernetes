package bolt_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"testing"
	"testing/quick"

	"github.com/boltdb/bolt"
)

// Ensure that a cursor can return a reference to the bucket that created it.
func TestCursor_Bucket(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		b, _ := tx.CreateBucket([]byte("widgets"))
		c := b.Cursor()
		equals(t, b, c.Bucket())
		return nil
	})
}

// Ensure that a Tx cursor can seek to the appropriate keys.
func TestCursor_Seek(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		ok(t, err)
		ok(t, b.Put([]byte("foo"), []byte("0001")))
		ok(t, b.Put([]byte("bar"), []byte("0002")))
		ok(t, b.Put([]byte("baz"), []byte("0003")))
		_, err = b.CreateBucket([]byte("bkt"))
		ok(t, err)
		return nil
	})
	db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()

		// Exact match should go to the key.
		k, v := c.Seek([]byte("bar"))
		equals(t, []byte("bar"), k)
		equals(t, []byte("0002"), v)

		// Inexact match should go to the next key.
		k, v = c.Seek([]byte("bas"))
		equals(t, []byte("baz"), k)
		equals(t, []byte("0003"), v)

		// Low key should go to the first key.
		k, v = c.Seek([]byte(""))
		equals(t, []byte("bar"), k)
		equals(t, []byte("0002"), v)

		// High key should return no key.
		k, v = c.Seek([]byte("zzz"))
		assert(t, k == nil, "")
		assert(t, v == nil, "")

		// Buckets should return their key but no value.
		k, v = c.Seek([]byte("bkt"))
		equals(t, []byte("bkt"), k)
		assert(t, v == nil, "")

		return nil
	})
}

func TestCursor_Delete(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	var count = 1000

	// Insert every other key between 0 and $count.
	db.Update(func(tx *bolt.Tx) error {
		b, _ := tx.CreateBucket([]byte("widgets"))
		for i := 0; i < count; i += 1 {
			k := make([]byte, 8)
			binary.BigEndian.PutUint64(k, uint64(i))
			b.Put(k, make([]byte, 100))
		}
		b.CreateBucket([]byte("sub"))
		return nil
	})

	db.Update(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		bound := make([]byte, 8)
		binary.BigEndian.PutUint64(bound, uint64(count/2))
		for key, _ := c.First(); bytes.Compare(key, bound) < 0; key, _ = c.Next() {
			if err := c.Delete(); err != nil {
				return err
			}
		}
		c.Seek([]byte("sub"))
		err := c.Delete()
		equals(t, err, bolt.ErrIncompatibleValue)
		return nil
	})

	db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		equals(t, b.Stats().KeyN, count/2+1)
		return nil
	})
}

// Ensure that a Tx cursor can seek to the appropriate keys when there are a
// large number of keys. This test also checks that seek will always move
// forward to the next key.
//
// Related: https://github.com/boltdb/bolt/pull/187
func TestCursor_Seek_Large(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	var count = 10000

	// Insert every other key between 0 and $count.
	db.Update(func(tx *bolt.Tx) error {
		b, _ := tx.CreateBucket([]byte("widgets"))
		for i := 0; i < count; i += 100 {
			for j := i; j < i+100; j += 2 {
				k := make([]byte, 8)
				binary.BigEndian.PutUint64(k, uint64(j))
				b.Put(k, make([]byte, 100))
			}
		}
		return nil
	})

	db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		for i := 0; i < count; i++ {
			seek := make([]byte, 8)
			binary.BigEndian.PutUint64(seek, uint64(i))

			k, _ := c.Seek(seek)

			// The last seek is beyond the end of the the range so
			// it should return nil.
			if i == count-1 {
				assert(t, k == nil, "")
				continue
			}

			// Otherwise we should seek to the exact key or the next key.
			num := binary.BigEndian.Uint64(k)
			if i%2 == 0 {
				equals(t, uint64(i), num)
			} else {
				equals(t, uint64(i+1), num)
			}
		}

		return nil
	})
}

// Ensure that a cursor can iterate over an empty bucket without error.
func TestCursor_EmptyBucket(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	})
	db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		k, v := c.First()
		assert(t, k == nil, "")
		assert(t, v == nil, "")
		return nil
	})
}

// Ensure that a Tx cursor can reverse iterate over an empty bucket without error.
func TestCursor_EmptyBucketReverse(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	})
	db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("widgets")).Cursor()
		k, v := c.Last()
		assert(t, k == nil, "")
		assert(t, v == nil, "")
		return nil
	})
}

// Ensure that a Tx cursor can iterate over a single root with a couple elements.
func TestCursor_Iterate_Leaf(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("baz"), []byte{})
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte{0})
		tx.Bucket([]byte("widgets")).Put([]byte("bar"), []byte{1})
		return nil
	})
	tx, _ := db.Begin(false)
	c := tx.Bucket([]byte("widgets")).Cursor()

	k, v := c.First()
	equals(t, string(k), "bar")
	equals(t, v, []byte{1})

	k, v = c.Next()
	equals(t, string(k), "baz")
	equals(t, v, []byte{})

	k, v = c.Next()
	equals(t, string(k), "foo")
	equals(t, v, []byte{0})

	k, v = c.Next()
	assert(t, k == nil, "")
	assert(t, v == nil, "")

	k, v = c.Next()
	assert(t, k == nil, "")
	assert(t, v == nil, "")

	tx.Rollback()
}

// Ensure that a Tx cursor can iterate in reverse over a single root with a couple elements.
func TestCursor_LeafRootReverse(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("baz"), []byte{})
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte{0})
		tx.Bucket([]byte("widgets")).Put([]byte("bar"), []byte{1})
		return nil
	})
	tx, _ := db.Begin(false)
	c := tx.Bucket([]byte("widgets")).Cursor()

	k, v := c.Last()
	equals(t, string(k), "foo")
	equals(t, v, []byte{0})

	k, v = c.Prev()
	equals(t, string(k), "baz")
	equals(t, v, []byte{})

	k, v = c.Prev()
	equals(t, string(k), "bar")
	equals(t, v, []byte{1})

	k, v = c.Prev()
	assert(t, k == nil, "")
	assert(t, v == nil, "")

	k, v = c.Prev()
	assert(t, k == nil, "")
	assert(t, v == nil, "")

	tx.Rollback()
}

// Ensure that a Tx cursor can restart from the beginning.
func TestCursor_Restart(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("bar"), []byte{})
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte{})
		return nil
	})

	tx, _ := db.Begin(false)
	c := tx.Bucket([]byte("widgets")).Cursor()

	k, _ := c.First()
	equals(t, string(k), "bar")

	k, _ = c.Next()
	equals(t, string(k), "foo")

	k, _ = c.First()
	equals(t, string(k), "bar")

	k, _ = c.Next()
	equals(t, string(k), "foo")

	tx.Rollback()
}

// Ensure that a Tx can iterate over all elements in a bucket.
func TestCursor_QuickCheck(t *testing.T) {
	f := func(items testdata) bool {
		db := NewTestDB()
		defer db.Close()

		// Bulk insert all values.
		tx, _ := db.Begin(true)
		tx.CreateBucket([]byte("widgets"))
		b := tx.Bucket([]byte("widgets"))
		for _, item := range items {
			ok(t, b.Put(item.Key, item.Value))
		}
		ok(t, tx.Commit())

		// Sort test data.
		sort.Sort(items)

		// Iterate over all items and check consistency.
		var index = 0
		tx, _ = db.Begin(false)
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.First(); k != nil && index < len(items); k, v = c.Next() {
			equals(t, k, items[index].Key)
			equals(t, v, items[index].Value)
			index++
		}
		equals(t, len(items), index)
		tx.Rollback()

		return true
	}
	if err := quick.Check(f, qconfig()); err != nil {
		t.Error(err)
	}
}

// Ensure that a transaction can iterate over all elements in a bucket in reverse.
func TestCursor_QuickCheck_Reverse(t *testing.T) {
	f := func(items testdata) bool {
		db := NewTestDB()
		defer db.Close()

		// Bulk insert all values.
		tx, _ := db.Begin(true)
		tx.CreateBucket([]byte("widgets"))
		b := tx.Bucket([]byte("widgets"))
		for _, item := range items {
			ok(t, b.Put(item.Key, item.Value))
		}
		ok(t, tx.Commit())

		// Sort test data.
		sort.Sort(revtestdata(items))

		// Iterate over all items and check consistency.
		var index = 0
		tx, _ = db.Begin(false)
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.Last(); k != nil && index < len(items); k, v = c.Prev() {
			equals(t, k, items[index].Key)
			equals(t, v, items[index].Value)
			index++
		}
		equals(t, len(items), index)
		tx.Rollback()

		return true
	}
	if err := quick.Check(f, qconfig()); err != nil {
		t.Error(err)
	}
}

// Ensure that a Tx cursor can iterate over subbuckets.
func TestCursor_QuickCheck_BucketsOnly(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("foo"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("bar"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("baz"))
		ok(t, err)
		return nil
	})
	db.View(func(tx *bolt.Tx) error {
		var names []string
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			names = append(names, string(k))
			assert(t, v == nil, "")
		}
		equals(t, names, []string{"bar", "baz", "foo"})
		return nil
	})
}

// Ensure that a Tx cursor can reverse iterate over subbuckets.
func TestCursor_QuickCheck_BucketsOnly_Reverse(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("foo"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("bar"))
		ok(t, err)
		_, err = b.CreateBucket([]byte("baz"))
		ok(t, err)
		return nil
	})
	db.View(func(tx *bolt.Tx) error {
		var names []string
		c := tx.Bucket([]byte("widgets")).Cursor()
		for k, v := c.Last(); k != nil; k, v = c.Prev() {
			names = append(names, string(k))
			assert(t, v == nil, "")
		}
		equals(t, names, []string{"foo", "baz", "bar"})
		return nil
	})
}

func ExampleCursor() {
	// Open the database.
	db, _ := bolt.Open(tempfile(), 0666, nil)
	defer os.Remove(db.Path())
	defer db.Close()

	// Start a read-write transaction.
	db.Update(func(tx *bolt.Tx) error {
		// Create a new bucket.
		tx.CreateBucket([]byte("animals"))

		// Insert data into a bucket.
		b := tx.Bucket([]byte("animals"))
		b.Put([]byte("dog"), []byte("fun"))
		b.Put([]byte("cat"), []byte("lame"))
		b.Put([]byte("liger"), []byte("awesome"))

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
	})

	// Output:
	// A cat is lame.
	// A dog is fun.
	// A liger is awesome.
}

func ExampleCursor_reverse() {
	// Open the database.
	db, _ := bolt.Open(tempfile(), 0666, nil)
	defer os.Remove(db.Path())
	defer db.Close()

	// Start a read-write transaction.
	db.Update(func(tx *bolt.Tx) error {
		// Create a new bucket.
		tx.CreateBucket([]byte("animals"))

		// Insert data into a bucket.
		b := tx.Bucket([]byte("animals"))
		b.Put([]byte("dog"), []byte("fun"))
		b.Put([]byte("cat"), []byte("lame"))
		b.Put([]byte("liger"), []byte("awesome"))

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
	})

	// Output:
	// A liger is awesome.
	// A dog is fun.
	// A cat is lame.
}
