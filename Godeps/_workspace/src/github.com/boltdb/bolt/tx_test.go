package bolt_test

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/boltdb/bolt"
)

// Ensure that committing a closed transaction returns an error.
func TestTx_Commit_Closed(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	tx, _ := db.Begin(true)
	tx.CreateBucket([]byte("foo"))
	ok(t, tx.Commit())
	equals(t, tx.Commit(), bolt.ErrTxClosed)
}

// Ensure that rolling back a closed transaction returns an error.
func TestTx_Rollback_Closed(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	tx, _ := db.Begin(true)
	ok(t, tx.Rollback())
	equals(t, tx.Rollback(), bolt.ErrTxClosed)
}

// Ensure that committing a read-only transaction returns an error.
func TestTx_Commit_ReadOnly(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	tx, _ := db.Begin(false)
	equals(t, tx.Commit(), bolt.ErrTxNotWritable)
}

// Ensure that a transaction can retrieve a cursor on the root bucket.
func TestTx_Cursor(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.CreateBucket([]byte("woojits"))
		c := tx.Cursor()

		k, v := c.First()
		equals(t, "widgets", string(k))
		assert(t, v == nil, "")

		k, v = c.Next()
		equals(t, "woojits", string(k))
		assert(t, v == nil, "")

		k, v = c.Next()
		assert(t, k == nil, "")
		assert(t, v == nil, "")

		return nil
	})
}

// Ensure that creating a bucket with a read-only transaction returns an error.
func TestTx_CreateBucket_ReadOnly(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.View(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("foo"))
		assert(t, b == nil, "")
		equals(t, bolt.ErrTxNotWritable, err)
		return nil
	})
}

// Ensure that creating a bucket on a closed transaction returns an error.
func TestTx_CreateBucket_Closed(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	tx, _ := db.Begin(true)
	tx.Commit()
	b, err := tx.CreateBucket([]byte("foo"))
	assert(t, b == nil, "")
	equals(t, bolt.ErrTxClosed, err)
}

// Ensure that a Tx can retrieve a bucket.
func TestTx_Bucket(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		b := tx.Bucket([]byte("widgets"))
		assert(t, b != nil, "")
		return nil
	})
}

// Ensure that a Tx retrieving a non-existent key returns nil.
func TestTx_Get_Missing(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		value := tx.Bucket([]byte("widgets")).Get([]byte("no_such_key"))
		assert(t, value == nil, "")
		return nil
	})
}

// Ensure that a bucket can be created and retrieved.
func TestTx_CreateBucket(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	// Create a bucket.
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		assert(t, b != nil, "")
		ok(t, err)
		return nil
	})

	// Read the bucket through a separate transaction.
	db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		assert(t, b != nil, "")
		return nil
	})
}

// Ensure that a bucket can be created if it doesn't already exist.
func TestTx_CreateBucketIfNotExists(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("widgets"))
		assert(t, b != nil, "")
		ok(t, err)

		b, err = tx.CreateBucketIfNotExists([]byte("widgets"))
		assert(t, b != nil, "")
		ok(t, err)

		b, err = tx.CreateBucketIfNotExists([]byte{})
		assert(t, b == nil, "")
		equals(t, bolt.ErrBucketNameRequired, err)

		b, err = tx.CreateBucketIfNotExists(nil)
		assert(t, b == nil, "")
		equals(t, bolt.ErrBucketNameRequired, err)
		return nil
	})

	// Read the bucket through a separate transaction.
	db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("widgets"))
		assert(t, b != nil, "")
		return nil
	})
}

// Ensure that a bucket cannot be created twice.
func TestTx_CreateBucket_Exists(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	// Create a bucket.
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		assert(t, b != nil, "")
		ok(t, err)
		return nil
	})

	// Create the same bucket again.
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte("widgets"))
		assert(t, b == nil, "")
		equals(t, bolt.ErrBucketExists, err)
		return nil
	})
}

// Ensure that a bucket is created with a non-blank name.
func TestTx_CreateBucket_NameRequired(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket(nil)
		assert(t, b == nil, "")
		equals(t, bolt.ErrBucketNameRequired, err)
		return nil
	})
}

// Ensure that a bucket can be deleted.
func TestTx_DeleteBucket(t *testing.T) {
	db := NewTestDB()
	defer db.Close()

	// Create a bucket and add a value.
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		return nil
	})

	// Delete the bucket and make sure we can't get the value.
	db.Update(func(tx *bolt.Tx) error {
		ok(t, tx.DeleteBucket([]byte("widgets")))
		assert(t, tx.Bucket([]byte("widgets")) == nil, "")
		return nil
	})

	db.Update(func(tx *bolt.Tx) error {
		// Create the bucket again and make sure there's not a phantom value.
		b, err := tx.CreateBucket([]byte("widgets"))
		assert(t, b != nil, "")
		ok(t, err)
		assert(t, tx.Bucket([]byte("widgets")).Get([]byte("foo")) == nil, "")
		return nil
	})
}

// Ensure that deleting a bucket on a closed transaction returns an error.
func TestTx_DeleteBucket_Closed(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	tx, _ := db.Begin(true)
	tx.Commit()
	equals(t, tx.DeleteBucket([]byte("foo")), bolt.ErrTxClosed)
}

// Ensure that deleting a bucket with a read-only transaction returns an error.
func TestTx_DeleteBucket_ReadOnly(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.View(func(tx *bolt.Tx) error {
		equals(t, tx.DeleteBucket([]byte("foo")), bolt.ErrTxNotWritable)
		return nil
	})
}

// Ensure that nothing happens when deleting a bucket that doesn't exist.
func TestTx_DeleteBucket_NotFound(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		equals(t, bolt.ErrBucketNotFound, tx.DeleteBucket([]byte("widgets")))
		return nil
	})
}

// Ensure that no error is returned when a tx.ForEach function does not return
// an error.
func TestTx_ForEach_NoError(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))

		equals(t, nil, tx.ForEach(func(name []byte, b *bolt.Bucket) error {
			return nil
		}))
		return nil
	})
}

// Ensure that an error is returned when a tx.ForEach function returns an error.
func TestTx_ForEach_WithError(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))

		err := errors.New("foo")
		equals(t, err, tx.ForEach(func(name []byte, b *bolt.Bucket) error {
			return err
		}))
		return nil
	})
}

// Ensure that Tx commit handlers are called after a transaction successfully commits.
func TestTx_OnCommit(t *testing.T) {
	var x int
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.OnCommit(func() { x += 1 })
		tx.OnCommit(func() { x += 2 })
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	})
	equals(t, 3, x)
}

// Ensure that Tx commit handlers are NOT called after a transaction rolls back.
func TestTx_OnCommit_Rollback(t *testing.T) {
	var x int
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.OnCommit(func() { x += 1 })
		tx.OnCommit(func() { x += 2 })
		tx.CreateBucket([]byte("widgets"))
		return errors.New("rollback this commit")
	})
	equals(t, 0, x)
}

// Ensure that the database can be copied to a file path.
func TestTx_CopyFile(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	var dest = tempfile()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		tx.Bucket([]byte("widgets")).Put([]byte("baz"), []byte("bat"))
		return nil
	})

	ok(t, db.View(func(tx *bolt.Tx) error { return tx.CopyFile(dest, 0600) }))

	db2, err := bolt.Open(dest, 0600, nil)
	ok(t, err)
	defer db2.Close()

	db2.View(func(tx *bolt.Tx) error {
		equals(t, []byte("bar"), tx.Bucket([]byte("widgets")).Get([]byte("foo")))
		equals(t, []byte("bat"), tx.Bucket([]byte("widgets")).Get([]byte("baz")))
		return nil
	})
}

type failWriterError struct{}

func (failWriterError) Error() string {
	return "error injected for tests"
}

type failWriter struct {
	// fail after this many bytes
	After int
}

func (f *failWriter) Write(p []byte) (n int, err error) {
	n = len(p)
	if n > f.After {
		n = f.After
		err = failWriterError{}
	}
	f.After -= n
	return n, err
}

// Ensure that Copy handles write errors right.
func TestTx_CopyFile_Error_Meta(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		tx.Bucket([]byte("widgets")).Put([]byte("baz"), []byte("bat"))
		return nil
	})

	err := db.View(func(tx *bolt.Tx) error { return tx.Copy(&failWriter{}) })
	equals(t, err.Error(), "meta copy: error injected for tests")
}

// Ensure that Copy handles write errors right.
func TestTx_CopyFile_Error_Normal(t *testing.T) {
	db := NewTestDB()
	defer db.Close()
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		tx.Bucket([]byte("widgets")).Put([]byte("baz"), []byte("bat"))
		return nil
	})

	err := db.View(func(tx *bolt.Tx) error { return tx.Copy(&failWriter{3 * db.Info().PageSize}) })
	equals(t, err.Error(), "error injected for tests")
}

func ExampleTx_Rollback() {
	// Open the database.
	db, _ := bolt.Open(tempfile(), 0666, nil)
	defer os.Remove(db.Path())
	defer db.Close()

	// Create a bucket.
	db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte("widgets"))
		return err
	})

	// Set a value for a key.
	db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
	})

	// Update the key but rollback the transaction so it never saves.
	tx, _ := db.Begin(true)
	b := tx.Bucket([]byte("widgets"))
	b.Put([]byte("foo"), []byte("baz"))
	tx.Rollback()

	// Ensure that our original value is still set.
	db.View(func(tx *bolt.Tx) error {
		value := tx.Bucket([]byte("widgets")).Get([]byte("foo"))
		fmt.Printf("The value for 'foo' is still: %s\n", value)
		return nil
	})

	// Output:
	// The value for 'foo' is still: bar
}

func ExampleTx_CopyFile() {
	// Open the database.
	db, _ := bolt.Open(tempfile(), 0666, nil)
	defer os.Remove(db.Path())
	defer db.Close()

	// Create a bucket and a key.
	db.Update(func(tx *bolt.Tx) error {
		tx.CreateBucket([]byte("widgets"))
		tx.Bucket([]byte("widgets")).Put([]byte("foo"), []byte("bar"))
		return nil
	})

	// Copy the database to another file.
	toFile := tempfile()
	db.View(func(tx *bolt.Tx) error { return tx.CopyFile(toFile, 0666) })
	defer os.Remove(toFile)

	// Open the cloned database.
	db2, _ := bolt.Open(toFile, 0666, nil)
	defer db2.Close()

	// Ensure that the key exists in the copy.
	db2.View(func(tx *bolt.Tx) error {
		value := tx.Bucket([]byte("widgets")).Get([]byte("foo"))
		fmt.Printf("The value for 'foo' in the clone is: %s\n", value)
		return nil
	})

	// Output:
	// The value for 'foo' in the clone is: bar
}
