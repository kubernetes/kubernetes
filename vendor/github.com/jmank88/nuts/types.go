package nuts

import "github.com/boltdb/bolt"

var _ Buckets = &bolt.Bucket{}
var _ Buckets = &bolt.Tx{}

// Buckets is a collection of methods for managing bolt.Buckets which is satisfied
// by *bolt.Tx and *bolt.Bucket.
type Buckets interface {
	Bucket([]byte) *bolt.Bucket
	CreateBucket([]byte) (*bolt.Bucket, error)
	CreateBucketIfNotExists([]byte) (*bolt.Bucket, error)
	DeleteBucket([]byte) error
	Cursor() *bolt.Cursor
}
