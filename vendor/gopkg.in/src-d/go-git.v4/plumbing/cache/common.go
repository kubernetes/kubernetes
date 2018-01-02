package cache

import "gopkg.in/src-d/go-git.v4/plumbing"

const (
	Byte FileSize = 1 << (iota * 10)
	KiByte
	MiByte
	GiByte
)

type FileSize int64

// Object is an interface to a object cache.
type Object interface {
	// Put puts the given object into the cache. Whether this object will
	// actually be put into the cache or not is implementation specific.
	Put(o plumbing.EncodedObject)
	// Get gets an object from the cache given its hash. The second return value
	// is true if the object was returned, and false otherwise.
	Get(k plumbing.Hash) (plumbing.EncodedObject, bool)
	// Clear clears every object from the cache.
	Clear()
}
