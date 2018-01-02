package plumbing

import (
	"bytes"
	"io"
	"io/ioutil"
)

// MemoryObject on memory Object implementation
type MemoryObject struct {
	t    ObjectType
	h    Hash
	cont []byte
	sz   int64
}

// Hash return the object Hash, the hash is calculated on-the-fly the first
// time is called, the subsequent calls the same Hash is returned even if the
// type or the content has changed. The Hash is only generated if the size of
// the content is exactly the Object.Size
func (o *MemoryObject) Hash() Hash {
	if o.h == ZeroHash && int64(len(o.cont)) == o.sz {
		o.h = ComputeHash(o.t, o.cont)
	}

	return o.h
}

// Type return the ObjectType
func (o *MemoryObject) Type() ObjectType { return o.t }

// SetType sets the ObjectType
func (o *MemoryObject) SetType(t ObjectType) { o.t = t }

// Size return the size of the object
func (o *MemoryObject) Size() int64 { return o.sz }

// SetSize set the object size, a content of the given size should be written
// afterwards
func (o *MemoryObject) SetSize(s int64) { o.sz = s }

// Reader returns a ObjectReader used to read the object's content.
func (o *MemoryObject) Reader() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewBuffer(o.cont)), nil
}

// Writer returns a ObjectWriter used to write the object's content.
func (o *MemoryObject) Writer() (io.WriteCloser, error) {
	return o, nil
}

func (o *MemoryObject) Write(p []byte) (n int, err error) {
	o.cont = append(o.cont, p...)
	o.sz = int64(len(o.cont))

	return len(p), nil
}

// Close releases any resources consumed by the object when it is acting as a
// ObjectWriter.
func (o *MemoryObject) Close() error { return nil }
