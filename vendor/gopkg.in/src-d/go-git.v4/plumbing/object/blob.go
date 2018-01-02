package object

import (
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

// Blob is used to store arbitrary data - it is generally a file.
type Blob struct {
	// Hash of the blob.
	Hash plumbing.Hash
	// Size of the (uncompressed) blob.
	Size int64

	obj plumbing.EncodedObject
}

// GetBlob gets a blob from an object storer and decodes it.
func GetBlob(s storer.EncodedObjectStorer, h plumbing.Hash) (*Blob, error) {
	o, err := s.EncodedObject(plumbing.BlobObject, h)
	if err != nil {
		return nil, err
	}

	return DecodeBlob(o)
}

// DecodeObject decodes an encoded object into a *Blob.
func DecodeBlob(o plumbing.EncodedObject) (*Blob, error) {
	b := &Blob{}
	if err := b.Decode(o); err != nil {
		return nil, err
	}

	return b, nil
}

// ID returns the object ID of the blob. The returned value will always match
// the current value of Blob.Hash.
//
// ID is present to fulfill the Object interface.
func (b *Blob) ID() plumbing.Hash {
	return b.Hash
}

// Type returns the type of object. It always returns plumbing.BlobObject.
//
// Type is present to fulfill the Object interface.
func (b *Blob) Type() plumbing.ObjectType {
	return plumbing.BlobObject
}

// Decode transforms a plumbing.EncodedObject into a Blob struct.
func (b *Blob) Decode(o plumbing.EncodedObject) error {
	if o.Type() != plumbing.BlobObject {
		return ErrUnsupportedObject
	}

	b.Hash = o.Hash()
	b.Size = o.Size()
	b.obj = o

	return nil
}

// Encode transforms a Blob into a plumbing.EncodedObject.
func (b *Blob) Encode(o plumbing.EncodedObject) error {
	o.SetType(plumbing.BlobObject)

	w, err := o.Writer()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)

	r, err := b.Reader()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(r, &err)

	_, err = io.Copy(w, r)
	return err
}

// Reader returns a reader allow the access to the content of the blob
func (b *Blob) Reader() (io.ReadCloser, error) {
	return b.obj.Reader()
}

// BlobIter provides an iterator for a set of blobs.
type BlobIter struct {
	storer.EncodedObjectIter
	s storer.EncodedObjectStorer
}

// NewBlobIter takes a storer.EncodedObjectStorer and a
// storer.EncodedObjectIter and returns a *BlobIter that iterates over all
// blobs contained in the storer.EncodedObjectIter.
//
// Any non-blob object returned by the storer.EncodedObjectIter is skipped.
func NewBlobIter(s storer.EncodedObjectStorer, iter storer.EncodedObjectIter) *BlobIter {
	return &BlobIter{iter, s}
}

// Next moves the iterator to the next blob and returns a pointer to it. If
// there are no more blobs, it returns io.EOF.
func (iter *BlobIter) Next() (*Blob, error) {
	for {
		obj, err := iter.EncodedObjectIter.Next()
		if err != nil {
			return nil, err
		}

		if obj.Type() != plumbing.BlobObject {
			continue
		}

		return DecodeBlob(obj)
	}
}

// ForEach call the cb function for each blob contained on this iter until
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *BlobIter) ForEach(cb func(*Blob) error) error {
	return iter.EncodedObjectIter.ForEach(func(obj plumbing.EncodedObject) error {
		if obj.Type() != plumbing.BlobObject {
			return nil
		}

		b, err := DecodeBlob(obj)
		if err != nil {
			return err
		}

		return cb(b)
	})
}
