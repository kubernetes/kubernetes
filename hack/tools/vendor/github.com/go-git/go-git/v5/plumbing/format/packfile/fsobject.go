package packfile

import (
	"io"

	billy "github.com/go-git/go-billy/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/cache"
	"github.com/go-git/go-git/v5/plumbing/format/idxfile"
)

// FSObject is an object from the packfile on the filesystem.
type FSObject struct {
	hash   plumbing.Hash
	h      *ObjectHeader
	offset int64
	size   int64
	typ    plumbing.ObjectType
	index  idxfile.Index
	fs     billy.Filesystem
	path   string
	cache  cache.Object
}

// NewFSObject creates a new filesystem object.
func NewFSObject(
	hash plumbing.Hash,
	finalType plumbing.ObjectType,
	offset int64,
	contentSize int64,
	index idxfile.Index,
	fs billy.Filesystem,
	path string,
	cache cache.Object,
) *FSObject {
	return &FSObject{
		hash:   hash,
		offset: offset,
		size:   contentSize,
		typ:    finalType,
		index:  index,
		fs:     fs,
		path:   path,
		cache:  cache,
	}
}

// Reader implements the plumbing.EncodedObject interface.
func (o *FSObject) Reader() (io.ReadCloser, error) {
	obj, ok := o.cache.Get(o.hash)
	if ok && obj != o {
		reader, err := obj.Reader()
		if err != nil {
			return nil, err
		}

		return reader, nil
	}

	f, err := o.fs.Open(o.path)
	if err != nil {
		return nil, err
	}

	p := NewPackfileWithCache(o.index, nil, f, o.cache)
	r, err := p.getObjectContent(o.offset)
	if err != nil {
		_ = f.Close()
		return nil, err
	}

	if err := f.Close(); err != nil {
		return nil, err
	}

	return r, nil
}

// SetSize implements the plumbing.EncodedObject interface. This method
// is a noop.
func (o *FSObject) SetSize(int64) {}

// SetType implements the plumbing.EncodedObject interface. This method is
// a noop.
func (o *FSObject) SetType(plumbing.ObjectType) {}

// Hash implements the plumbing.EncodedObject interface.
func (o *FSObject) Hash() plumbing.Hash { return o.hash }

// Size implements the plumbing.EncodedObject interface.
func (o *FSObject) Size() int64 { return o.size }

// Type implements the plumbing.EncodedObject interface.
func (o *FSObject) Type() plumbing.ObjectType {
	return o.typ
}

// Writer implements the plumbing.EncodedObject interface. This method always
// returns a nil writer.
func (o *FSObject) Writer() (io.WriteCloser, error) {
	return nil, nil
}

type objectReader struct {
	io.ReadCloser
	f billy.File
}

func (r *objectReader) Close() error {
	if err := r.ReadCloser.Close(); err != nil {
		_ = r.f.Close()
		return err
	}

	return r.f.Close()
}
