package packfile

import (
	"bytes"
	"io"
	"os"

	billy "github.com/go-git/go-billy/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/cache"
	"github.com/go-git/go-git/v5/plumbing/format/idxfile"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

var (
	// ErrInvalidObject is returned by Decode when an invalid object is
	// found in the packfile.
	ErrInvalidObject = NewError("invalid git object")
	// ErrZLib is returned by Decode when there was an error unzipping
	// the packfile contents.
	ErrZLib = NewError("zlib reading error")
)

// When reading small objects from packfile it is beneficial to do so at
// once to exploit the buffered I/O. In many cases the objects are so small
// that they were already loaded to memory when the object header was
// loaded from the packfile. Wrapping in FSObject would cause this buffered
// data to be thrown away and then re-read later, with the additional
// seeking causing reloads from disk. Objects smaller than this threshold
// are now always read into memory and stored in cache instead of being
// wrapped in FSObject.
const smallObjectThreshold = 16 * 1024

// Packfile allows retrieving information from inside a packfile.
type Packfile struct {
	idxfile.Index
	fs             billy.Filesystem
	file           billy.File
	s              *Scanner
	deltaBaseCache cache.Object
	offsetToType   map[int64]plumbing.ObjectType
}

// NewPackfileWithCache creates a new Packfile with the given object cache.
// If the filesystem is provided, the packfile will return FSObjects, otherwise
// it will return MemoryObjects.
func NewPackfileWithCache(
	index idxfile.Index,
	fs billy.Filesystem,
	file billy.File,
	cache cache.Object,
) *Packfile {
	s := NewScanner(file)
	return &Packfile{
		index,
		fs,
		file,
		s,
		cache,
		make(map[int64]plumbing.ObjectType),
	}
}

// NewPackfile returns a packfile representation for the given packfile file
// and packfile idx.
// If the filesystem is provided, the packfile will return FSObjects, otherwise
// it will return MemoryObjects.
func NewPackfile(index idxfile.Index, fs billy.Filesystem, file billy.File) *Packfile {
	return NewPackfileWithCache(index, fs, file, cache.NewObjectLRUDefault())
}

// Get retrieves the encoded object in the packfile with the given hash.
func (p *Packfile) Get(h plumbing.Hash) (plumbing.EncodedObject, error) {
	offset, err := p.FindOffset(h)
	if err != nil {
		return nil, err
	}

	return p.objectAtOffset(offset, h)
}

// GetByOffset retrieves the encoded object from the packfile at the given
// offset.
func (p *Packfile) GetByOffset(o int64) (plumbing.EncodedObject, error) {
	hash, err := p.FindHash(o)
	if err != nil {
		return nil, err
	}

	return p.objectAtOffset(o, hash)
}

// GetSizeByOffset retrieves the size of the encoded object from the
// packfile with the given offset.
func (p *Packfile) GetSizeByOffset(o int64) (size int64, err error) {
	if _, err := p.s.SeekFromStart(o); err != nil {
		if err == io.EOF || isInvalid(err) {
			return 0, plumbing.ErrObjectNotFound
		}

		return 0, err
	}

	h, err := p.nextObjectHeader()
	if err != nil {
		return 0, err
	}
	return p.getObjectSize(h)
}

func (p *Packfile) objectHeaderAtOffset(offset int64) (*ObjectHeader, error) {
	h, err := p.s.SeekObjectHeader(offset)
	p.s.pendingObject = nil
	return h, err
}

func (p *Packfile) nextObjectHeader() (*ObjectHeader, error) {
	h, err := p.s.NextObjectHeader()
	p.s.pendingObject = nil
	return h, err
}

func (p *Packfile) getDeltaObjectSize(buf *bytes.Buffer) int64 {
	delta := buf.Bytes()
	_, delta = decodeLEB128(delta) // skip src size
	sz, _ := decodeLEB128(delta)
	return int64(sz)
}

func (p *Packfile) getObjectSize(h *ObjectHeader) (int64, error) {
	switch h.Type {
	case plumbing.CommitObject, plumbing.TreeObject, plumbing.BlobObject, plumbing.TagObject:
		return h.Length, nil
	case plumbing.REFDeltaObject, plumbing.OFSDeltaObject:
		buf := bufPool.Get().(*bytes.Buffer)
		defer bufPool.Put(buf)
		buf.Reset()

		if _, _, err := p.s.NextObject(buf); err != nil {
			return 0, err
		}

		return p.getDeltaObjectSize(buf), nil
	default:
		return 0, ErrInvalidObject.AddDetails("type %q", h.Type)
	}
}

func (p *Packfile) getObjectType(h *ObjectHeader) (typ plumbing.ObjectType, err error) {
	switch h.Type {
	case plumbing.CommitObject, plumbing.TreeObject, plumbing.BlobObject, plumbing.TagObject:
		return h.Type, nil
	case plumbing.REFDeltaObject, plumbing.OFSDeltaObject:
		var offset int64
		if h.Type == plumbing.REFDeltaObject {
			offset, err = p.FindOffset(h.Reference)
			if err != nil {
				return
			}
		} else {
			offset = h.OffsetReference
		}

		if baseType, ok := p.offsetToType[offset]; ok {
			typ = baseType
		} else {
			h, err = p.objectHeaderAtOffset(offset)
			if err != nil {
				return
			}

			typ, err = p.getObjectType(h)
			if err != nil {
				return
			}
		}
	default:
		err = ErrInvalidObject.AddDetails("type %q", h.Type)
	}

	p.offsetToType[h.Offset] = typ

	return
}

func (p *Packfile) objectAtOffset(offset int64, hash plumbing.Hash) (plumbing.EncodedObject, error) {
	if obj, ok := p.cacheGet(hash); ok {
		return obj, nil
	}

	h, err := p.objectHeaderAtOffset(offset)
	if err != nil {
		if err == io.EOF || isInvalid(err) {
			return nil, plumbing.ErrObjectNotFound
		}
		return nil, err
	}

	return p.getNextObject(h, hash)
}

func (p *Packfile) getNextObject(h *ObjectHeader, hash plumbing.Hash) (plumbing.EncodedObject, error) {
	var err error

	// If we have no filesystem, we will return a MemoryObject instead
	// of an FSObject.
	if p.fs == nil {
		return p.getNextMemoryObject(h)
	}

	// If the object is small enough then read it completely into memory now since
	// it is already read from disk into buffer anyway. For delta objects we want
	// to perform the optimization too, but we have to be careful about applying
	// small deltas on big objects.
	var size int64
	if h.Length <= smallObjectThreshold {
		if h.Type != plumbing.OFSDeltaObject && h.Type != plumbing.REFDeltaObject {
			return p.getNextMemoryObject(h)
		}

		// For delta objects we read the delta data and apply the small object
		// optimization only if the expanded version of the object still meets
		// the small object threshold condition.
		buf := bufPool.Get().(*bytes.Buffer)
		defer bufPool.Put(buf)
		buf.Reset()
		if _, _, err := p.s.NextObject(buf); err != nil {
			return nil, err
		}

		size = p.getDeltaObjectSize(buf)
		if size <= smallObjectThreshold {
			var obj = new(plumbing.MemoryObject)
			obj.SetSize(size)
			if h.Type == plumbing.REFDeltaObject {
				err = p.fillREFDeltaObjectContentWithBuffer(obj, h.Reference, buf)
			} else {
				err = p.fillOFSDeltaObjectContentWithBuffer(obj, h.OffsetReference, buf)
			}
			return obj, err
		}
	} else {
		size, err = p.getObjectSize(h)
		if err != nil {
			return nil, err
		}
	}

	typ, err := p.getObjectType(h)
	if err != nil {
		return nil, err
	}

	p.offsetToType[h.Offset] = typ

	return NewFSObject(
		hash,
		typ,
		h.Offset,
		size,
		p.Index,
		p.fs,
		p.file.Name(),
		p.deltaBaseCache,
	), nil
}

func (p *Packfile) getObjectContent(offset int64) (io.ReadCloser, error) {
	h, err := p.objectHeaderAtOffset(offset)
	if err != nil {
		return nil, err
	}

	// getObjectContent is called from FSObject, so we have to explicitly
	// get memory object here to avoid recursive cycle
	obj, err := p.getNextMemoryObject(h)
	if err != nil {
		return nil, err
	}

	return obj.Reader()
}

func (p *Packfile) getNextMemoryObject(h *ObjectHeader) (plumbing.EncodedObject, error) {
	var obj = new(plumbing.MemoryObject)
	obj.SetSize(h.Length)
	obj.SetType(h.Type)

	var err error
	switch h.Type {
	case plumbing.CommitObject, plumbing.TreeObject, plumbing.BlobObject, plumbing.TagObject:
		err = p.fillRegularObjectContent(obj)
	case plumbing.REFDeltaObject:
		err = p.fillREFDeltaObjectContent(obj, h.Reference)
	case plumbing.OFSDeltaObject:
		err = p.fillOFSDeltaObjectContent(obj, h.OffsetReference)
	default:
		err = ErrInvalidObject.AddDetails("type %q", h.Type)
	}

	if err != nil {
		return nil, err
	}

	p.offsetToType[h.Offset] = obj.Type()

	return obj, nil
}

func (p *Packfile) fillRegularObjectContent(obj plumbing.EncodedObject) (err error) {
	w, err := obj.Writer()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)

	_, _, err = p.s.NextObject(w)
	p.cachePut(obj)

	return err
}

func (p *Packfile) fillREFDeltaObjectContent(obj plumbing.EncodedObject, ref plumbing.Hash) error {
	buf := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(buf)
	buf.Reset()
	_, _, err := p.s.NextObject(buf)
	if err != nil {
		return err
	}

	return p.fillREFDeltaObjectContentWithBuffer(obj, ref, buf)
}

func (p *Packfile) fillREFDeltaObjectContentWithBuffer(obj plumbing.EncodedObject, ref plumbing.Hash, buf *bytes.Buffer) error {
	var err error

	base, ok := p.cacheGet(ref)
	if !ok {
		base, err = p.Get(ref)
		if err != nil {
			return err
		}
	}

	obj.SetType(base.Type())
	err = ApplyDelta(obj, base, buf.Bytes())
	p.cachePut(obj)

	return err
}

func (p *Packfile) fillOFSDeltaObjectContent(obj plumbing.EncodedObject, offset int64) error {
	buf := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(buf)
	buf.Reset()
	_, _, err := p.s.NextObject(buf)
	if err != nil {
		return err
	}

	return p.fillOFSDeltaObjectContentWithBuffer(obj, offset, buf)
}

func (p *Packfile) fillOFSDeltaObjectContentWithBuffer(obj plumbing.EncodedObject, offset int64, buf *bytes.Buffer) error {
	hash, err := p.FindHash(offset)
	if err != nil {
		return err
	}

	base, err := p.objectAtOffset(offset, hash)
	if err != nil {
		return err
	}

	obj.SetType(base.Type())
	err = ApplyDelta(obj, base, buf.Bytes())
	p.cachePut(obj)

	return err
}

func (p *Packfile) cacheGet(h plumbing.Hash) (plumbing.EncodedObject, bool) {
	if p.deltaBaseCache == nil {
		return nil, false
	}

	return p.deltaBaseCache.Get(h)
}

func (p *Packfile) cachePut(obj plumbing.EncodedObject) {
	if p.deltaBaseCache == nil {
		return
	}

	p.deltaBaseCache.Put(obj)
}

// GetAll returns an iterator with all encoded objects in the packfile.
// The iterator returned is not thread-safe, it should be used in the same
// thread as the Packfile instance.
func (p *Packfile) GetAll() (storer.EncodedObjectIter, error) {
	return p.GetByType(plumbing.AnyObject)
}

// GetByType returns all the objects of the given type.
func (p *Packfile) GetByType(typ plumbing.ObjectType) (storer.EncodedObjectIter, error) {
	switch typ {
	case plumbing.AnyObject,
		plumbing.BlobObject,
		plumbing.TreeObject,
		plumbing.CommitObject,
		plumbing.TagObject:
		entries, err := p.EntriesByOffset()
		if err != nil {
			return nil, err
		}

		return &objectIter{
			// Easiest way to provide an object decoder is just to pass a Packfile
			// instance. To not mess with the seeks, it's a new instance with a
			// different scanner but the same cache and offset to hash map for
			// reusing as much cache as possible.
			p:    p,
			iter: entries,
			typ:  typ,
		}, nil
	default:
		return nil, plumbing.ErrInvalidType
	}
}

// ID returns the ID of the packfile, which is the checksum at the end of it.
func (p *Packfile) ID() (plumbing.Hash, error) {
	prev, err := p.file.Seek(-20, io.SeekEnd)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	var hash plumbing.Hash
	if _, err := io.ReadFull(p.file, hash[:]); err != nil {
		return plumbing.ZeroHash, err
	}

	if _, err := p.file.Seek(prev, io.SeekStart); err != nil {
		return plumbing.ZeroHash, err
	}

	return hash, nil
}

// Scanner returns the packfile's Scanner
func (p *Packfile) Scanner() *Scanner {
	return p.s
}

// Close the packfile and its resources.
func (p *Packfile) Close() error {
	closer, ok := p.file.(io.Closer)
	if !ok {
		return nil
	}

	return closer.Close()
}

type objectIter struct {
	p    *Packfile
	typ  plumbing.ObjectType
	iter idxfile.EntryIter
}

func (i *objectIter) Next() (plumbing.EncodedObject, error) {
	for {
		e, err := i.iter.Next()
		if err != nil {
			return nil, err
		}

		if i.typ != plumbing.AnyObject {
			if typ, ok := i.p.offsetToType[int64(e.Offset)]; ok {
				if typ != i.typ {
					continue
				}
			} else if obj, ok := i.p.cacheGet(e.Hash); ok {
				if obj.Type() != i.typ {
					i.p.offsetToType[int64(e.Offset)] = obj.Type()
					continue
				}
				return obj, nil
			} else {
				h, err := i.p.objectHeaderAtOffset(int64(e.Offset))
				if err != nil {
					return nil, err
				}

				if h.Type == plumbing.REFDeltaObject || h.Type == plumbing.OFSDeltaObject {
					typ, err := i.p.getObjectType(h)
					if err != nil {
						return nil, err
					}
					if typ != i.typ {
						i.p.offsetToType[int64(e.Offset)] = typ
						continue
					}
					// getObjectType will seek in the file so we cannot use getNextObject safely
					return i.p.objectAtOffset(int64(e.Offset), e.Hash)
				} else {
					if h.Type != i.typ {
						i.p.offsetToType[int64(e.Offset)] = h.Type
						continue
					}
					return i.p.getNextObject(h, e.Hash)
				}
			}
		}

		obj, err := i.p.objectAtOffset(int64(e.Offset), e.Hash)
		if err != nil {
			return nil, err
		}

		return obj, nil
	}
}

func (i *objectIter) ForEach(f func(plumbing.EncodedObject) error) error {
	for {
		o, err := i.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		if err := f(o); err != nil {
			return err
		}
	}
}

func (i *objectIter) Close() {
	i.iter.Close()
}

// isInvalid checks whether an error is an os.PathError with an os.ErrInvalid
// error inside. It also checks for the windows error, which is different from
// os.ErrInvalid.
func isInvalid(err error) bool {
	pe, ok := err.(*os.PathError)
	if !ok {
		return false
	}

	errstr := pe.Err.Error()
	return errstr == errInvalidUnix || errstr == errInvalidWindows
}

// errInvalidWindows is the Windows equivalent to os.ErrInvalid
const errInvalidWindows = "The parameter is incorrect."

var errInvalidUnix = os.ErrInvalid.Error()
