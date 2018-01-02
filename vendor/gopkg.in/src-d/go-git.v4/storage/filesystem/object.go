package filesystem

import (
	"io"
	"os"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/cache"
	"gopkg.in/src-d/go-git.v4/plumbing/format/idxfile"
	"gopkg.in/src-d/go-git.v4/plumbing/format/objfile"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit"
	"gopkg.in/src-d/go-git.v4/storage/memory"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"

	"gopkg.in/src-d/go-billy.v3"
)

const DefaultMaxDeltaBaseCacheSize = 92 * cache.MiByte

type ObjectStorage struct {
	// DeltaBaseCache is an object cache uses to cache delta's bases when
	DeltaBaseCache cache.Object

	dir   *dotgit.DotGit
	index map[plumbing.Hash]*packfile.Index
}

func newObjectStorage(dir *dotgit.DotGit) (ObjectStorage, error) {
	s := ObjectStorage{
		DeltaBaseCache: cache.NewObjectLRU(DefaultMaxDeltaBaseCacheSize),
		dir:            dir,
	}

	return s, nil
}

func (s *ObjectStorage) requireIndex() error {
	if s.index != nil {
		return nil
	}

	s.index = make(map[plumbing.Hash]*packfile.Index, 0)
	packs, err := s.dir.ObjectPacks()
	if err != nil {
		return err
	}

	for _, h := range packs {
		if err := s.loadIdxFile(h); err != nil {
			return err
		}
	}

	return nil
}

func (s *ObjectStorage) loadIdxFile(h plumbing.Hash) error {
	f, err := s.dir.ObjectPackIdx(h)
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)
	idxf := idxfile.NewIdxfile()
	d := idxfile.NewDecoder(f)
	if err = d.Decode(idxf); err != nil {
		return err
	}

	s.index[h] = packfile.NewIndexFromIdxFile(idxf)
	return err
}

func (s *ObjectStorage) NewEncodedObject() plumbing.EncodedObject {
	return &plumbing.MemoryObject{}
}

func (s *ObjectStorage) PackfileWriter() (io.WriteCloser, error) {
	if err := s.requireIndex(); err != nil {
		return nil, err
	}

	w, err := s.dir.NewObjectPack()
	if err != nil {
		return nil, err
	}

	w.Notify = func(h plumbing.Hash, idx *packfile.Index) {
		s.index[h] = idx
	}

	return w, nil
}

// SetEncodedObject adds a new object to the storage.
func (s *ObjectStorage) SetEncodedObject(o plumbing.EncodedObject) (plumbing.Hash, error) {
	if o.Type() == plumbing.OFSDeltaObject || o.Type() == plumbing.REFDeltaObject {
		return plumbing.ZeroHash, plumbing.ErrInvalidType
	}

	ow, err := s.dir.NewObject()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	defer ioutil.CheckClose(ow, &err)

	or, err := o.Reader()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	defer ioutil.CheckClose(or, &err)

	if err := ow.WriteHeader(o.Type(), o.Size()); err != nil {
		return plumbing.ZeroHash, err
	}

	if _, err := io.Copy(ow, or); err != nil {
		return plumbing.ZeroHash, err
	}

	return o.Hash(), err
}

// EncodedObject returns the object with the given hash, by searching for it in
// the packfile and the git object directories.
func (s *ObjectStorage) EncodedObject(t plumbing.ObjectType, h plumbing.Hash) (plumbing.EncodedObject, error) {
	obj, err := s.getFromUnpacked(h)
	if err == plumbing.ErrObjectNotFound {
		obj, err = s.getFromPackfile(h, false)
	}

	if err != nil {
		return nil, err
	}

	if plumbing.AnyObject != t && obj.Type() != t {
		return nil, plumbing.ErrObjectNotFound
	}

	return obj, nil
}

// DeltaObject returns the object with the given hash, by searching for
// it in the packfile and the git object directories.
func (s *ObjectStorage) DeltaObject(t plumbing.ObjectType,
	h plumbing.Hash) (plumbing.EncodedObject, error) {
	obj, err := s.getFromUnpacked(h)
	if err == plumbing.ErrObjectNotFound {
		obj, err = s.getFromPackfile(h, true)
	}

	if err != nil {
		return nil, err
	}

	if plumbing.AnyObject != t && obj.Type() != t {
		return nil, plumbing.ErrObjectNotFound
	}

	return obj, nil
}

func (s *ObjectStorage) getFromUnpacked(h plumbing.Hash) (obj plumbing.EncodedObject, err error) {
	f, err := s.dir.Object(h)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, plumbing.ErrObjectNotFound
		}

		return nil, err
	}

	defer ioutil.CheckClose(f, &err)

	obj = s.NewEncodedObject()
	r, err := objfile.NewReader(f)
	if err != nil {
		return nil, err
	}

	defer ioutil.CheckClose(r, &err)

	t, size, err := r.Header()
	if err != nil {
		return nil, err
	}

	obj.SetType(t)
	obj.SetSize(size)
	w, err := obj.Writer()
	if err != nil {
		return nil, err
	}

	_, err = io.Copy(w, r)
	return obj, err
}

// Get returns the object with the given hash, by searching for it in
// the packfile.
func (s *ObjectStorage) getFromPackfile(h plumbing.Hash, canBeDelta bool) (
	plumbing.EncodedObject, error) {

	if err := s.requireIndex(); err != nil {
		return nil, err
	}

	pack, hash, offset := s.findObjectInPackfile(h)
	if offset == -1 {
		return nil, plumbing.ErrObjectNotFound
	}

	f, err := s.dir.ObjectPack(pack)
	if err != nil {
		return nil, err
	}

	defer ioutil.CheckClose(f, &err)

	idx := s.index[pack]
	if canBeDelta {
		return s.decodeDeltaObjectAt(f, idx, offset, hash)
	}

	return s.decodeObjectAt(f, idx, offset)
}

func (s *ObjectStorage) decodeObjectAt(
	f billy.File,
	idx *packfile.Index,
	offset int64) (plumbing.EncodedObject, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	p := packfile.NewScanner(f)

	d, err := packfile.NewDecoder(p, memory.NewStorage())
	if err != nil {
		return nil, err
	}

	d.SetIndex(idx)
	d.DeltaBaseCache = s.DeltaBaseCache
	obj, err := d.DecodeObjectAt(offset)
	return obj, err
}

func (s *ObjectStorage) decodeDeltaObjectAt(
	f billy.File,
	idx *packfile.Index,
	offset int64,
	hash plumbing.Hash) (plumbing.EncodedObject, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	p := packfile.NewScanner(f)
	if _, err := p.SeekFromStart(offset); err != nil {
		return nil, err
	}

	header, err := p.NextObjectHeader()
	if err != nil {
		return nil, err
	}

	var (
		base plumbing.Hash
	)

	switch header.Type {
	case plumbing.REFDeltaObject:
		base = header.Reference
	case plumbing.OFSDeltaObject:
		e, ok := idx.LookupOffset(uint64(header.OffsetReference))
		if !ok {
			return nil, plumbing.ErrObjectNotFound
		}

		base = e.Hash
	default:
		return s.decodeObjectAt(f, idx, offset)
	}

	obj := &plumbing.MemoryObject{}
	obj.SetType(header.Type)
	w, err := obj.Writer()
	if err != nil {
		return nil, err
	}

	if _, _, err := p.NextObject(w); err != nil {
		return nil, err
	}

	return newDeltaObject(obj, hash, base, header.Length), nil
}

func (s *ObjectStorage) findObjectInPackfile(h plumbing.Hash) (plumbing.Hash, plumbing.Hash, int64) {
	for packfile, index := range s.index {
		if e, ok := index.LookupHash(h); ok {
			return packfile, e.Hash, int64(e.Offset)
		}
	}

	return plumbing.ZeroHash, plumbing.ZeroHash, -1
}

// IterEncodedObjects returns an iterator for all the objects in the packfile
// with the given type.
func (s *ObjectStorage) IterEncodedObjects(t plumbing.ObjectType) (storer.EncodedObjectIter, error) {
	objects, err := s.dir.Objects()
	if err != nil {
		return nil, err
	}

	seen := make(map[plumbing.Hash]bool, 0)
	var iters []storer.EncodedObjectIter
	if len(objects) != 0 {
		iters = append(iters, &objectsIter{s: s, t: t, h: objects})
		seen = hashListAsMap(objects)
	}

	packi, err := s.buildPackfileIters(t, seen)
	if err != nil {
		return nil, err
	}

	iters = append(iters, packi...)
	return storer.NewMultiEncodedObjectIter(iters), nil
}

func (s *ObjectStorage) buildPackfileIters(t plumbing.ObjectType, seen map[plumbing.Hash]bool) ([]storer.EncodedObjectIter, error) {
	if err := s.requireIndex(); err != nil {
		return nil, err
	}

	packs, err := s.dir.ObjectPacks()
	if err != nil {
		return nil, err
	}

	var iters []storer.EncodedObjectIter
	for _, h := range packs {
		pack, err := s.dir.ObjectPack(h)
		if err != nil {
			return nil, err
		}

		iter, err := newPackfileIter(pack, t, seen, s.index[h], s.DeltaBaseCache)
		if err != nil {
			return nil, err
		}

		iters = append(iters, iter)
	}

	return iters, nil
}

type packfileIter struct {
	f billy.File
	d *packfile.Decoder
	t plumbing.ObjectType

	seen     map[plumbing.Hash]bool
	position uint32
	total    uint32
}

func NewPackfileIter(f billy.File, t plumbing.ObjectType) (storer.EncodedObjectIter, error) {
	return newPackfileIter(f, t, make(map[plumbing.Hash]bool), nil, nil)
}

func newPackfileIter(f billy.File, t plumbing.ObjectType, seen map[plumbing.Hash]bool,
	index *packfile.Index, cache cache.Object) (storer.EncodedObjectIter, error) {
	s := packfile.NewScanner(f)
	_, total, err := s.Header()
	if err != nil {
		return nil, err
	}

	d, err := packfile.NewDecoderForType(s, memory.NewStorage(), t)
	if err != nil {
		return nil, err
	}

	d.SetIndex(index)
	d.DeltaBaseCache = cache

	return &packfileIter{
		f: f,
		d: d,
		t: t,

		total: total,
		seen:  seen,
	}, nil
}

func (iter *packfileIter) Next() (plumbing.EncodedObject, error) {
	for {
		if iter.position >= iter.total {
			return nil, io.EOF
		}

		obj, err := iter.d.DecodeObject()
		if err != nil {
			return nil, err
		}

		iter.position++
		if obj == nil {
			continue
		}

		if iter.seen[obj.Hash()] {
			return iter.Next()
		}

		return obj, nil
	}
}

// ForEach is never called since is used inside of a MultiObjectIterator
func (iter *packfileIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return nil
}

func (iter *packfileIter) Close() {
	iter.f.Close()
	iter.d.Close()
}

type objectsIter struct {
	s *ObjectStorage
	t plumbing.ObjectType
	h []plumbing.Hash
}

func (iter *objectsIter) Next() (plumbing.EncodedObject, error) {
	if len(iter.h) == 0 {
		return nil, io.EOF
	}

	obj, err := iter.s.getFromUnpacked(iter.h[0])
	iter.h = iter.h[1:]

	if err != nil {
		return nil, err
	}

	if iter.t != plumbing.AnyObject && iter.t != obj.Type() {
		return iter.Next()
	}

	return obj, err
}

// ForEach is never called since is used inside of a MultiObjectIterator
func (iter *objectsIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return nil
}

func (iter *objectsIter) Close() {
	iter.h = []plumbing.Hash{}
}

func hashListAsMap(l []plumbing.Hash) map[plumbing.Hash]bool {
	m := make(map[plumbing.Hash]bool, len(l))
	for _, h := range l {
		m[h] = true
	}

	return m
}
