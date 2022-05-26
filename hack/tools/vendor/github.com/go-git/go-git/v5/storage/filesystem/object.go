package filesystem

import (
	"bytes"
	"io"
	"os"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/cache"
	"github.com/go-git/go-git/v5/plumbing/format/idxfile"
	"github.com/go-git/go-git/v5/plumbing/format/objfile"
	"github.com/go-git/go-git/v5/plumbing/format/packfile"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/storage/filesystem/dotgit"
	"github.com/go-git/go-git/v5/utils/ioutil"

	"github.com/go-git/go-billy/v5"
)

type ObjectStorage struct {
	options Options

	// objectCache is an object cache uses to cache delta's bases and also recently
	// loaded loose objects
	objectCache cache.Object

	dir   *dotgit.DotGit
	index map[plumbing.Hash]idxfile.Index

	packList    []plumbing.Hash
	packListIdx int
	packfiles   map[plumbing.Hash]*packfile.Packfile
}

// NewObjectStorage creates a new ObjectStorage with the given .git directory and cache.
func NewObjectStorage(dir *dotgit.DotGit, objectCache cache.Object) *ObjectStorage {
	return NewObjectStorageWithOptions(dir, objectCache, Options{})
}

// NewObjectStorageWithOptions creates a new ObjectStorage with the given .git directory, cache and extra options
func NewObjectStorageWithOptions(dir *dotgit.DotGit, objectCache cache.Object, ops Options) *ObjectStorage {
	return &ObjectStorage{
		options:     ops,
		objectCache: objectCache,
		dir:         dir,
	}
}

func (s *ObjectStorage) requireIndex() error {
	if s.index != nil {
		return nil
	}

	s.index = make(map[plumbing.Hash]idxfile.Index)
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

// Reindex indexes again all packfiles. Useful if git changed packfiles externally
func (s *ObjectStorage) Reindex() {
	s.index = nil
}

func (s *ObjectStorage) loadIdxFile(h plumbing.Hash) (err error) {
	f, err := s.dir.ObjectPackIdx(h)
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)

	idxf := idxfile.NewMemoryIndex()
	d := idxfile.NewDecoder(f)
	if err = d.Decode(idxf); err != nil {
		return err
	}

	s.index[h] = idxf
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

	w.Notify = func(h plumbing.Hash, writer *idxfile.Writer) {
		index, err := writer.Index()
		if err == nil {
			s.index[h] = index
		}
	}

	return w, nil
}

// SetEncodedObject adds a new object to the storage.
func (s *ObjectStorage) SetEncodedObject(o plumbing.EncodedObject) (h plumbing.Hash, err error) {
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

	if err = ow.WriteHeader(o.Type(), o.Size()); err != nil {
		return plumbing.ZeroHash, err
	}

	if _, err = io.Copy(ow, or); err != nil {
		return plumbing.ZeroHash, err
	}

	return o.Hash(), err
}

// HasEncodedObject returns nil if the object exists, without actually
// reading the object data from storage.
func (s *ObjectStorage) HasEncodedObject(h plumbing.Hash) (err error) {
	// Check unpacked objects
	f, err := s.dir.Object(h)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		// Fall through to check packed objects.
	} else {
		defer ioutil.CheckClose(f, &err)
		return nil
	}

	// Check packed objects.
	if err := s.requireIndex(); err != nil {
		return err
	}
	_, _, offset := s.findObjectInPackfile(h)
	if offset == -1 {
		return plumbing.ErrObjectNotFound
	}
	return nil
}

func (s *ObjectStorage) encodedObjectSizeFromUnpacked(h plumbing.Hash) (
	size int64, err error) {
	f, err := s.dir.Object(h)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, plumbing.ErrObjectNotFound
		}

		return 0, err
	}

	r, err := objfile.NewReader(f)
	if err != nil {
		return 0, err
	}
	defer ioutil.CheckClose(r, &err)

	_, size, err = r.Header()
	return size, err
}

func (s *ObjectStorage) packfile(idx idxfile.Index, pack plumbing.Hash) (*packfile.Packfile, error) {
	if p := s.packfileFromCache(pack); p != nil {
		return p, nil
	}

	f, err := s.dir.ObjectPack(pack)
	if err != nil {
		return nil, err
	}

	var p *packfile.Packfile
	if s.objectCache != nil {
		p = packfile.NewPackfileWithCache(idx, s.dir.Fs(), f, s.objectCache)
	} else {
		p = packfile.NewPackfile(idx, s.dir.Fs(), f)
	}

	return p, s.storePackfileInCache(pack, p)
}

func (s *ObjectStorage) packfileFromCache(hash plumbing.Hash) *packfile.Packfile {
	if s.packfiles == nil {
		if s.options.KeepDescriptors {
			s.packfiles = make(map[plumbing.Hash]*packfile.Packfile)
		} else if s.options.MaxOpenDescriptors > 0 {
			s.packList = make([]plumbing.Hash, s.options.MaxOpenDescriptors)
			s.packfiles = make(map[plumbing.Hash]*packfile.Packfile, s.options.MaxOpenDescriptors)
		}
	}

	return s.packfiles[hash]
}

func (s *ObjectStorage) storePackfileInCache(hash plumbing.Hash, p *packfile.Packfile) error {
	if s.options.KeepDescriptors {
		s.packfiles[hash] = p
		return nil
	}

	if s.options.MaxOpenDescriptors <= 0 {
		return nil
	}

	// start over as the limit of packList is hit
	if s.packListIdx >= len(s.packList) {
		s.packListIdx = 0
	}

	// close the existing packfile if open
	if next := s.packList[s.packListIdx]; !next.IsZero() {
		open := s.packfiles[next]
		delete(s.packfiles, next)
		if open != nil {
			if err := open.Close(); err != nil {
				return err
			}
		}
	}

	// cache newly open packfile
	s.packList[s.packListIdx] = hash
	s.packfiles[hash] = p
	s.packListIdx++

	return nil
}

func (s *ObjectStorage) encodedObjectSizeFromPackfile(h plumbing.Hash) (
	size int64, err error) {
	if err := s.requireIndex(); err != nil {
		return 0, err
	}

	pack, _, offset := s.findObjectInPackfile(h)
	if offset == -1 {
		return 0, plumbing.ErrObjectNotFound
	}

	idx := s.index[pack]
	hash, err := idx.FindHash(offset)
	if err == nil {
		obj, ok := s.objectCache.Get(hash)
		if ok {
			return obj.Size(), nil
		}
	} else if err != nil && err != plumbing.ErrObjectNotFound {
		return 0, err
	}

	p, err := s.packfile(idx, pack)
	if err != nil {
		return 0, err
	}

	if !s.options.KeepDescriptors && s.options.MaxOpenDescriptors == 0 {
		defer ioutil.CheckClose(p, &err)
	}

	return p.GetSizeByOffset(offset)
}

// EncodedObjectSize returns the plaintext size of the given object,
// without actually reading the full object data from storage.
func (s *ObjectStorage) EncodedObjectSize(h plumbing.Hash) (
	size int64, err error) {
	size, err = s.encodedObjectSizeFromUnpacked(h)
	if err != nil && err != plumbing.ErrObjectNotFound {
		return 0, err
	} else if err == nil {
		return size, nil
	}

	return s.encodedObjectSizeFromPackfile(h)
}

// EncodedObject returns the object with the given hash, by searching for it in
// the packfile and the git object directories.
func (s *ObjectStorage) EncodedObject(t plumbing.ObjectType, h plumbing.Hash) (plumbing.EncodedObject, error) {
	var obj plumbing.EncodedObject
	var err error

	if s.index != nil {
		obj, err = s.getFromPackfile(h, false)
		if err == plumbing.ErrObjectNotFound {
			obj, err = s.getFromUnpacked(h)
		}
	} else {
		obj, err = s.getFromUnpacked(h)
		if err == plumbing.ErrObjectNotFound {
			obj, err = s.getFromPackfile(h, false)
		}
	}

	// If the error is still object not found, check if it's a shared object
	// repository.
	if err == plumbing.ErrObjectNotFound {
		dotgits, e := s.dir.Alternates()
		if e == nil {
			// Create a new object storage with the DotGit(s) and check for the
			// required hash object. Skip when not found.
			for _, dg := range dotgits {
				o := NewObjectStorage(dg, s.objectCache)
				enobj, enerr := o.EncodedObject(t, h)
				if enerr != nil {
					continue
				}
				return enobj, nil
			}
		}
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

	if cacheObj, found := s.objectCache.Get(h); found {
		return cacheObj, nil
	}

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

	defer ioutil.CheckClose(w, &err)

	s.objectCache.Put(obj)

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

	idx := s.index[pack]
	p, err := s.packfile(idx, pack)
	if err != nil {
		return nil, err
	}

	if !s.options.KeepDescriptors && s.options.MaxOpenDescriptors == 0 {
		defer ioutil.CheckClose(p, &err)
	}

	if canBeDelta {
		return s.decodeDeltaObjectAt(p, offset, hash)
	}

	return s.decodeObjectAt(p, offset)
}

func (s *ObjectStorage) decodeObjectAt(
	p *packfile.Packfile,
	offset int64,
) (plumbing.EncodedObject, error) {
	hash, err := p.FindHash(offset)
	if err == nil {
		obj, ok := s.objectCache.Get(hash)
		if ok {
			return obj, nil
		}
	}

	if err != nil && err != plumbing.ErrObjectNotFound {
		return nil, err
	}

	return p.GetByOffset(offset)
}

func (s *ObjectStorage) decodeDeltaObjectAt(
	p *packfile.Packfile,
	offset int64,
	hash plumbing.Hash,
) (plumbing.EncodedObject, error) {
	scan := p.Scanner()
	header, err := scan.SeekObjectHeader(offset)
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
		base, err = p.FindHash(header.OffsetReference)
		if err != nil {
			return nil, err
		}
	default:
		return s.decodeObjectAt(p, offset)
	}

	obj := &plumbing.MemoryObject{}
	obj.SetType(header.Type)
	w, err := obj.Writer()
	if err != nil {
		return nil, err
	}

	if _, _, err := scan.NextObject(w); err != nil {
		return nil, err
	}

	return newDeltaObject(obj, hash, base, header.Length), nil
}

func (s *ObjectStorage) findObjectInPackfile(h plumbing.Hash) (plumbing.Hash, plumbing.Hash, int64) {
	for packfile, index := range s.index {
		offset, err := index.FindOffset(h)
		if err == nil {
			return packfile, h, offset
		}
	}

	return plumbing.ZeroHash, plumbing.ZeroHash, -1
}

func (s *ObjectStorage) HashesWithPrefix(prefix []byte) ([]plumbing.Hash, error) {
	hashes, err := s.dir.ObjectsWithPrefix(prefix)
	if err != nil {
		return nil, err
	}

	// TODO: This could be faster with some idxfile changes,
	// or diving into the packfile.
	for _, index := range s.index {
		ei, err := index.Entries()
		if err != nil {
			return nil, err
		}
		for {
			e, err := ei.Next()
			if err == io.EOF {
				break
			} else if err != nil {
				return nil, err
			}
			if bytes.HasPrefix(e.Hash[:], prefix) {
				hashes = append(hashes, e.Hash)
			}
		}
		ei.Close()
	}

	return hashes, nil
}

// IterEncodedObjects returns an iterator for all the objects in the packfile
// with the given type.
func (s *ObjectStorage) IterEncodedObjects(t plumbing.ObjectType) (storer.EncodedObjectIter, error) {
	objects, err := s.dir.Objects()
	if err != nil {
		return nil, err
	}

	seen := make(map[plumbing.Hash]struct{})
	var iters []storer.EncodedObjectIter
	if len(objects) != 0 {
		iters = append(iters, &objectsIter{s: s, t: t, h: objects})
		seen = hashListAsMap(objects)
	}

	packi, err := s.buildPackfileIters(t, seen)
	if err != nil {
		return nil, err
	}

	iters = append(iters, packi)
	return storer.NewMultiEncodedObjectIter(iters), nil
}

func (s *ObjectStorage) buildPackfileIters(
	t plumbing.ObjectType,
	seen map[plumbing.Hash]struct{},
) (storer.EncodedObjectIter, error) {
	if err := s.requireIndex(); err != nil {
		return nil, err
	}

	packs, err := s.dir.ObjectPacks()
	if err != nil {
		return nil, err
	}
	return &lazyPackfilesIter{
		hashes: packs,
		open: func(h plumbing.Hash) (storer.EncodedObjectIter, error) {
			pack, err := s.dir.ObjectPack(h)
			if err != nil {
				return nil, err
			}
			return newPackfileIter(
				s.dir.Fs(), pack, t, seen, s.index[h],
				s.objectCache, s.options.KeepDescriptors,
			)
		},
	}, nil
}

// Close closes all opened files.
func (s *ObjectStorage) Close() error {
	var firstError error
	if s.options.KeepDescriptors || s.options.MaxOpenDescriptors > 0 {
		for _, packfile := range s.packfiles {
			err := packfile.Close()
			if firstError == nil && err != nil {
				firstError = err
			}
		}
	}

	s.packfiles = nil
	s.dir.Close()

	return firstError
}

type lazyPackfilesIter struct {
	hashes []plumbing.Hash
	open   func(h plumbing.Hash) (storer.EncodedObjectIter, error)
	cur    storer.EncodedObjectIter
}

func (it *lazyPackfilesIter) Next() (plumbing.EncodedObject, error) {
	for {
		if it.cur == nil {
			if len(it.hashes) == 0 {
				return nil, io.EOF
			}
			h := it.hashes[0]
			it.hashes = it.hashes[1:]

			sub, err := it.open(h)
			if err == io.EOF {
				continue
			} else if err != nil {
				return nil, err
			}
			it.cur = sub
		}
		ob, err := it.cur.Next()
		if err == io.EOF {
			it.cur.Close()
			it.cur = nil
			continue
		} else if err != nil {
			return nil, err
		}
		return ob, nil
	}
}

func (it *lazyPackfilesIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return storer.ForEachIterator(it, cb)
}

func (it *lazyPackfilesIter) Close() {
	if it.cur != nil {
		it.cur.Close()
		it.cur = nil
	}
	it.hashes = nil
}

type packfileIter struct {
	pack billy.File
	iter storer.EncodedObjectIter
	seen map[plumbing.Hash]struct{}

	// tells whether the pack file should be left open after iteration or not
	keepPack bool
}

// NewPackfileIter returns a new EncodedObjectIter for the provided packfile
// and object type. Packfile and index file will be closed after they're
// used. If keepPack is true the packfile won't be closed after the iteration
// finished.
func NewPackfileIter(
	fs billy.Filesystem,
	f billy.File,
	idxFile billy.File,
	t plumbing.ObjectType,
	keepPack bool,
) (storer.EncodedObjectIter, error) {
	idx := idxfile.NewMemoryIndex()
	if err := idxfile.NewDecoder(idxFile).Decode(idx); err != nil {
		return nil, err
	}

	if err := idxFile.Close(); err != nil {
		return nil, err
	}

	seen := make(map[plumbing.Hash]struct{})
	return newPackfileIter(fs, f, t, seen, idx, nil, keepPack)
}

func newPackfileIter(
	fs billy.Filesystem,
	f billy.File,
	t plumbing.ObjectType,
	seen map[plumbing.Hash]struct{},
	index idxfile.Index,
	cache cache.Object,
	keepPack bool,
) (storer.EncodedObjectIter, error) {
	var p *packfile.Packfile
	if cache != nil {
		p = packfile.NewPackfileWithCache(index, fs, f, cache)
	} else {
		p = packfile.NewPackfile(index, fs, f)
	}

	iter, err := p.GetByType(t)
	if err != nil {
		return nil, err
	}

	return &packfileIter{
		pack:     f,
		iter:     iter,
		seen:     seen,
		keepPack: keepPack,
	}, nil
}

func (iter *packfileIter) Next() (plumbing.EncodedObject, error) {
	for {
		obj, err := iter.iter.Next()
		if err != nil {
			return nil, err
		}

		if _, ok := iter.seen[obj.Hash()]; ok {
			continue
		}

		return obj, nil
	}
}

func (iter *packfileIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	for {
		o, err := iter.Next()
		if err != nil {
			if err == io.EOF {
				iter.Close()
				return nil
			}
			return err
		}

		if err := cb(o); err != nil {
			return err
		}
	}
}

func (iter *packfileIter) Close() {
	iter.iter.Close()
	if !iter.keepPack {
		_ = iter.pack.Close()
	}
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

func (iter *objectsIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	for {
		o, err := iter.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		if err := cb(o); err != nil {
			return err
		}
	}
}

func (iter *objectsIter) Close() {
	iter.h = []plumbing.Hash{}
}

func hashListAsMap(l []plumbing.Hash) map[plumbing.Hash]struct{} {
	m := make(map[plumbing.Hash]struct{}, len(l))
	for _, h := range l {
		m[h] = struct{}{}
	}
	return m
}

func (s *ObjectStorage) ForEachObjectHash(fun func(plumbing.Hash) error) error {
	err := s.dir.ForEachObjectHash(fun)
	if err == storer.ErrStop {
		return nil
	}
	return err
}

func (s *ObjectStorage) LooseObjectTime(hash plumbing.Hash) (time.Time, error) {
	fi, err := s.dir.ObjectStat(hash)
	if err != nil {
		return time.Time{}, err
	}
	return fi.ModTime(), nil
}

func (s *ObjectStorage) DeleteLooseObject(hash plumbing.Hash) error {
	return s.dir.ObjectDelete(hash)
}

func (s *ObjectStorage) ObjectPacks() ([]plumbing.Hash, error) {
	return s.dir.ObjectPacks()
}

func (s *ObjectStorage) DeleteOldObjectPackAndIndex(h plumbing.Hash, t time.Time) error {
	return s.dir.DeleteOldObjectPackAndIndex(h, t)
}
