package packfile

import (
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/idxfile"
)

// Index is an in-memory representation of a packfile index.
// This uses idxfile.Idxfile under the hood to obtain indexes from .idx files
// or to store them.
type Index struct {
	byHash   map[plumbing.Hash]*idxfile.Entry
	byOffset map[uint64]*idxfile.Entry
}

// NewIndex creates a new empty index with the given size. Size is a hint and
// can be 0. It is recommended to set it to the number of objects to be indexed
// if it is known beforehand (e.g. reading from a packfile).
func NewIndex(size int) *Index {
	return &Index{
		byHash:   make(map[plumbing.Hash]*idxfile.Entry, size),
		byOffset: make(map[uint64]*idxfile.Entry, size),
	}
}

// NewIndexFromIdxFile creates a new Index from an idxfile.IdxFile.
func NewIndexFromIdxFile(idxf *idxfile.Idxfile) *Index {
	idx := &Index{
		byHash:   make(map[plumbing.Hash]*idxfile.Entry, idxf.ObjectCount),
		byOffset: make(map[uint64]*idxfile.Entry, idxf.ObjectCount),
	}
	for _, e := range idxf.Entries {
		idx.add(e)
	}

	return idx
}

// Add adds a new Entry with the given values to the index.
func (idx *Index) Add(h plumbing.Hash, offset uint64, crc32 uint32) {
	e := idxfile.Entry{
		Hash:   h,
		Offset: offset,
		CRC32:  crc32,
	}
	idx.add(&e)
}

func (idx *Index) add(e *idxfile.Entry) {
	idx.byHash[e.Hash] = e
	idx.byOffset[e.Offset] = e
}

// LookupHash looks an entry up by its hash. An idxfile.Entry is returned and
// a bool, which is true if it was found or false if it wasn't.
func (idx *Index) LookupHash(h plumbing.Hash) (*idxfile.Entry, bool) {
	e, ok := idx.byHash[h]
	return e, ok
}

// LookupHash looks an entry up by its offset in the packfile. An idxfile.Entry
// is returned and a bool, which is true if it was found or false if it wasn't.
func (idx *Index) LookupOffset(offset uint64) (*idxfile.Entry, bool) {
	e, ok := idx.byOffset[offset]
	return e, ok
}

// Size returns the number of entries in the index.
func (idx *Index) Size() int {
	return len(idx.byHash)
}

// ToIdxFile converts the index to an idxfile.Idxfile, which can then be used
// to serialize.
func (idx *Index) ToIdxFile() *idxfile.Idxfile {
	idxf := idxfile.NewIdxfile()
	for _, e := range idx.byHash {
		idxf.Entries = append(idxf.Entries, e)
	}

	return idxf
}
