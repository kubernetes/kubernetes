package idxfile

import (
	"bytes"
	"io"
	"sort"

	encbin "encoding/binary"

	"github.com/go-git/go-git/v5/plumbing"
)

const (
	// VersionSupported is the only idx version supported.
	VersionSupported = 2

	noMapping = -1
)

var (
	idxHeader = []byte{255, 't', 'O', 'c'}
)

// Index represents an index of a packfile.
type Index interface {
	// Contains checks whether the given hash is in the index.
	Contains(h plumbing.Hash) (bool, error)
	// FindOffset finds the offset in the packfile for the object with
	// the given hash.
	FindOffset(h plumbing.Hash) (int64, error)
	// FindCRC32 finds the CRC32 of the object with the given hash.
	FindCRC32(h plumbing.Hash) (uint32, error)
	// FindHash finds the hash for the object with the given offset.
	FindHash(o int64) (plumbing.Hash, error)
	// Count returns the number of entries in the index.
	Count() (int64, error)
	// Entries returns an iterator to retrieve all index entries.
	Entries() (EntryIter, error)
	// EntriesByOffset returns an iterator to retrieve all index entries ordered
	// by offset.
	EntriesByOffset() (EntryIter, error)
}

// MemoryIndex is the in memory representation of an idx file.
type MemoryIndex struct {
	Version uint32
	Fanout  [256]uint32
	// FanoutMapping maps the position in the fanout table to the position
	// in the Names, Offset32 and CRC32 slices. This improves the memory
	// usage by not needing an array with unnecessary empty slots.
	FanoutMapping    [256]int
	Names            [][]byte
	Offset32         [][]byte
	CRC32            [][]byte
	Offset64         []byte
	PackfileChecksum [20]byte
	IdxChecksum      [20]byte

	offsetHash       map[int64]plumbing.Hash
	offsetHashIsFull bool
}

var _ Index = (*MemoryIndex)(nil)

// NewMemoryIndex returns an instance of a new MemoryIndex.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{}
}

func (idx *MemoryIndex) findHashIndex(h plumbing.Hash) (int, bool) {
	k := idx.FanoutMapping[h[0]]
	if k == noMapping {
		return 0, false
	}

	if len(idx.Names) <= k {
		return 0, false
	}

	data := idx.Names[k]
	high := uint64(len(idx.Offset32[k])) >> 2
	if high == 0 {
		return 0, false
	}

	low := uint64(0)
	for {
		mid := (low + high) >> 1
		offset := mid * objectIDLength

		cmp := bytes.Compare(h[:], data[offset:offset+objectIDLength])
		if cmp < 0 {
			high = mid
		} else if cmp == 0 {
			return int(mid), true
		} else {
			low = mid + 1
		}

		if low >= high {
			break
		}
	}

	return 0, false
}

// Contains implements the Index interface.
func (idx *MemoryIndex) Contains(h plumbing.Hash) (bool, error) {
	_, ok := idx.findHashIndex(h)
	return ok, nil
}

// FindOffset implements the Index interface.
func (idx *MemoryIndex) FindOffset(h plumbing.Hash) (int64, error) {
	if len(idx.FanoutMapping) <= int(h[0]) {
		return 0, plumbing.ErrObjectNotFound
	}

	k := idx.FanoutMapping[h[0]]
	i, ok := idx.findHashIndex(h)
	if !ok {
		return 0, plumbing.ErrObjectNotFound
	}

	offset := idx.getOffset(k, i)

	if !idx.offsetHashIsFull {
		// Save the offset for reverse lookup
		if idx.offsetHash == nil {
			idx.offsetHash = make(map[int64]plumbing.Hash)
		}
		idx.offsetHash[int64(offset)] = h
	}

	return int64(offset), nil
}

const isO64Mask = uint64(1) << 31

func (idx *MemoryIndex) getOffset(firstLevel, secondLevel int) uint64 {
	offset := secondLevel << 2
	ofs := encbin.BigEndian.Uint32(idx.Offset32[firstLevel][offset : offset+4])

	if (uint64(ofs) & isO64Mask) != 0 {
		offset := 8 * (uint64(ofs) & ^isO64Mask)
		n := encbin.BigEndian.Uint64(idx.Offset64[offset : offset+8])
		return n
	}

	return uint64(ofs)
}

// FindCRC32 implements the Index interface.
func (idx *MemoryIndex) FindCRC32(h plumbing.Hash) (uint32, error) {
	k := idx.FanoutMapping[h[0]]
	i, ok := idx.findHashIndex(h)
	if !ok {
		return 0, plumbing.ErrObjectNotFound
	}

	return idx.getCRC32(k, i), nil
}

func (idx *MemoryIndex) getCRC32(firstLevel, secondLevel int) uint32 {
	offset := secondLevel << 2
	return encbin.BigEndian.Uint32(idx.CRC32[firstLevel][offset : offset+4])
}

// FindHash implements the Index interface.
func (idx *MemoryIndex) FindHash(o int64) (plumbing.Hash, error) {
	var hash plumbing.Hash
	var ok bool

	if idx.offsetHash != nil {
		if hash, ok = idx.offsetHash[o]; ok {
			return hash, nil
		}
	}

	// Lazily generate the reverse offset/hash map if required.
	if !idx.offsetHashIsFull || idx.offsetHash == nil {
		if err := idx.genOffsetHash(); err != nil {
			return plumbing.ZeroHash, err
		}

		hash, ok = idx.offsetHash[o]
	}

	if !ok {
		return plumbing.ZeroHash, plumbing.ErrObjectNotFound
	}

	return hash, nil
}

// genOffsetHash generates the offset/hash mapping for reverse search.
func (idx *MemoryIndex) genOffsetHash() error {
	count, err := idx.Count()
	if err != nil {
		return err
	}

	idx.offsetHash = make(map[int64]plumbing.Hash, count)
	idx.offsetHashIsFull = true

	var hash plumbing.Hash
	i := uint32(0)
	for firstLevel, fanoutValue := range idx.Fanout {
		mappedFirstLevel := idx.FanoutMapping[firstLevel]
		for secondLevel := uint32(0); i < fanoutValue; i++ {
			copy(hash[:], idx.Names[mappedFirstLevel][secondLevel*objectIDLength:])
			offset := int64(idx.getOffset(mappedFirstLevel, int(secondLevel)))
			idx.offsetHash[offset] = hash
			secondLevel++
		}
	}

	return nil
}

// Count implements the Index interface.
func (idx *MemoryIndex) Count() (int64, error) {
	return int64(idx.Fanout[fanout-1]), nil
}

// Entries implements the Index interface.
func (idx *MemoryIndex) Entries() (EntryIter, error) {
	return &idxfileEntryIter{idx, 0, 0, 0}, nil
}

// EntriesByOffset implements the Index interface.
func (idx *MemoryIndex) EntriesByOffset() (EntryIter, error) {
	count, err := idx.Count()
	if err != nil {
		return nil, err
	}

	iter := &idxfileEntryOffsetIter{
		entries: make(entriesByOffset, count),
	}

	entries, err := idx.Entries()
	if err != nil {
		return nil, err
	}

	for pos := 0; int64(pos) < count; pos++ {
		entry, err := entries.Next()
		if err != nil {
			return nil, err
		}

		iter.entries[pos] = entry
	}

	sort.Sort(iter.entries)

	return iter, nil
}

// EntryIter is an iterator that will return the entries in a packfile index.
type EntryIter interface {
	// Next returns the next entry in the packfile index.
	Next() (*Entry, error)
	// Close closes the iterator.
	Close() error
}

type idxfileEntryIter struct {
	idx                     *MemoryIndex
	total                   int
	firstLevel, secondLevel int
}

func (i *idxfileEntryIter) Next() (*Entry, error) {
	for {
		if i.firstLevel >= fanout {
			return nil, io.EOF
		}

		if i.total >= int(i.idx.Fanout[i.firstLevel]) {
			i.firstLevel++
			i.secondLevel = 0
			continue
		}

		mappedFirstLevel := i.idx.FanoutMapping[i.firstLevel]
		entry := new(Entry)
		copy(entry.Hash[:], i.idx.Names[mappedFirstLevel][i.secondLevel*objectIDLength:])
		entry.Offset = i.idx.getOffset(mappedFirstLevel, i.secondLevel)
		entry.CRC32 = i.idx.getCRC32(mappedFirstLevel, i.secondLevel)

		i.secondLevel++
		i.total++

		return entry, nil
	}
}

func (i *idxfileEntryIter) Close() error {
	i.firstLevel = fanout
	return nil
}

// Entry is the in memory representation of an object entry in the idx file.
type Entry struct {
	Hash   plumbing.Hash
	CRC32  uint32
	Offset uint64
}

type idxfileEntryOffsetIter struct {
	entries entriesByOffset
	pos     int
}

func (i *idxfileEntryOffsetIter) Next() (*Entry, error) {
	if i.pos >= len(i.entries) {
		return nil, io.EOF
	}

	entry := i.entries[i.pos]
	i.pos++

	return entry, nil
}

func (i *idxfileEntryOffsetIter) Close() error {
	i.pos = len(i.entries) + 1
	return nil
}

type entriesByOffset []*Entry

func (o entriesByOffset) Len() int {
	return len(o)
}

func (o entriesByOffset) Less(i int, j int) bool {
	return o[i].Offset < o[j].Offset
}

func (o entriesByOffset) Swap(i int, j int) {
	o[i], o[j] = o[j], o[i]
}
