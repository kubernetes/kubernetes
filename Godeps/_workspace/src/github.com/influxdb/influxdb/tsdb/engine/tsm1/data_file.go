package tsm1

/*
A TSM file is composed for four sections: header, blocks, index and the footer.

┌────────┬────────────────────────────────────┬─────────────┬──────────────┐
│ Header │               Blocks               │    Index    │    Footer    │
│5 bytes │              N bytes               │   N bytes   │   4 bytes    │
└────────┴────────────────────────────────────┴─────────────┴──────────────┘

Header is composed of a magic number to identify the file type and a version
number.

┌───────────────────┐
│      Header       │
├─────────┬─────────┤
│  Magic  │ Version │
│ 4 bytes │ 1 byte  │
└─────────┴─────────┘

Blocks are sequences of pairs of CRC32 and data.  The block data is opaque to the
file.  The CRC32 is used for block level error detection.  The length of the blocks
is stored in the index.

┌───────────────────────────────────────────────────────────┐
│                          Blocks                           │
├───────────────────┬───────────────────┬───────────────────┤
│      Block 1      │      Block 2      │      Block N      │
├─────────┬─────────┼─────────┬─────────┼─────────┬─────────┤
│  CRC    │  Data   │  CRC    │  Data   │  CRC    │  Data   │
│ 4 bytes │ N bytes │ 4 bytes │ N bytes │ 4 bytes │ N bytes │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Following the blocks is the index for the blocks in the file.  The index is
composed of a sequence of index entries ordered lexicographically by key and
then by time.  Each index entry starts with a key length and key followed by a
count of the number of blocks in the file.  Each block entry is composed of
the min and max time for the block, the offset into the file where the block
is located and the the size of the block.

The index structure can provide efficient access to all blocks as well as the
ability to determine the cost associated with acessing a given key.  Given a key
and timestamp, we can determine whether a file contains the block for that
timestamp as well as where that block resides and how much data to read to
retrieve the block.  If we know we need to read all or multiple blocks in a
file, we can use the size to determine how much to read in a given IO.

┌────────────────────────────────────────────────────────────────────────────┐
│                                   Index                                    │
├─────────┬─────────┬──────┬───────┬─────────┬─────────┬────────┬────────┬───┤
│ Key Len │   Key   │ Type │ Count │Min Time │Max Time │ Offset │  Size  │...│
│ 2 bytes │ N bytes │1 byte│2 bytes│ 8 bytes │ 8 bytes │8 bytes │4 bytes │   │
└─────────┴─────────┴──────┴───────┴─────────┴─────────┴────────┴────────┴───┘

The last section is the footer that stores the offset of the start of the index.

┌─────────┐
│ Footer  │
├─────────┤
│Index Ofs│
│ 8 bytes │
└─────────┘
*/

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

const (
	// MagicNumber is written as the first 4 bytes of a data file to
	// identify the file as a tsm1 formatted file
	MagicNumber uint32 = 0x16D116D1

	Version byte = 1

	// Size in bytes of an index entry
	indexEntrySize = 28

	// Size in bytes used to store the count of index entries for a key
	indexCountSize = 2

	// Size in bytes used to store the type of block encoded
	indexTypeSize = 1

	// Max number of blocks for a given key that can exist in a single file
	maxIndexEntries = (1 << (indexCountSize * 8)) - 1
)

var (
	ErrNoValues  = fmt.Errorf("no values written")
	ErrTSMClosed = fmt.Errorf("tsm file closed")
)

// TSMWriter writes TSM formatted key and values.
type TSMWriter interface {
	// Write writes a new block for key containing and values.  Writes append
	// blocks in the order that the Write function is called.  The caller is
	// responsible for ensuring keys and blocks or sorted appropriately.
	// Values are encoded as a full block.  The caller is responsible for
	// ensuring a fixed number of values are encoded in each block as wells as
	// ensuring the Values are sorted. The first and last timestamp values are
	// used as the minimum and maximum values for the index entry.
	Write(key string, values Values) error

	// WriteIndex finishes the TSM write streams and writes the index.
	WriteIndex() error

	// Closes any underlying file resources.
	Close() error

	// Size returns the current size in bytes of the file
	Size() uint32
}

// TSMIndex represent the index section of a TSM file.  The index records all
// blocks, their locations, sizes, min and max times.
type TSMIndex interface {

	// Add records a new block entry for a key in the index.
	Add(key string, blockType byte, minTime, maxTime time.Time, offset int64, size uint32)

	// Delete removes the given keys from the index.
	Delete(keys []string)

	// Contains return true if the given key exists in the index.
	Contains(key string) bool

	// ContainsValue returns true if key and time might exists in this file.  This function could
	// return true even though the actual point does not exists.  For example, the key may
	// exists in this file, but not have point exactly at time t.
	ContainsValue(key string, timestamp time.Time) bool

	// Entries returns all index entries for a key.
	Entries(key string) []*IndexEntry

	// Entry returns the index entry for the specified key and timestamp.  If no entry
	// matches the key and timestamp, nil is returned.
	Entry(key string, timestamp time.Time) *IndexEntry

	// Keys returns the unique set of keys in the index.
	Keys() []string

	// Key returns the key in the index at the given postion.
	Key(index int) (string, []*IndexEntry)

	// KeyCount returns the count of unique keys in the index.
	KeyCount() int

	// Size returns the size of a the current index in bytes
	Size() uint32

	// TimeRange returns the min and max time across all keys in the file.
	TimeRange() (time.Time, time.Time)

	// KeyRange returns the min and max keys in the file.
	KeyRange() (string, string)

	// Type returns the block type of the values stored for the key.  Returns one of
	// BlockFloat64, BlockInt64, BlockBool, BlockString.  If key does not exist,
	// an error is returned.
	Type(key string) (byte, error)

	// MarshalBinary returns a byte slice encoded version of the index.
	MarshalBinary() ([]byte, error)

	// UnmarshalBinary populates an index from an encoded byte slice
	// representation of an index.
	UnmarshalBinary(b []byte) error
}

// IndexEntry is the index information for a given block in a TSM file.
type IndexEntry struct {

	// The min and max time of all points stored in the block.
	MinTime, MaxTime time.Time

	// The absolute position in the file where this block is located.
	Offset int64

	// The size in bytes of the block in the file.
	Size uint32
}

func (e *IndexEntry) UnmarshalBinary(b []byte) error {
	if len(b) != indexEntrySize {
		return fmt.Errorf("unmarshalBinary: short buf: %v != %v", indexEntrySize, len(b))
	}
	e.MinTime = time.Unix(0, int64(btou64(b[:8])))
	e.MaxTime = time.Unix(0, int64(btou64(b[8:16])))
	e.Offset = int64(btou64(b[16:24]))
	e.Size = btou32(b[24:28])
	return nil
}

// Returns true if this IndexEntry may contain values for the given time.  The min and max
// times are inclusive.
func (e *IndexEntry) Contains(t time.Time) bool {
	return (e.MinTime.Equal(t) || e.MinTime.Before(t)) &&
		(e.MaxTime.Equal(t) || e.MaxTime.After(t))
}

func (e *IndexEntry) OverlapsTimeRange(min, max time.Time) bool {
	return (e.MinTime.Equal(max) || e.MinTime.Before(max)) &&
		(e.MaxTime.Equal(min) || e.MaxTime.After(min))
}

func (e *IndexEntry) String() string {
	return fmt.Sprintf("min=%s max=%s ofs=%d siz=%d", e.MinTime.UTC(), e.MaxTime.UTC(), e.Offset, e.Size)
}

func NewDirectIndex() TSMIndex {
	return &directIndex{
		blocks: map[string]*indexEntries{},
	}
}

// directIndex is a simple in-memory index implementation for a TSM file.  The full index
// must fit in memory.
type directIndex struct {
	mu sync.RWMutex

	blocks map[string]*indexEntries
}

func (d *directIndex) Add(key string, blockType byte, minTime, maxTime time.Time, offset int64, size uint32) {
	d.mu.Lock()
	defer d.mu.Unlock()

	entries := d.blocks[key]
	if entries == nil {
		entries = &indexEntries{
			Type: blockType,
		}
		d.blocks[key] = entries
	}
	entries.Append(&IndexEntry{
		MinTime: minTime,
		MaxTime: maxTime,
		Offset:  offset,
		Size:    size,
	})
}

func (d *directIndex) Entries(key string) []*IndexEntry {
	d.mu.RLock()
	defer d.mu.RUnlock()

	entries := d.blocks[key]
	if entries == nil {
		return nil
	}
	return d.blocks[key].entries
}

func (d *directIndex) Entry(key string, t time.Time) *IndexEntry {
	d.mu.RLock()
	defer d.mu.RUnlock()

	entries := d.Entries(key)
	for _, entry := range entries {
		if entry.Contains(t) {
			return entry
		}
	}
	return nil
}

func (d *directIndex) Type(key string) (byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	entries := d.blocks[key]
	if entries != nil {
		return entries.Type, nil
	}
	return 0, fmt.Errorf("key does not exist: %v", key)
}

func (d *directIndex) Contains(key string) bool {
	return len(d.Entries(key)) > 0
}

func (d *directIndex) ContainsValue(key string, t time.Time) bool {
	return d.Entry(key, t) != nil
}

func (d *directIndex) Delete(keys []string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	for _, k := range keys {
		delete(d.blocks, k)
	}
}

func (d *directIndex) Keys() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var keys []string
	for k := range d.blocks {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (d *directIndex) Key(idx int) (string, []*IndexEntry) {
	if idx < 0 || idx >= len(d.blocks) {
		return "", nil
	}
	k := d.Keys()[idx]
	return k, d.blocks[k].entries
}

func (d *directIndex) KeyCount() int {
	return len(d.Keys())
}

func (d *directIndex) KeyRange() (string, string) {
	var min, max string
	for k := range d.blocks {
		if min == "" || k < min {
			min = k
		}
		if max == "" || k > max {
			max = k
		}

	}
	return min, max
}

func (d *directIndex) TimeRange() (time.Time, time.Time) {
	min, max := time.Unix(0, math.MaxInt64), time.Unix(0, math.MinInt64)
	for _, entries := range d.blocks {
		for _, e := range entries.entries {
			if e.MinTime.Before(min) {
				min = e.MinTime
			}
			if e.MaxTime.After(max) {
				max = e.MaxTime
			}
		}
	}
	return min, max
}

func (d *directIndex) addEntries(key string, entries *indexEntries) {
	existing := d.blocks[key]
	if existing == nil {
		d.blocks[key] = entries
		return
	}
	existing.Append(entries.entries...)
}

func (d *directIndex) Write(w io.Writer) error {
	b, err := d.MarshalBinary()
	if err != nil {
		return fmt.Errorf("write: marshal error: %v", err)
	}

	// Write out the index bytes
	_, err = w.Write(b)
	if err != nil {
		return fmt.Errorf("write: writer error: %v", err)
	}
	return nil
}

func (d *directIndex) MarshalBinary() ([]byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	// Index blocks are writtens sorted by key
	var keys []string
	for k := range d.blocks {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Buffer to build up the index and write in bulk
	var b []byte

	// For each key, individual entries are sorted by time
	for _, key := range keys {
		entries := d.blocks[key]

		if entries.Len() > maxIndexEntries {
			return nil, fmt.Errorf("key '%s' exceeds max index entries: %d > %d",
				key, entries.Len(), maxIndexEntries)
		}
		sort.Sort(entries)

		// Append the key length and key
		b = append(b, u16tob(uint16(len(key)))...)
		b = append(b, key...)

		// Append the block type
		b = append(b, entries.Type)

		// Append the index block count
		b = append(b, u16tob(uint16(entries.Len()))...)

		// Append each index entry for all blocks for this key
		eb, err := entries.MarshalBinary()
		if err != nil {
			return nil, err
		}
		b = append(b, eb...)
	}
	return b, nil
}

func (d *directIndex) UnmarshalBinary(b []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	var pos int
	for pos < len(b) {
		n, key, err := readKey(b[pos:])
		if err != nil {
			return fmt.Errorf("readIndex: read key error: %v", err)
		}
		pos += n

		n, entries, err := readEntries(b[pos:])
		if err != nil {
			return fmt.Errorf("readIndex: read entries error: %v", err)
		}

		pos += n
		d.addEntries(string(key), entries)
	}
	return nil
}

func (d *directIndex) Size() uint32 {
	return 0
}

// indirectIndex is a TSMIndex that uses a raw byte slice representation of an index.  This
// implementation can be used for indexes that may be MMAPed into memory.
type indirectIndex struct {
	mu sync.RWMutex

	// indirectIndex works a follows.  Assuming we have an index structure in memory as
	// the diagram below:
	//
	// ┌────────────────────────────────────────────────────────────────────┐
	// │                               Index                                │
	// ├─┬──────────────────────┬──┬───────────────────────┬───┬────────────┘
	// │0│                      │62│                       │145│
	// ├─┴───────┬─────────┬────┼──┴──────┬─────────┬──────┼───┴─────┬──────┐
	// │Key 1 Len│   Key   │... │Key 2 Len│  Key 2  │ ...  │  Key 3  │ ...  │
	// │ 2 bytes │ N bytes │    │ 2 bytes │ N bytes │      │ 2 bytes │      │
	// └─────────┴─────────┴────┴─────────┴─────────┴──────┴─────────┴──────┘

	// We would build an `offsets` slices where each element pointers to the byte location
	// for the first key in the index slice.

	// ┌────────────────────────────────────────────────────────────────────┐
	// │                              Offsets                               │
	// ├────┬────┬────┬─────────────────────────────────────────────────────┘
	// │ 0  │ 62 │145 │
	// └────┴────┴────┘

	// Using this offset slice we can find `Key 2` by doing a binary search
	// over the offsets slice.  Instead of comparing the value in the offsets
	// (e.g. `62`), we use that as an index into the underlying index to
	// retrieve the key at postion `62` and perform our comparisons with that.

	// When we have identified the correct position in the index for a given
	// key, we could perform another binary search or a linear scan.  This
	// should be fast as well since each index entry is 28 bytes and all
	// contiguous in memory.  The current implementation uses a linear scan since the
	// number of block entries is expected to be < 100 per key.

	// b is the underlying index byte slice.  This could be a copy on the heap or an MMAP
	// slice reference
	b []byte

	// offsets contains the positions in b for each key.  It points to the 2 byte length of
	// key.
	offsets []int32

	// minKey, maxKey are the minium and maximum (lexicographically sorted) contained in the
	// file
	minKey, maxKey string

	// minTime, maxTime are the minimum and maximum times contained in the file across all
	// series.
	minTime, maxTime time.Time
}

func NewIndirectIndex() TSMIndex {
	return &indirectIndex{}
}

// Add records a new block entry for a key in the index.
func (d *indirectIndex) Add(key string, blockType byte, minTime, maxTime time.Time, offset int64, size uint32) {
	panic("unsupported operation")
}

// search returns the index of i in offsets for where key is located.  If key is not
// in the index, len(index) is returned.
func (d *indirectIndex) search(key []byte) int {
	// We use a binary search across our indirect offsets (pointers to all the keys
	// in the index slice).
	i := sort.Search(len(d.offsets), func(i int) bool {
		// i is the position in offsets we are at so get offset it points to
		offset := d.offsets[i]

		// It's pointing to the start of the key which is a 2 byte length
		keyLen := int32(btou16(d.b[offset : offset+2]))

		// See if it matches
		return bytes.Compare(d.b[offset+2:offset+2+keyLen], key) >= 0
	})

	// See if we might have found the right index
	if i < len(d.offsets) {
		ofs := d.offsets[i]
		_, k, err := readKey(d.b[ofs:])
		if err != nil {
			panic(fmt.Sprintf("error reading key: %v", err))
		}

		// The search may have returned an i == 0 which could indicated that the value
		// searched should be inserted at postion 0.  Make sure the key in the index
		// matches the search value.
		if !bytes.Equal(key, k) {
			return len(d.b)
		}

		return int(ofs)
	}

	// The key is not in the index.  i is the index where it would be inserted so return
	// a value outside our offset range.
	return len(d.b)
}

// Entries returns all index entries for a key.
func (d *indirectIndex) Entries(key string) []*IndexEntry {
	d.mu.RLock()
	defer d.mu.RUnlock()

	kb := []byte(key)

	ofs := d.search(kb)
	if ofs < len(d.b) {
		n, k, err := readKey(d.b[ofs:])
		if err != nil {
			panic(fmt.Sprintf("error reading key: %v", err))
		}

		// The search may have returned an i == 0 which could indicated that the value
		// searched should be inserted at postion 0.  Make sure the key in the index
		// matches the search value.
		if !bytes.Equal(kb, k) {
			return nil
		}

		// Read and return all the entries
		ofs += n
		_, entries, err := readEntries(d.b[ofs:])
		if err != nil {
			panic(fmt.Sprintf("error reading entries: %v", err))

		}
		return entries.entries
	}

	// The key is not in the index.  i is the index where it would be inserted.
	return nil
}

// Entry returns the index entry for the specified key and timestamp.  If no entry
// matches the key an timestamp, nil is returned.
func (d *indirectIndex) Entry(key string, timestamp time.Time) *IndexEntry {
	entries := d.Entries(key)
	for _, entry := range entries {
		if entry.Contains(timestamp) {
			return entry
		}
	}
	return nil
}

func (d *indirectIndex) Keys() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var keys []string
	for _, offset := range d.offsets {
		_, key, _ := readKey(d.b[offset:])
		keys = append(keys, string(key))
	}
	return keys
}

func (d *indirectIndex) Key(idx int) (string, []*IndexEntry) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if idx < 0 || idx >= len(d.offsets) {
		return "", nil
	}
	n, key, _ := readKey(d.b[d.offsets[idx]:])
	_, entries, _ := readEntries(d.b[int(d.offsets[idx])+n:])
	return string(key), entries.entries
}

func (d *indirectIndex) KeyCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return len(d.offsets)
}

func (d *indirectIndex) Delete(keys []string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	lookup := map[string]struct{}{}
	for _, k := range keys {
		lookup[k] = struct{}{}
	}

	var offsets []int32
	for _, offset := range d.offsets {
		_, indexKey, _ := readKey(d.b[offset:])

		if _, ok := lookup[string(indexKey)]; ok {
			continue
		}
		offsets = append(offsets, int32(offset))
	}
	d.offsets = offsets
}

func (d *indirectIndex) Contains(key string) bool {
	return len(d.Entries(key)) > 0
}

func (d *indirectIndex) ContainsValue(key string, timestamp time.Time) bool {
	return d.Entry(key, timestamp) != nil
}

func (d *indirectIndex) Type(key string) (byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	kb := []byte(key)
	ofs := d.search(kb)
	if ofs < len(d.b) {
		n, _, err := readKey(d.b[ofs:])
		if err != nil {
			panic(fmt.Sprintf("error reading key: %v", err))
		}

		ofs += n
		return d.b[ofs], nil
	}
	return 0, fmt.Errorf("key does not exist: %v", key)
}

func (d *indirectIndex) KeyRange() (string, string) {
	return d.minKey, d.maxKey
}

func (d *indirectIndex) TimeRange() (time.Time, time.Time) {
	return d.minTime, d.maxTime
}

// MarshalBinary returns a byte slice encoded version of the index.
func (d *indirectIndex) MarshalBinary() ([]byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return d.b, nil
}

// UnmarshalBinary populates an index from an encoded byte slice
// representation of an index.
func (d *indirectIndex) UnmarshalBinary(b []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Keep a reference to the actual index bytes
	d.b = b

	// To create our "indirect" index, we need to find he location of all the keys in
	// the raw byte slice.  The keys are listed once each (in sorted order).  Following
	// each key is a time ordered list of index entry blocks for that key.  The loop below
	// basically skips across the slice keeping track of the counter when we are at a key
	// field.
	var i int32
	for i < int32(len(b)) {
		d.offsets = append(d.offsets, i)

		_, kb, err := readKey(b[i:])
		if err != nil {
			return err
		}
		key := string(kb)

		if d.minKey == "" || key < d.minKey {
			d.minKey = key
		}

		if d.maxKey == "" || key > d.maxKey {
			d.maxKey = key
		}

		keyLen := int32(btou16(b[i : i+2]))
		// Skip to the start of the key
		i += 2

		// Skip over the key
		i += keyLen

		n, entries, err := readEntries(d.b[i:])

		minTime := entries.entries[0].MinTime
		if d.minTime.IsZero() || minTime.Before(d.minTime) {
			d.minTime = minTime
		}

		maxTime := entries.entries[len(entries.entries)-1].MaxTime
		if d.maxTime.IsZero() || maxTime.After(d.maxTime) {
			d.maxTime = maxTime
		}

		i += int32(n)
	}

	return nil
}

func (d *indirectIndex) Size() uint32 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return uint32(len(d.b))
}

// tsmWriter writes keys and values in the TSM format
type tsmWriter struct {
	w     io.Writer
	index TSMIndex
	n     int64
}

func NewTSMWriter(w io.Writer) (TSMWriter, error) {
	index := &directIndex{
		blocks: map[string]*indexEntries{},
	}

	return &tsmWriter{w: w, index: index}, nil
}

func (t *tsmWriter) Write(key string, values Values) error {
	// Write header only after we have some data to write.
	if t.n == 0 {
		n, err := t.w.Write(append(u32tob(MagicNumber), Version))
		if err != nil {
			return err
		}
		t.n = int64(n)
	}

	block, err := values.Encode(nil)
	if err != nil {
		return err
	}

	checksum := crc32.ChecksumIEEE(block)

	n, err := t.w.Write(append(u32tob(checksum), block...))
	if err != nil {
		return err
	}

	blockType, err := BlockType(block)
	if err != nil {
		return err
	}
	// Record this block in index
	t.index.Add(key, blockType, values[0].Time(), values[len(values)-1].Time(), t.n, uint32(n))

	// Increment file position pointer
	t.n += int64(n)
	return nil
}

// WriteIndex writes the index section of the file.  If there are no index entries to write,
// this returns ErrNoValues
func (t *tsmWriter) WriteIndex() error {
	indexPos := t.n

	// Generate the index bytes
	b, err := t.index.MarshalBinary()
	if err != nil {
		return err
	}

	// Don't write an index if we don't actually have any blocks in the file.
	if len(b) == 0 {
		return ErrNoValues
	}

	// Write the index followed by index position
	_, err = t.w.Write(append(b, u64tob(uint64(indexPos))...))
	if err != nil {
		return err
	}

	return nil
}

func (t *tsmWriter) Close() error {
	if c, ok := t.w.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func (t *tsmWriter) Size() uint32 {
	return uint32(t.n) + t.index.Size()
}

type TSMReader struct {
	mu sync.RWMutex

	// accessor provides access and decoding of blocks for the reader
	accessor blockAccessor

	// index is the index of all blocks.
	index TSMIndex

	// tombstoner ensures tombstoned keys are not available by the index.
	tombstoner *Tombstoner

	// size is the size of the file on disk.
	size int64

	// lastModified is the last time this file was modified on disk
	lastModified time.Time
}

// blockAccessor abstracts a method of accessing blocks from a
// TSM file.
type blockAccessor interface {
	init() (TSMIndex, error)
	read(key string, timestamp time.Time) ([]Value, error)
	readAll(key string) ([]Value, error)
	readBlock(entry *IndexEntry, values []Value) ([]Value, error)
	path() string
	close() error
}

type TSMReaderOptions struct {
	// Reader is used to create file IO based reader.
	Reader io.ReadSeeker

	// MMAPFile is used to create an MMAP based reader.
	MMAPFile *os.File
}

func NewTSMReader(r io.ReadSeeker) (*TSMReader, error) {
	return NewTSMReaderWithOptions(
		TSMReaderOptions{
			Reader: r,
		})
}

func NewTSMReaderWithOptions(opt TSMReaderOptions) (*TSMReader, error) {
	t := &TSMReader{}
	if opt.Reader != nil {
		// Seek to the end of the file to determine the size
		size, err := opt.Reader.Seek(2, 0)
		if err != nil {
			return nil, err
		}
		t.size = size
		if f, ok := opt.Reader.(*os.File); ok {
			stat, err := f.Stat()
			if err != nil {
				return nil, err
			}

			t.lastModified = stat.ModTime()
		}
		t.accessor = &fileAccessor{
			r: opt.Reader,
		}

	} else if opt.MMAPFile != nil {
		stat, err := opt.MMAPFile.Stat()
		if err != nil {
			return nil, err
		}
		t.size = stat.Size()
		t.lastModified = stat.ModTime()
		t.accessor = &mmapAccessor{
			f: opt.MMAPFile,
		}
	} else {
		panic("invalid options: need Reader or MMAPFile")
	}

	index, err := t.accessor.init()
	if err != nil {
		return nil, err
	}

	t.index = index
	t.tombstoner = &Tombstoner{Path: t.Path()}

	if err := t.applyTombstones(); err != nil {
		return nil, err
	}

	return t, nil
}

func (t *TSMReader) applyTombstones() error {
	// Read any tombstone entries if the exist
	tombstones, err := t.tombstoner.ReadAll()
	if err != nil {
		return fmt.Errorf("init: read tombstones: %v", err)
	}

	// Update our index
	t.index.Delete(tombstones)
	return nil
}

func (t *TSMReader) Path() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.accessor.path()
}

func (t *TSMReader) Keys() []string {
	return t.index.Keys()
}

func (t *TSMReader) Key(index int) (string, []*IndexEntry) {
	return t.index.Key(index)
}

func (t *TSMReader) ReadAt(entry *IndexEntry, vals []Value) ([]Value, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.accessor.readBlock(entry, vals)
}

func (t *TSMReader) Read(key string, timestamp time.Time) ([]Value, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.accessor.read(key, timestamp)
}

// ReadAll returns all values for a key in all blocks.
func (t *TSMReader) ReadAll(key string) ([]Value, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.accessor.readAll(key)
}

func (t *TSMReader) Type(key string) (byte, error) {
	return t.index.Type(key)
}

func (t *TSMReader) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.accessor.close()
}

// Remove removes any underlying files stored on disk for this reader.
func (t *TSMReader) Remove() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	path := t.accessor.path()
	if path != "" {
		os.RemoveAll(path)
	}

	if err := t.tombstoner.Delete(); err != nil {
		return err
	}
	return nil
}

func (t *TSMReader) Contains(key string) bool {
	return t.index.Contains(key)
}

// ContainsValue returns true if key and time might exists in this file.  This function could
// return true even though the actual point does not exists.  For example, the key may
// exists in this file, but not have point exactly at time t.
func (t *TSMReader) ContainsValue(key string, ts time.Time) bool {
	return t.index.ContainsValue(key, ts)
}

func (t *TSMReader) Delete(keys []string) error {
	if err := t.tombstoner.Add(keys); err != nil {
		return err
	}

	t.index.Delete(keys)
	return nil
}

// TimeRange returns the min and max time across all keys in the file.
func (t *TSMReader) TimeRange() (time.Time, time.Time) {
	return t.index.TimeRange()
}

// KeyRange returns the min and max key across all keys in the file.
func (t *TSMReader) KeyRange() (string, string) {
	return t.index.KeyRange()
}

func (t *TSMReader) Entries(key string) []*IndexEntry {
	return t.index.Entries(key)
}

func (t *TSMReader) IndexSize() uint32 {
	return t.index.Size()
}

func (t *TSMReader) Size() uint32 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return uint32(t.size)
}

func (t *TSMReader) LastModified() time.Time {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.lastModified
}

// HasTombstones return true if there are any tombstone entries recorded.
func (t *TSMReader) HasTombstones() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.tombstoner.HasTombstones()
}

func (t *TSMReader) Stats() FileStat {
	minTime, maxTime := t.index.TimeRange()
	minKey, maxKey := t.index.KeyRange()
	return FileStat{
		Path:         t.Path(),
		Size:         t.Size(),
		LastModified: t.LastModified(),
		MinTime:      minTime,
		MaxTime:      maxTime,
		MinKey:       minKey,
		MaxKey:       maxKey,
		HasTombstone: t.tombstoner.HasTombstones(),
	}
}

// fileAccessor is file IO based block accessor.  It provides access to blocks
// using a file IO based approach (seek, read, etc.)
type fileAccessor struct {
	mu    sync.Mutex
	r     io.ReadSeeker
	index TSMIndex
}

func (f *fileAccessor) init() (TSMIndex, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Verify it's a TSM file of the right version
	if err := verifyVersion(f.r); err != nil {
		return nil, err
	}

	// Current the readers size
	size, err := f.r.Seek(0, os.SEEK_END)
	if err != nil {
		return nil, fmt.Errorf("init: failed to seek: %v", err)
	}

	indexEnd := size - 8

	// Seek to index location pointer
	_, err = f.r.Seek(-8, os.SEEK_END)
	if err != nil {
		return nil, fmt.Errorf("init: failed to seek to index ptr: %v", err)
	}

	// Read the absolute position of the start of the index
	b := make([]byte, 8)
	_, err = f.r.Read(b)
	if err != nil {
		return nil, fmt.Errorf("init: failed to read index ptr: %v", err)

	}

	indexStart := int64(btou64(b))

	_, err = f.r.Seek(indexStart, os.SEEK_SET)
	if err != nil {
		return nil, fmt.Errorf("init: failed to seek to index: %v", err)
	}

	b = make([]byte, indexEnd-indexStart)
	f.index = &directIndex{
		blocks: map[string]*indexEntries{},
	}
	_, err = f.r.Read(b)
	if err != nil {
		return nil, fmt.Errorf("init: read index: %v", err)
	}

	if err := f.index.UnmarshalBinary(b); err != nil {
		return nil, fmt.Errorf("init: unmarshal error: %v", err)
	}

	return f.index, nil
}

func (f *fileAccessor) read(key string, timestamp time.Time) ([]Value, error) {
	entry := f.index.Entry(key, timestamp)

	if entry == nil {
		return nil, nil
	}

	return f.readBlock(entry, nil)
}

func (f *fileAccessor) readBlock(entry *IndexEntry, values []Value) ([]Value, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// TODO: remove this allocation
	b := make([]byte, 16*1024)
	_, err := f.r.Seek(entry.Offset, os.SEEK_SET)
	if err != nil {
		return nil, err
	}

	if int(entry.Size) > len(b) {
		b = make([]byte, entry.Size)
	}

	n, err := f.r.Read(b)
	if err != nil {
		return nil, err
	}

	//TODO: Validate checksum
	values, err = DecodeBlock(b[4:n], values)
	if err != nil {
		return nil, err
	}

	return values, nil
}

// ReadAll returns all values for a key in all blocks.
func (f *fileAccessor) readAll(key string) ([]Value, error) {
	var values []Value
	blocks := f.index.Entries(key)
	if len(blocks) == 0 {
		return values, nil
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	var temp []Value
	// TODO: we can determine the max block size when loading the file create/re-use
	// a reader level buf then.
	b := make([]byte, 16*1024)
	var pos int64
	for _, block := range blocks {
		// Skip the seek call if we are already at the position we're seeking to
		if pos != block.Offset {
			_, err := f.r.Seek(block.Offset, os.SEEK_SET)
			if err != nil {
				return nil, err
			}
			pos = block.Offset
		}

		if int(block.Size) > len(b) {
			b = make([]byte, block.Size)
		}

		n, err := f.r.Read(b[:block.Size])
		if err != nil {
			return nil, err
		}
		pos += int64(block.Size)

		//TODO: Validate checksum
		temp = temp[:0]
		temp, err = DecodeBlock(b[4:n], temp)
		if err != nil {
			return nil, err
		}
		values = append(values, temp...)
	}

	return values, nil
}

func (f *fileAccessor) path() string {
	f.mu.Lock()
	defer f.mu.Unlock()

	if fd, ok := f.r.(*os.File); ok {
		return fd.Name()
	}
	return ""
}

func (f *fileAccessor) close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if c, ok := f.r.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// mmapAccess is mmap based block accessor.  It access blocks through an
// MMAP file interface.
type mmapAccessor struct {
	mu sync.RWMutex

	f     *os.File
	b     []byte
	index TSMIndex
}

func (m *mmapAccessor) init() (TSMIndex, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := verifyVersion(m.f); err != nil {
		return nil, err
	}

	var err error

	if _, err := m.f.Seek(0, 0); err != nil {
		return nil, err
	}

	stat, err := m.f.Stat()
	if err != nil {
		return nil, err
	}

	m.b, err = mmap(m.f, 0, int(stat.Size()))
	if err != nil {
		return nil, err
	}

	indexOfsPos := len(m.b) - 8
	indexStart := btou64(m.b[indexOfsPos : indexOfsPos+8])

	m.index = NewIndirectIndex()
	if err := m.index.UnmarshalBinary(m.b[indexStart:indexOfsPos]); err != nil {
		return nil, err
	}

	return m.index, nil
}

func (m *mmapAccessor) read(key string, timestamp time.Time) ([]Value, error) {
	entry := m.index.Entry(key, timestamp)
	if entry == nil {
		return nil, nil
	}

	return m.readBlock(entry, nil)
}

func (m *mmapAccessor) readBlock(entry *IndexEntry, values []Value) ([]Value, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		return nil, ErrTSMClosed
	}
	//TODO: Validate checksum
	var err error
	values, err = DecodeBlock(m.b[entry.Offset+4:entry.Offset+int64(entry.Size)], values)
	if err != nil {
		return nil, err
	}

	return values, nil
}

// ReadAll returns all values for a key in all blocks.
func (m *mmapAccessor) readAll(key string) ([]Value, error) {
	blocks := m.index.Entries(key)
	if len(blocks) == 0 {
		return nil, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	var temp []Value
	var err error
	var values []Value
	for _, block := range blocks {
		//TODO: Validate checksum
		temp = temp[:0]
		// The +4 is the 4 byte checksum length
		temp, err = DecodeBlock(m.b[block.Offset+4:block.Offset+int64(block.Size)], temp)
		if err != nil {
			return nil, err
		}
		values = append(values, temp...)
	}

	return values, nil
}

func (m *mmapAccessor) path() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.f.Name()
}

func (m *mmapAccessor) close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.b == nil {
		return nil
	}

	err := munmap(m.b)
	if err != nil {
		return err
	}

	m.b = nil
	return m.f.Close()
}

type indexEntries struct {
	Type    byte
	entries []*IndexEntry
}

func (a *indexEntries) Len() int      { return len(a.entries) }
func (a *indexEntries) Swap(i, j int) { a.entries[i], a.entries[j] = a.entries[j], a.entries[i] }
func (a *indexEntries) Less(i, j int) bool {
	return a.entries[i].MinTime.UnixNano() < a.entries[j].MinTime.UnixNano()
}

func (a *indexEntries) Append(entry ...*IndexEntry) {
	a.entries = append(a.entries, entry...)
}

func (a *indexEntries) MarshalBinary() (b []byte, err error) {
	for _, entry := range a.entries {
		b = append(b, u64tob(uint64(entry.MinTime.UnixNano()))...)
		b = append(b, u64tob(uint64(entry.MaxTime.UnixNano()))...)
		b = append(b, u64tob(uint64(entry.Offset))...)
		b = append(b, u32tob(entry.Size)...)
	}
	return b, nil
}

func readKey(b []byte) (n int, key []byte, err error) {
	// 2 byte size of key
	n, size := 2, int(btou16(b[:2]))

	// N byte key
	key = b[n : n+size]

	n += len(key)
	return
}

func readEntries(b []byte) (n int, entries *indexEntries, err error) {
	// 1 byte block type
	blockType := b[n]
	entries = &indexEntries{
		Type:    blockType,
		entries: []*IndexEntry{},
	}
	n++

	// 2 byte count of index entries
	count := int(btou16(b[n : n+indexCountSize]))
	n += indexCountSize

	for i := 0; i < count; i++ {
		ie := &IndexEntry{}
		if err := ie.UnmarshalBinary(b[i*indexEntrySize+indexCountSize+indexTypeSize : i*indexEntrySize+indexCountSize+indexEntrySize+indexTypeSize]); err != nil {
			return 0, nil, fmt.Errorf("readEntries: unmarshal error: %v", err)
		}
		entries.Append(ie)
		n += indexEntrySize
	}
	return
}

func u16tob(v uint16) []byte {
	b := make([]byte, 2)
	binary.BigEndian.PutUint16(b, v)
	return b
}

func btou16(b []byte) uint16 {
	return uint16(binary.BigEndian.Uint16(b))
}

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

func btou64(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}

func u32tob(v uint32) []byte {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, v)
	return b
}

func btou32(b []byte) uint32 {
	return uint32(binary.BigEndian.Uint32(b))
}

// verifyVersion will verify that the reader's bytes are a TSM byte
// stream of the correct version (1)
func verifyVersion(r io.ReadSeeker) error {
	_, err := r.Seek(0, 0)
	if err != nil {
		return fmt.Errorf("init: failed to seek: %v", err)
	}
	b := make([]byte, 4)
	_, err = r.Read(b)
	if err != nil {
		return fmt.Errorf("init: error reading magic number of file: %v", err)
	}
	if bytes.Compare(b, u32tob(MagicNumber)) != 0 {
		return fmt.Errorf("can only read from tsm file")
	}
	_, err = r.Read(b)
	if err != nil {
		return fmt.Errorf("init: error reading version: %v", err)
	}
	if b[0] != Version {
		return fmt.Errorf("init: file is version %b. expected %b", b[0], Version)
	}

	return nil
}
