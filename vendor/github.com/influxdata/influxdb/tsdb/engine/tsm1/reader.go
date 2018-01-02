package tsm1

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"sync"
	"sync/atomic"
)

var ErrFileInUse = fmt.Errorf("file still in use")

type TSMReader struct {
	// refs is the count of active references to this reader
	refs int64

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
	lastModified int64
}

// TSMIndex represent the index section of a TSM file.  The index records all
// blocks, their locations, sizes, min and max times.
type TSMIndex interface {

	// Delete removes the given keys from the index.
	Delete(keys []string)

	// DeleteRange removes the given keys with data between minTime and maxTime from the index.
	DeleteRange(keys []string, minTime, maxTime int64)

	// Contains return true if the given key exists in the index.
	Contains(key string) bool

	// ContainsValue returns true if key and time might exists in this file.  This function could
	// return true even though the actual point does not exists.  For example, the key may
	// exists in this file, but not have point exactly at time t.
	ContainsValue(key string, timestamp int64) bool

	// Entries returns all index entries for a key.
	Entries(key string) []IndexEntry

	// ReadEntries reads the index entries for key into entries.
	ReadEntries(key string, entries *[]IndexEntry)

	// Entry returns the index entry for the specified key and timestamp.  If no entry
	// matches the key and timestamp, nil is returned.
	Entry(key string, timestamp int64) *IndexEntry

	// Key returns the key in the index at the given postion.
	Key(index int) (string, []IndexEntry)

	// KeyAt returns the key in the index at the given postion.
	KeyAt(index int) ([]byte, byte)

	// KeyCount returns the count of unique keys in the index.
	KeyCount() int

	// Size returns the size of a the current index in bytes
	Size() uint32

	// TimeRange returns the min and max time across all keys in the file.
	TimeRange() (int64, int64)

	// TombstoneRange returns ranges of time that are deleted for the given key.
	TombstoneRange(key string) []TimeRange

	// KeyRange returns the min and max keys in the file.
	KeyRange() (string, string)

	// Type returns the block type of the values stored for the key.  Returns one of
	// BlockFloat64, BlockInt64, BlockBool, BlockString.  If key does not exist,
	// an error is returned.
	Type(key string) (byte, error)

	// UnmarshalBinary populates an index from an encoded byte slice
	// representation of an index.
	UnmarshalBinary(b []byte) error
}

// BlockIterator allows iterating over each block in a TSM file in order.  It provides
// raw access to the block bytes without decoding them.
type BlockIterator struct {
	r *TSMReader

	// i is the current key index
	i int

	// n is the total number of keys
	n int

	key     string
	entries []IndexEntry
	err     error
}

func (b *BlockIterator) PeekNext() string {
	if len(b.entries) > 1 {
		return b.key
	} else if b.n-b.i > 1 {
		key, _ := b.r.KeyAt(b.i + 1)
		return string(key)
	}
	return ""
}

func (b *BlockIterator) Next() bool {
	if b.n-b.i == 0 && len(b.entries) == 0 {
		return false
	}

	if len(b.entries) > 0 {
		b.entries = b.entries[1:]
		if len(b.entries) > 0 {
			return true
		}
	}

	if b.n-b.i > 0 {
		b.key, b.entries = b.r.Key(b.i)
		b.i++

		if len(b.entries) > 0 {
			return true
		}
	}

	return false
}

func (b *BlockIterator) Read() (string, int64, int64, uint32, []byte, error) {
	if b.err != nil {
		return "", 0, 0, 0, nil, b.err
	}
	checksum, buf, err := b.r.readBytes(&b.entries[0], nil)
	if err != nil {
		return "", 0, 0, 0, nil, err
	}
	return b.key, b.entries[0].MinTime, b.entries[0].MaxTime, checksum, buf, err
}

// blockAccessor abstracts a method of accessing blocks from a
// TSM file.
type blockAccessor interface {
	init() (*indirectIndex, error)
	read(key string, timestamp int64) ([]Value, error)
	readAll(key string) ([]Value, error)
	readBlock(entry *IndexEntry, values []Value) ([]Value, error)
	readFloatBlock(entry *IndexEntry, values *[]FloatValue) ([]FloatValue, error)
	readIntegerBlock(entry *IndexEntry, values *[]IntegerValue) ([]IntegerValue, error)
	readStringBlock(entry *IndexEntry, values *[]StringValue) ([]StringValue, error)
	readBooleanBlock(entry *IndexEntry, values *[]BooleanValue) ([]BooleanValue, error)
	readBytes(entry *IndexEntry, buf []byte) (uint32, []byte, error)
	rename(path string) error
	path() string
	close() error
}

func NewTSMReader(f *os.File) (*TSMReader, error) {
	t := &TSMReader{}

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	t.size = stat.Size()
	t.lastModified = stat.ModTime().UnixNano()
	t.accessor = &mmapAccessor{
		f: f,
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
	var cur, prev Tombstone
	batch := make([]string, 0, 4096)

	if err := t.tombstoner.Walk(func(ts Tombstone) error {
		cur = ts
		if len(batch) > 0 {
			if prev.Min != cur.Min || prev.Max != cur.Max {
				t.index.DeleteRange(batch, prev.Min, prev.Max)
				batch = batch[:0]
			}
		}
		batch = append(batch, ts.Key)

		if len(batch) >= 4096 {
			t.index.DeleteRange(batch, prev.Min, prev.Max)
			batch = batch[:0]
		}
		prev = ts
		return nil
	}); err != nil {
		return fmt.Errorf("init: read tombstones: %v", err)
	}

	if len(batch) > 0 {
		t.index.DeleteRange(batch, cur.Min, cur.Max)
	}
	return nil
}

func (t *TSMReader) Path() string {
	t.mu.RLock()
	p := t.accessor.path()
	t.mu.RUnlock()
	return p
}

func (t *TSMReader) Key(index int) (string, []IndexEntry) {
	return t.index.Key(index)
}

// KeyAt returns the key and key type at position idx in the index.
func (t *TSMReader) KeyAt(idx int) ([]byte, byte) {
	return t.index.KeyAt(idx)
}

func (t *TSMReader) ReadAt(entry *IndexEntry, vals []Value) ([]Value, error) {
	t.mu.RLock()
	v, err := t.accessor.readBlock(entry, vals)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) ReadFloatBlockAt(entry *IndexEntry, vals *[]FloatValue) ([]FloatValue, error) {
	t.mu.RLock()
	v, err := t.accessor.readFloatBlock(entry, vals)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) ReadIntegerBlockAt(entry *IndexEntry, vals *[]IntegerValue) ([]IntegerValue, error) {
	t.mu.RLock()
	v, err := t.accessor.readIntegerBlock(entry, vals)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) ReadStringBlockAt(entry *IndexEntry, vals *[]StringValue) ([]StringValue, error) {
	t.mu.RLock()
	v, err := t.accessor.readStringBlock(entry, vals)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) ReadBooleanBlockAt(entry *IndexEntry, vals *[]BooleanValue) ([]BooleanValue, error) {
	t.mu.RLock()
	v, err := t.accessor.readBooleanBlock(entry, vals)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) Read(key string, timestamp int64) ([]Value, error) {
	t.mu.RLock()
	v, err := t.accessor.read(key, timestamp)
	t.mu.RUnlock()
	return v, err
}

// ReadAll returns all values for a key in all blocks.
func (t *TSMReader) ReadAll(key string) ([]Value, error) {
	t.mu.RLock()
	v, err := t.accessor.readAll(key)
	t.mu.RUnlock()
	return v, err
}

func (t *TSMReader) readBytes(e *IndexEntry, b []byte) (uint32, []byte, error) {
	t.mu.RLock()
	n, v, err := t.accessor.readBytes(e, b)
	t.mu.RUnlock()
	return n, v, err
}

func (t *TSMReader) Type(key string) (byte, error) {
	return t.index.Type(key)
}

func (t *TSMReader) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.refs > 0 {
		return ErrFileInUse
	}

	if err := t.accessor.close(); err != nil {
		return err
	}

	return nil
}

// Ref records a usage of this TSMReader.  If there are active references
// when the reader is closed or removed, the reader will remain open until
// there are no more references.
func (t *TSMReader) Ref() {
	atomic.AddInt64(&t.refs, 1)
}

// Unref removes a usage record of this TSMReader.  If the Reader was closed
// by another goroutine while there were active references, the file will
// be closed and remove
func (t *TSMReader) Unref() {
	atomic.AddInt64(&t.refs, -1)
}

func (t *TSMReader) InUse() bool {
	refs := atomic.LoadInt64(&t.refs)
	return refs > 0
}

// Remove removes any underlying files stored on disk for this reader.
func (t *TSMReader) Remove() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.remove()
}

func (t *TSMReader) Rename(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.accessor.rename(path)
}

// Remove removes any underlying files stored on disk for this reader.
func (t *TSMReader) remove() error {
	path := t.accessor.path()

	if t.InUse() {
		return ErrFileInUse
	}

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
func (t *TSMReader) ContainsValue(key string, ts int64) bool {
	return t.index.ContainsValue(key, ts)
}

// DeleteRange removes the given points for keys between minTime and maxTime
func (t *TSMReader) DeleteRange(keys []string, minTime, maxTime int64) error {
	if err := t.tombstoner.AddRange(keys, minTime, maxTime); err != nil {
		return err
	}

	t.index.DeleteRange(keys, minTime, maxTime)
	return nil
}

func (t *TSMReader) Delete(keys []string) error {
	if err := t.tombstoner.Add(keys); err != nil {
		return err
	}

	t.index.Delete(keys)
	return nil
}

// TimeRange returns the min and max time across all keys in the file.
func (t *TSMReader) TimeRange() (int64, int64) {
	return t.index.TimeRange()
}

// KeyRange returns the min and max key across all keys in the file.
func (t *TSMReader) KeyRange() (string, string) {
	return t.index.KeyRange()
}

func (t *TSMReader) KeyCount() int {
	return t.index.KeyCount()
}

func (t *TSMReader) Entries(key string) []IndexEntry {
	return t.index.Entries(key)
}

func (t *TSMReader) ReadEntries(key string, entries *[]IndexEntry) {
	t.index.ReadEntries(key, entries)
}

func (t *TSMReader) IndexSize() uint32 {
	return t.index.Size()
}

func (t *TSMReader) Size() uint32 {
	t.mu.RLock()
	size := t.size
	t.mu.RUnlock()
	return uint32(size)
}

func (t *TSMReader) LastModified() int64 {
	t.mu.RLock()
	lm := t.lastModified
	t.mu.RUnlock()
	return lm
}

// HasTombstones return true if there are any tombstone entries recorded.
func (t *TSMReader) HasTombstones() bool {
	t.mu.RLock()
	b := t.tombstoner.HasTombstones()
	t.mu.RUnlock()
	return b
}

// TombstoneFiles returns any tombstone files associated with this TSM file.
func (t *TSMReader) TombstoneFiles() []FileStat {
	t.mu.RLock()
	fs := t.tombstoner.TombstoneFiles()
	t.mu.RUnlock()
	return fs
}

// TombstoneRange returns ranges of time that are deleted for the given key.
func (t *TSMReader) TombstoneRange(key string) []TimeRange {
	t.mu.RLock()
	tr := t.index.TombstoneRange(key)
	t.mu.RUnlock()
	return tr
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

func (t *TSMReader) BlockIterator() *BlockIterator {
	return &BlockIterator{
		r: t,
		n: t.index.KeyCount(),
	}
}

// deref removes mmap references held by another object.
func (t *TSMReader) deref(d dereferencer) {
	if acc, ok := t.accessor.(*mmapAccessor); ok && acc.b != nil {
		d.Dereference(acc.b)
	}
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
	minTime, maxTime int64

	// tombstones contains only the tombstoned keys with subset of time values deleted.  An
	// entry would exist here if a subset of the points for a key were deleted and the file
	// had not be re-compacted to remove the points on disk.
	tombstones map[string][]TimeRange
}

type TimeRange struct {
	Min, Max int64
}

func NewIndirectIndex() *indirectIndex {
	return &indirectIndex{
		tombstones: make(map[string][]TimeRange),
	}
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
		keyLen := int32(binary.BigEndian.Uint16(d.b[offset : offset+2]))

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
func (d *indirectIndex) Entries(key string) []IndexEntry {
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
		// searched should be inserted at position 0.  Make sure the key in the index
		// matches the search value.
		if !bytes.Equal(kb, k) {
			return nil
		}

		// Read and return all the entries
		ofs += n
		var entries indexEntries
		if _, err := readEntries(d.b[ofs:], &entries); err != nil {
			panic(fmt.Sprintf("error reading entries: %v", err))
		}
		return entries.entries
	}

	// The key is not in the index.  i is the index where it would be inserted.
	return nil
}

// ReadEntries returns all index entries for a key.
func (d *indirectIndex) ReadEntries(key string, entries *[]IndexEntry) {
	*entries = d.Entries(key)
}

// Entry returns the index entry for the specified key and timestamp.  If no entry
// matches the key an timestamp, nil is returned.
func (d *indirectIndex) Entry(key string, timestamp int64) *IndexEntry {
	entries := d.Entries(key)
	for _, entry := range entries {
		if entry.Contains(timestamp) {
			return &entry
		}
	}
	return nil
}

func (d *indirectIndex) Key(idx int) (string, []IndexEntry) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if idx < 0 || idx >= len(d.offsets) {
		return "", nil
	}
	n, key, err := readKey(d.b[d.offsets[idx]:])
	if err != nil {
		return "", nil
	}

	var entries indexEntries
	if _, err := readEntries(d.b[int(d.offsets[idx])+n:], &entries); err != nil {
		return "", nil
	}
	return string(key), entries.entries
}

func (d *indirectIndex) KeyAt(idx int) ([]byte, byte) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if idx < 0 || idx >= len(d.offsets) {
		return nil, 0
	}
	n, key, _ := readKey(d.b[d.offsets[idx]:])
	return key, d.b[d.offsets[idx]+int32(n)]
}

func (d *indirectIndex) KeyCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return len(d.offsets)
}

func (d *indirectIndex) Delete(keys []string) {
	if len(keys) == 0 {
		return
	}

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

func (d *indirectIndex) DeleteRange(keys []string, minTime, maxTime int64) {
	// No keys, nothing to do
	if len(keys) == 0 {
		return
	}

	// If we're deleting the max time range, just use tombstoning to remove the
	// key from the offsets slice
	if minTime == math.MinInt64 && maxTime == math.MaxInt64 {
		d.Delete(keys)
		return
	}

	// Is the range passed in outside of the time range for the file?
	min, max := d.TimeRange()
	if minTime > max || maxTime < min {
		return
	}

	tombstones := map[string][]TimeRange{}
	for _, k := range keys {

		// Is the range passed in outside the time range for this key?
		entries := d.Entries(k)

		// If multiple tombstones are saved for the same key
		if len(entries) == 0 {
			continue
		}

		min, max := entries[0].MinTime, entries[len(entries)-1].MaxTime
		if minTime > max || maxTime < min {
			continue
		}

		// Is the range passed in cover every value for the key?
		if minTime <= min && maxTime >= max {
			d.Delete(keys)
			continue
		}

		tombstones[k] = append(tombstones[k], TimeRange{minTime, maxTime})
	}

	if len(tombstones) == 0 {
		return
	}

	d.mu.Lock()
	for k, v := range tombstones {
		d.tombstones[k] = append(d.tombstones[k], v...)
	}
	d.mu.Unlock()

}

func (d *indirectIndex) TombstoneRange(key string) []TimeRange {
	d.mu.RLock()
	r := d.tombstones[key]
	d.mu.RUnlock()
	return r
}

func (d *indirectIndex) Contains(key string) bool {
	return len(d.Entries(key)) > 0
}

func (d *indirectIndex) ContainsValue(key string, timestamp int64) bool {
	entry := d.Entry(key, timestamp)
	if entry == nil {
		return false
	}

	d.mu.RLock()
	tombstones := d.tombstones[key]
	d.mu.RUnlock()

	for _, t := range tombstones {
		if t.Min <= timestamp && t.Max >= timestamp {
			return false
		}
	}
	return true
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

func (d *indirectIndex) TimeRange() (int64, int64) {
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
	if len(b) == 0 {
		return nil
	}

	//var minKey, maxKey []byte
	var minTime, maxTime int64 = math.MaxInt64, 0

	// To create our "indirect" index, we need to find the location of all the keys in
	// the raw byte slice.  The keys are listed once each (in sorted order).  Following
	// each key is a time ordered list of index entry blocks for that key.  The loop below
	// basically skips across the slice keeping track of the counter when we are at a key
	// field.
	var i int32
	iMax := int32(len(b))
	for i < iMax {
		d.offsets = append(d.offsets, i)

		// Skip to the start of the values
		// key length value (2) + type (1) + length of key
		if i+2 >= iMax {
			return fmt.Errorf("indirectIndex: not enough data for key length value")
		}
		i += 3 + int32(binary.BigEndian.Uint16(b[i:i+2]))

		// count of index entries
		if i+indexCountSize >= iMax {
			return fmt.Errorf("indirectIndex: not enough data for index entries count")
		}
		count := int32(binary.BigEndian.Uint16(b[i : i+indexCountSize]))
		i += indexCountSize

		// Find the min time for the block
		if i+8 >= iMax {
			return fmt.Errorf("indirectIndex: not enough data for min time")
		}
		minT := int64(binary.BigEndian.Uint64(b[i : i+8]))
		if minT < minTime {
			minTime = minT
		}

		i += (count - 1) * indexEntrySize

		// Find the max time for the block
		if i+16 >= iMax {
			return fmt.Errorf("indirectIndex: not enough data for max time")
		}
		maxT := int64(binary.BigEndian.Uint64(b[i+8 : i+16]))
		if maxT > maxTime {
			maxTime = maxT
		}

		i += indexEntrySize
	}

	firstOfs := d.offsets[0]
	_, key, err := readKey(b[firstOfs:])
	if err != nil {
		return err
	}
	d.minKey = string(key)

	lastOfs := d.offsets[len(d.offsets)-1]
	_, key, err = readKey(b[lastOfs:])
	if err != nil {
		return err
	}
	d.maxKey = string(key)

	d.minTime = minTime
	d.maxTime = maxTime

	return nil
}

func (d *indirectIndex) Size() uint32 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return uint32(len(d.b))
}

// mmapAccess is mmap based block accessor.  It access blocks through an
// MMAP file interface.
type mmapAccessor struct {
	mu sync.RWMutex

	f     *os.File
	b     []byte
	index *indirectIndex
}

func (m *mmapAccessor) init() (*indirectIndex, error) {
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
	if len(m.b) < 8 {
		return nil, fmt.Errorf("mmapAccessor: byte slice too small for indirectIndex")
	}

	indexOfsPos := len(m.b) - 8
	indexStart := binary.BigEndian.Uint64(m.b[indexOfsPos : indexOfsPos+8])
	if indexStart >= uint64(indexOfsPos) {
		return nil, fmt.Errorf("mmapAccessor: invalid indexStart")
	}

	m.index = NewIndirectIndex()
	if err := m.index.UnmarshalBinary(m.b[indexStart:indexOfsPos]); err != nil {
		return nil, err
	}

	return m.index, nil
}

func (m *mmapAccessor) rename(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	err := munmap(m.b)
	if err != nil {
		return err
	}

	if err := m.f.Close(); err != nil {
		return err
	}

	if err := renameFile(m.f.Name(), path); err != nil {
		return err
	}

	m.f, err = os.Open(path)
	if err != nil {
		return err
	}

	if _, err := m.f.Seek(0, 0); err != nil {
		return err
	}

	stat, err := m.f.Stat()
	if err != nil {
		return err
	}

	m.b, err = mmap(m.f, 0, int(stat.Size()))
	if err != nil {
		return err
	}

	return nil
}

func (m *mmapAccessor) read(key string, timestamp int64) ([]Value, error) {
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

func (m *mmapAccessor) readFloatBlock(entry *IndexEntry, values *[]FloatValue) ([]FloatValue, error) {
	m.mu.RLock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		m.mu.RUnlock()
		return nil, ErrTSMClosed
	}

	a, err := DecodeFloatBlock(m.b[entry.Offset+4:entry.Offset+int64(entry.Size)], values)
	m.mu.RUnlock()

	if err != nil {
		return nil, err
	}

	return a, nil
}

func (m *mmapAccessor) readIntegerBlock(entry *IndexEntry, values *[]IntegerValue) ([]IntegerValue, error) {
	m.mu.RLock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		m.mu.RUnlock()
		return nil, ErrTSMClosed
	}

	a, err := DecodeIntegerBlock(m.b[entry.Offset+4:entry.Offset+int64(entry.Size)], values)
	m.mu.RUnlock()

	if err != nil {
		return nil, err
	}

	return a, nil
}

func (m *mmapAccessor) readStringBlock(entry *IndexEntry, values *[]StringValue) ([]StringValue, error) {
	m.mu.RLock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		m.mu.RUnlock()
		return nil, ErrTSMClosed
	}

	a, err := DecodeStringBlock(m.b[entry.Offset+4:entry.Offset+int64(entry.Size)], values)
	m.mu.RUnlock()

	if err != nil {
		return nil, err
	}

	return a, nil
}

func (m *mmapAccessor) readBooleanBlock(entry *IndexEntry, values *[]BooleanValue) ([]BooleanValue, error) {
	m.mu.RLock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		m.mu.RUnlock()
		return nil, ErrTSMClosed
	}

	a, err := DecodeBooleanBlock(m.b[entry.Offset+4:entry.Offset+int64(entry.Size)], values)
	m.mu.RUnlock()

	if err != nil {
		return nil, err
	}

	return a, nil
}

func (m *mmapAccessor) readBytes(entry *IndexEntry, b []byte) (uint32, []byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if int64(len(m.b)) < entry.Offset+int64(entry.Size) {
		return 0, nil, ErrTSMClosed
	}

	// return the bytes after the 4 byte checksum
	return binary.BigEndian.Uint32(m.b[entry.Offset : entry.Offset+4]), m.b[entry.Offset+4 : entry.Offset+int64(entry.Size)], nil
}

// ReadAll returns all values for a key in all blocks.
func (m *mmapAccessor) readAll(key string) ([]Value, error) {
	blocks := m.index.Entries(key)
	if len(blocks) == 0 {
		return nil, nil
	}

	tombstones := m.index.TombstoneRange(key)

	m.mu.RLock()
	defer m.mu.RUnlock()

	var temp []Value
	var err error
	var values []Value
	for _, block := range blocks {
		var skip bool
		for _, t := range tombstones {
			// Should we skip this block because it contains points that have been deleted
			if t.Min <= block.MinTime && t.Max >= block.MaxTime {
				skip = true
				break
			}
		}

		if skip {
			continue
		}
		//TODO: Validate checksum
		temp = temp[:0]
		// The +4 is the 4 byte checksum length
		temp, err = DecodeBlock(m.b[block.Offset+4:block.Offset+int64(block.Size)], temp)
		if err != nil {
			return nil, err
		}

		// Filter out any values that were deleted
		for _, t := range tombstones {
			temp = Values(temp).Exclude(t.Min, t.Max)
		}

		values = append(values, temp...)
	}

	return values, nil
}

func (m *mmapAccessor) path() string {
	m.mu.RLock()
	path := m.f.Name()
	m.mu.RUnlock()
	return path
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
	entries []IndexEntry
}

func (a *indexEntries) Len() int      { return len(a.entries) }
func (a *indexEntries) Swap(i, j int) { a.entries[i], a.entries[j] = a.entries[j], a.entries[i] }
func (a *indexEntries) Less(i, j int) bool {
	return a.entries[i].MinTime < a.entries[j].MinTime
}

func (a *indexEntries) MarshalBinary() ([]byte, error) {
	buf := make([]byte, len(a.entries)*indexEntrySize)

	for i, entry := range a.entries {
		entry.AppendTo(buf[indexEntrySize*i:])
	}

	return buf, nil
}

func (a *indexEntries) WriteTo(w io.Writer) (total int64, err error) {
	var buf [indexEntrySize]byte
	var n int

	for _, entry := range a.entries {
		entry.AppendTo(buf[:])
		n, err = w.Write(buf[:])
		total += int64(n)
		if err != nil {
			return total, err
		}
	}

	return total, nil
}

func readKey(b []byte) (n int, key []byte, err error) {
	// 2 byte size of key
	n, size := 2, int(binary.BigEndian.Uint16(b[:2]))

	// N byte key
	key = b[n : n+size]

	n += len(key)
	return
}

func readEntries(b []byte, entries *indexEntries) (n int, err error) {
	if len(b) < 1+indexCountSize {
		return 0, fmt.Errorf("readEntries: data too short for headers")
	}

	// 1 byte block type
	entries.Type = b[n]
	n++

	// 2 byte count of index entries
	count := int(binary.BigEndian.Uint16(b[n : n+indexCountSize]))
	n += indexCountSize

	entries.entries = make([]IndexEntry, count)
	for i := 0; i < count; i++ {
		var ie IndexEntry
		start := i*indexEntrySize + indexCountSize + indexTypeSize
		end := start + indexEntrySize
		if end > len(b) {
			return 0, fmt.Errorf("readEntries: data too short for indexEntry %d", i)
		}
		if err := ie.UnmarshalBinary(b[start:end]); err != nil {
			return 0, fmt.Errorf("readEntries: unmarshal error: %v", err)
		}
		entries.entries[i] = ie
		n += indexEntrySize
	}
	return
}
