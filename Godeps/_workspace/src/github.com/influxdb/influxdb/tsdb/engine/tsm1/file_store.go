package tsm1

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type TSMFile interface {
	// Path returns the underlying file path for the TSMFile.  If the file
	// has not be written or loaded from disk, the zero value is returne.
	Path() string

	// Read returns all the values in the block where time t resides
	Read(key string, t time.Time) ([]Value, error)

	// Read returns all the values in the block identified by entry.
	ReadAt(entry *IndexEntry, values []Value) ([]Value, error)

	// Entries returns the index entries for all blocks for the given key.
	Entries(key string) []*IndexEntry

	// Returns true if the TSMFile may contain a value with the specified
	// key and time
	ContainsValue(key string, t time.Time) bool

	// Contains returns true if the file contains any values for the given
	// key.
	Contains(key string) bool

	// TimeRange returns the min and max time across all keys in the file.
	TimeRange() (time.Time, time.Time)

	// KeyRange returns the min and max keys in the file.
	KeyRange() (string, string)

	// Keys returns all keys contained in the file.
	Keys() []string

	// Type returns the block type of the values stored for the key.  Returns one of
	// BlockFloat64, BlockInt64, BlockBool, BlockString.  If key does not exist,
	// an error is returned.
	Type(key string) (byte, error)

	// Delete removes the keys from the set of keys available in this file.
	Delete(keys []string) error

	// HasTombstones returns true if file contains values that have been deleted.
	HasTombstones() bool

	// Close the underlying file resources
	Close() error

	// Size returns the size of the file on disk in bytes.
	Size() uint32

	// Remove deletes the file from the filesystem
	Remove() error

	// Stats returns summary information about the TSM file.
	Stats() FileStat
}

type FileStore struct {
	mu           sync.RWMutex
	lastModified time.Time

	currentGeneration int
	dir               string

	files []TSMFile

	Logger       *log.Logger
	traceLogging bool
}

type FileStat struct {
	Path             string
	HasTombstone     bool
	Size             uint32
	LastModified     time.Time
	MinTime, MaxTime time.Time
	MinKey, MaxKey   string
}

func (f FileStat) OverlapsTimeRange(min, max time.Time) bool {
	return (f.MinTime.Equal(max) || f.MinTime.Before(max)) &&
		(f.MaxTime.Equal(min) || f.MaxTime.After(min))
}

func (f FileStat) OverlapsKeyRange(min, max string) bool {
	return min != "" && max != "" && f.MinKey <= max && f.MaxKey >= min
}

func (f FileStat) ContainsKey(key string) bool {
	return f.MinKey >= key || key <= f.MaxKey
}

func NewFileStore(dir string) *FileStore {
	return &FileStore{
		dir:          dir,
		lastModified: time.Now(),
		Logger:       log.New(os.Stderr, "[filestore]", log.LstdFlags),
	}
}

// Returns the number of TSM files currently loaded
func (f *FileStore) Count() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.files)
}

// CurrentGeneration returns the current generation of the TSM files
func (f *FileStore) CurrentGeneration() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.currentGeneration
}

// NextGeneration returns the max file ID + 1
func (f *FileStore) NextGeneration() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.currentGeneration++
	return f.currentGeneration
}

func (f *FileStore) Add(files ...TSMFile) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.files = append(f.files, files...)
}

// Remove removes the files with matching paths from the set of active files.  It does
// not remove the paths from disk.
func (f *FileStore) Remove(paths ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var active []TSMFile
	for _, file := range f.files {
		keep := true
		for _, remove := range paths {
			if remove == file.Path() {
				keep = false
				break
			}
		}

		if keep {
			active = append(active, file)
		}
	}
	f.files = active
}

func (f *FileStore) Keys() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	uniqueKeys := map[string]struct{}{}
	for _, f := range f.files {
		for _, key := range f.Keys() {
			uniqueKeys[key] = struct{}{}
		}
	}

	var keys []string
	for key := range uniqueKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (f *FileStore) Type(key string) (byte, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, f := range f.files {
		if f.Contains(key) {
			return f.Type(key)
		}
	}
	return 0, fmt.Errorf("unknown type for %v", key)
}

func (f *FileStore) Delete(keys []string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.lastModified = time.Now()

	for _, file := range f.files {
		if err := file.Delete(keys); err != nil {
			return err
		}
	}
	return nil
}

func (f *FileStore) Open() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Not loading files from disk so nothing to do
	if f.dir == "" {
		return nil
	}

	files, err := filepath.Glob(filepath.Join(f.dir, fmt.Sprintf("*.%s", TSMFileExtension)))
	if err != nil {
		return err
	}

	for i, fn := range files {
		start := time.Now()

		// Keep track of the latest ID
		generation, _, err := ParseTSMFileName(fn)
		if err != nil {
			return err
		}

		if generation >= f.currentGeneration {
			f.currentGeneration = generation + 1
		}

		file, err := os.OpenFile(fn, os.O_RDONLY, 0666)
		if err != nil {
			return fmt.Errorf("error opening file %s: %v", fn, err)
		}

		df, err := NewTSMReaderWithOptions(TSMReaderOptions{
			MMAPFile: file,
		})
		if err != nil {
			return fmt.Errorf("error opening memory map for file %s: %v", fn, err)
		}
		if f.traceLogging {
			f.Logger.Printf("%s (#%d) opened in %v", fn, i, time.Now().Sub(start))
		}

		f.files = append(f.files, df)
	}
	return nil
}

func (f *FileStore) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	for _, f := range f.files {
		f.Close()
	}

	f.files = nil
	return nil
}

func (f *FileStore) Read(key string, t time.Time) ([]Value, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, f := range f.files {
		// Can this file possibly contain this key and timestamp?
		if !f.Contains(key) {
			continue
		}

		// May have the key and time we are looking for so try to find
		v, err := f.Read(key, t)
		if err != nil {
			return nil, err
		}

		if len(v) > 0 {
			return v, nil
		}
	}
	return nil, nil
}

func (f *FileStore) KeyCursor(key string) *KeyCursor {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return &KeyCursor{key: key, fs: f}
}

func (f *FileStore) Stats() []FileStat {
	f.mu.RLock()
	defer f.mu.RUnlock()
	stats := make([]FileStat, len(f.files))
	for i, fd := range f.files {
		stats[i] = fd.Stats()
	}

	return stats
}

func (f *FileStore) Replace(oldFiles, newFiles []string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.lastModified = time.Now()

	// Copy the current set of active files while we rename
	// and load the new files.  We copy the pointers here to minimize
	// the time that locks are held as well as to ensure that the replacement
	// is atomic.©
	var updated []TSMFile
	for _, t := range f.files {
		updated = append(updated, t)
	}

	// Rename all the new files to make them live on restart
	for _, file := range newFiles {
		var newName = file
		if strings.HasSuffix(file, ".tmp") {
			// The new TSM files have a tmp extension.  First rename them.
			newName = file[:len(file)-4]
			if err := os.Rename(file, newName); err != nil {
				return err
			}
		}

		fd, err := os.Open(newName)
		if err != nil {
			return err
		}

		tsm, err := NewTSMReaderWithOptions(TSMReaderOptions{
			MMAPFile: fd,
		})
		if err != nil {
			return err
		}
		updated = append(updated, tsm)
	}

	// We need to prune our set of active files now
	var active []TSMFile
	for _, file := range updated {
		keep := true
		for _, remove := range oldFiles {
			if remove == file.Path() {
				keep = false
				if err := file.Close(); err != nil {
					return err
				}

				if err := file.Remove(); err != nil {
					return err
				}
				break
			}
		}

		if keep {
			active = append(active, file)
		}
	}

	f.files = active

	return nil
}

// LastModified returns the last time the file store was updated with new
// TSM files or a delete
func (f *FileStore) LastModified() time.Time {
	f.mu.RLock()
	defer f.mu.RUnlock()

	return f.lastModified
}

// locations returns the files and index blocks for a key and time.  ascending indicates
// whether the key will be scan in ascending time order or descenging time order.
func (f *FileStore) locations(key string, t time.Time, ascending bool) []*location {
	var locations []*location

	f.mu.RLock()
	filesSnapshot := make([]TSMFile, len(f.files))
	for i := range f.files {
		filesSnapshot[i] = f.files[i]
	}
	f.mu.RUnlock()

	for _, fd := range filesSnapshot {
		minTime, maxTime := fd.TimeRange()

		// If we ascending and the max time of the file is before where we want to start
		// skip it.
		if ascending && maxTime.Before(t) {
			continue
			// If we are descending and the min time fo the file is after where we want to start,
			// then skip it.
		} else if !ascending && minTime.After(t) {
			continue
		}

		// This file could potential contain points we are looking for so find the blocks for
		// the given key.
		for _, ie := range fd.Entries(key) {
			// If we ascending and the max time of a block is before where we are looking, skip
			// it since the data is out of our range
			if ascending && ie.MaxTime.Before(t) {
				continue
				// If we descending and the min time of a block is after where we are looking, skip
				// it since the data is out of our range
			} else if !ascending && minTime.After(t) {
				continue
			}

			// Otherwise, add this file and block location
			locations = append(locations, &location{
				r:     fd,
				entry: ie,
			})
		}
	}
	return locations
}

// ParseTSMFileName parses the generation and sequence from a TSM file name.
func ParseTSMFileName(name string) (int, int, error) {
	base := filepath.Base(name)
	idx := strings.Index(base, ".")
	if idx == -1 {
		return 0, 0, fmt.Errorf("file %s is named incorrectly", name)
	}

	id := base[:idx]

	parts := strings.Split(id, "-")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("file %s is named incorrectly", name)
	}

	generation, err := strconv.ParseUint(parts[0], 10, 32)
	sequence, err := strconv.ParseUint(parts[1], 10, 32)

	return int(generation), int(sequence), err
}

type KeyCursor struct {
	key string
	fs  *FileStore

	// seeks is all the file locations that we need to return during iteration.
	seeks []*location

	// current is the set of blocks possibly containing the next set of points.
	// Normally this is just one entry, but there may be multiple if points have
	// been overwritten.
	current []*location
	buf     []Value

	// pos is the index within seeks.  Based on ascending, it will increment or
	// decrement through the size of seeks slice.
	pos       int
	ascending bool

	// ready indicates that we know the files and blocks to seek to for the key.
	ready bool

	// duplicates is a hint that there are overlapping blocks for this key in
	// multiple files (e.g. points have been overwritten but not fully compacted)
	// If this is true, we need to scan the duplicate blocks and dedup the points
	// as query time until they are compacted.
	duplicates bool
}

type location struct {
	r     TSMFile
	entry *IndexEntry
}

func (c *KeyCursor) init(t time.Time, ascending bool) {
	if c.ready {
		return
	}
	c.ascending = ascending
	c.seeks = c.fs.locations(c.key, t, ascending)

	if len(c.seeks) > 0 {
		for i := 1; i < len(c.seeks); i++ {
			prev := c.seeks[i-1]
			cur := c.seeks[i]

			if prev.entry.MaxTime.Equal(cur.entry.MinTime) || prev.entry.MaxTime.After(cur.entry.MinTime) {
				c.duplicates = true
				break
			}
		}
	}
	c.buf = make([]Value, 1000)
	c.ready = true
}

func (c *KeyCursor) SeekTo(t time.Time, ascending bool) ([]Value, error) {
	c.init(t, ascending)
	if len(c.seeks) == 0 {
		return nil, nil
	}
	c.current = nil

	if ascending {
		for i, e := range c.seeks {
			if t.Before(e.entry.MinTime) || e.entry.Contains(t) {
				// Record the position of the first block matching our seek time
				if len(c.current) == 0 {
					c.pos = i
				}

				c.current = append(c.current, e)

				// If we don't have duplicates, break.  Otherwise, keep looking for additional blocks containing
				// this point.
				if !c.duplicates {
					break
				}
			}
		}
	} else {
		for i := len(c.seeks) - 1; i >= 0; i-- {
			e := c.seeks[i]
			if t.After(e.entry.MaxTime) || e.entry.Contains(t) {
				// Record the position of the first block matching our seek time
				if len(c.current) == 0 {
					c.pos = i
				}

				c.current = append(c.current, e)

				// If we don't have duplicates, break.  Otherwise, keep looking for additional blocks containing
				// this point.
				if !c.duplicates {
					break
				}
			}
		}
	}

	return c.readAt()
}

func (c *KeyCursor) readAt() ([]Value, error) {
	// No matching blocks to decode
	if len(c.current) == 0 {
		return nil, nil
	}

	// First block is the oldest block containing the points we're search for.
	first := c.current[0]
	values, err := first.r.ReadAt(first.entry, c.buf[:0])

	// Only one block with this key and time range so return it
	if len(c.current) == 1 {
		return values, err
	}

	// Otherwise, search the remaining blocks that overlap and append their values so we can
	// dedup them.
	for i := 1; i < len(c.current); i++ {
		cur := c.current[i]
		if c.ascending && cur.entry.OverlapsTimeRange(first.entry.MinTime, first.entry.MaxTime) {
			c.pos++
			v, err := cur.r.ReadAt(cur.entry, nil)
			if err != nil {
				return nil, err
			}
			values = append(values, v...)

		} else if !c.ascending && cur.entry.OverlapsTimeRange(first.entry.MinTime, first.entry.MaxTime) {
			c.pos--

			v, err := cur.r.ReadAt(cur.entry, nil)
			if err != nil {
				return nil, err
			}
			values = append(v, values...)
		}
	}

	return Values(values).Deduplicate(), err
}

func (c *KeyCursor) Next(ascending bool) ([]Value, error) {
	c.current = c.current[:0]

	if ascending {
		c.pos++
		if c.pos >= len(c.seeks) {
			return nil, nil
		}

		// Append the first matching block
		c.current = []*location{c.seeks[c.pos]}

		// If we have ovelapping blocks, append all their values so we can dedup
		if c.duplicates {
			first := c.seeks[c.pos]
			for i := c.pos; i < len(c.seeks); i++ {
				if c.seeks[i].entry.MinTime.Before(first.entry.MaxTime) || c.seeks[i].entry.MinTime.Equal(first.entry.MaxTime) {
					c.current = append(c.current, c.seeks[i])
				}
			}
		}

		return c.readAt()

	} else {
		c.pos--
		if c.pos < 0 {
			return nil, nil
		}

		// Append the first matching block
		c.current = []*location{c.seeks[c.pos]}

		// If we have ovelapping blocks, append all their values so we can dedup
		if c.duplicates {
			first := c.seeks[c.pos]
			for i := c.pos; i >= 0; i-- {
				if c.seeks[i].entry.MaxTime.After(first.entry.MinTime) || c.seeks[i].entry.MaxTime.Equal(first.entry.MinTime) {
					c.current = append(c.current, c.seeks[i])
				}
			}
		}

		return c.readAt()
	}
}
