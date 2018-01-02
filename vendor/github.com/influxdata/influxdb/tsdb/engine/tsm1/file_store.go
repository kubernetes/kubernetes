package tsm1

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/models"
)

type TSMFile interface {
	// Path returns the underlying file path for the TSMFile.  If the file
	// has not be written or loaded from disk, the zero value is returned.
	Path() string

	// Read returns all the values in the block where time t resides
	Read(key string, t int64) ([]Value, error)

	// ReadAt returns all the values in the block identified by entry.
	ReadAt(entry *IndexEntry, values []Value) ([]Value, error)
	ReadFloatBlockAt(entry *IndexEntry, values *[]FloatValue) ([]FloatValue, error)
	ReadIntegerBlockAt(entry *IndexEntry, values *[]IntegerValue) ([]IntegerValue, error)
	ReadStringBlockAt(entry *IndexEntry, values *[]StringValue) ([]StringValue, error)
	ReadBooleanBlockAt(entry *IndexEntry, values *[]BooleanValue) ([]BooleanValue, error)

	// Entries returns the index entries for all blocks for the given key.
	Entries(key string) []IndexEntry
	ReadEntries(key string, entries *[]IndexEntry)

	// Returns true if the TSMFile may contain a value with the specified
	// key and time
	ContainsValue(key string, t int64) bool

	// Contains returns true if the file contains any values for the given
	// key.
	Contains(key string) bool

	// TimeRange returns the min and max time across all keys in the file.
	TimeRange() (int64, int64)

	// TombstoneRange returns ranges of time that are deleted for the given key.
	TombstoneRange(key string) []TimeRange

	// KeyRange returns the min and max keys in the file.
	KeyRange() (string, string)

	// KeyCount returns the number of distict keys in the file.
	KeyCount() int

	// KeyAt returns the key located at index position idx
	KeyAt(idx int) ([]byte, byte)

	// Type returns the block type of the values stored for the key.  Returns one of
	// BlockFloat64, BlockInt64, BlockBoolean, BlockString.  If key does not exist,
	// an error is returned.
	Type(key string) (byte, error)

	// Delete removes the keys from the set of keys available in this file.
	Delete(keys []string) error

	// DeleteRange removes the values for keys between min and max.
	DeleteRange(keys []string, min, max int64) error

	// HasTombstones returns true if file contains values that have been deleted.
	HasTombstones() bool

	// TombstoneFiles returns the tombstone filestats if there are any tombstones
	// written for this file.
	TombstoneFiles() []FileStat

	// Close the underlying file resources
	Close() error

	// Size returns the size of the file on disk in bytes.
	Size() uint32

	// Rename renames the existing TSM file to a new name and replaces the mmap backing slice using the new
	// file name.  Index and Reader state are not re-initialized.
	Rename(path string) error

	// Remove deletes the file from the filesystem
	Remove() error

	// Returns true if the file is currently in use by queries
	InUse() bool

	// Ref records that this file is actively in use
	Ref()

	// Unref records that this file is no longer in user
	Unref()

	// Stats returns summary information about the TSM file.
	Stats() FileStat

	// BlockIterator returns an iterator pointing to the first block in the file and
	// allows sequential iteration to each every block.
	BlockIterator() *BlockIterator

	// Removes mmap references held by another object.
	deref(dereferencer)
}

type dereferencer interface {
	Dereference([]byte)
}

// Statistics gathered by the FileStore.
const (
	statFileStoreBytes = "diskBytes"
	statFileStoreCount = "numFiles"
)

type FileStore struct {
	mu           sync.RWMutex
	lastModified time.Time
	// Most recently known file stats. If nil then stats will need to be
	// recalculated
	lastFileStats []FileStat

	currentGeneration int
	dir               string

	files []TSMFile

	logger       *log.Logger // Logger to be used for important messages
	traceLogger  *log.Logger // Logger to be used when trace-logging is on.
	logOutput    io.Writer   // Writer to be logger and traceLogger if active.
	traceLogging bool

	stats  *FileStoreStatistics
	purger *purger

	currentTempDirID int

	dereferencer dereferencer
}

type FileStat struct {
	Path             string
	HasTombstone     bool
	Size             uint32
	LastModified     int64
	MinTime, MaxTime int64
	MinKey, MaxKey   string
}

func (f FileStat) OverlapsTimeRange(min, max int64) bool {
	return f.MinTime <= max && f.MaxTime >= min
}

func (f FileStat) OverlapsKeyRange(min, max string) bool {
	return min != "" && max != "" && f.MinKey <= max && f.MaxKey >= min
}

func (f FileStat) ContainsKey(key string) bool {
	return f.MinKey >= key || key <= f.MaxKey
}

func NewFileStore(dir string) *FileStore {
	logger := log.New(os.Stderr, "[filestore] ", log.LstdFlags)
	fs := &FileStore{
		dir:          dir,
		lastModified: time.Now(),
		logger:       logger,
		traceLogger:  log.New(ioutil.Discard, "[filestore] ", log.LstdFlags),
		logOutput:    os.Stderr,
		stats:        &FileStoreStatistics{},
		purger: &purger{
			files:  map[string]TSMFile{},
			logger: logger,
		},
	}
	fs.purger.fileStore = fs
	return fs
}

// enableTraceLogging must be called before the FileStore is opened.
func (f *FileStore) enableTraceLogging(enabled bool) {
	f.traceLogging = enabled
	if enabled {
		f.traceLogger.SetOutput(f.logOutput)
	}
}

// SetLogOutput sets the logger used for all messages. It is safe for concurrent
// use.
func (f *FileStore) SetLogOutput(w io.Writer) {
	f.logger.SetOutput(w)

	// Set the trace logger's output only if trace logging is enabled.
	if f.traceLogging {
		f.traceLogger.SetOutput(w)
	}

	f.mu.Lock()
	f.logOutput = w
	f.mu.Unlock()
}

// FileStoreStatistics keeps statistics about the file store.
type FileStoreStatistics struct {
	DiskBytes int64
	FileCount int64
}

// Statistics returns statistics for periodic monitoring.
func (f *FileStore) Statistics(tags map[string]string) []models.Statistic {
	return []models.Statistic{{
		Name: "tsm1_filestore",
		Tags: tags,
		Values: map[string]interface{}{
			statFileStoreBytes: atomic.LoadInt64(&f.stats.DiskBytes),
			statFileStoreCount: atomic.LoadInt64(&f.stats.FileCount),
		},
	}}
}

// Returns the number of TSM files currently loaded
func (f *FileStore) Count() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.files)
}

// Files returns TSM files currently loaded.
func (f *FileStore) Files() []TSMFile {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.files
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
	for _, file := range files {
		atomic.AddInt64(&f.stats.DiskBytes, int64(file.Size()))
	}
	f.lastFileStats = nil
	f.files = append(f.files, files...)
	sort.Sort(tsmReaders(f.files))
	atomic.StoreInt64(&f.stats.FileCount, int64(len(f.files)))
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
		} else {
			// Removing the file, remove the file size from the total file store bytes
			atomic.AddInt64(&f.stats.DiskBytes, -int64(file.Size()))
		}
	}
	f.lastFileStats = nil
	f.files = active
	sort.Sort(tsmReaders(f.files))
	atomic.StoreInt64(&f.stats.FileCount, int64(len(f.files)))
}

// WalkKeys calls fn for every key in every TSM file known to the FileStore.  If the key
// exists in multiple files, it will be invoked for each file.
func (f *FileStore) WalkKeys(fn func(key []byte, typ byte) error) error {
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, f := range f.files {
		for i := 0; i < f.KeyCount(); i++ {
			key, typ := f.KeyAt(i)
			if err := fn(key, typ); err != nil {
				return err
			}
		}
	}
	return nil
}

// Keys returns all keys and types for all files
func (f *FileStore) Keys() map[string]byte {
	f.mu.RLock()
	defer f.mu.RUnlock()

	uniqueKeys := map[string]byte{}
	for _, f := range f.files {
		for i := 0; i < f.KeyCount(); i++ {
			key, typ := f.KeyAt(i)
			uniqueKeys[string(key)] = typ
		}
	}

	return uniqueKeys
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
	return f.DeleteRange(keys, math.MinInt64, math.MaxInt64)
}

// DeleteRange removes the values for keys between min and max.
func (f *FileStore) DeleteRange(keys []string, min, max int64) error {
	f.mu.Lock()
	f.lastModified = time.Now()
	f.mu.Unlock()

	return f.walkFiles(func(tsm TSMFile) error {
		return tsm.DeleteRange(keys, min, max)
	})
}

func (f *FileStore) Open() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Not loading files from disk so nothing to do
	if f.dir == "" {
		return nil
	}

	// find the current max ID for temp directories
	tmpfiles, err := ioutil.ReadDir(f.dir)
	if err != nil {
		return err
	}
	for _, fi := range tmpfiles {
		if fi.IsDir() && strings.HasSuffix(fi.Name(), ".tmp") {
			ss := strings.Split(filepath.Base(fi.Name()), ".")
			if len(ss) == 2 {
				if i, err := strconv.Atoi(ss[0]); err != nil {
					if i > f.currentTempDirID {
						f.currentTempDirID = i
					}
				}
			}
		}
	}

	files, err := filepath.Glob(filepath.Join(f.dir, fmt.Sprintf("*.%s", TSMFileExtension)))
	if err != nil {
		return err
	}

	// struct to hold the result of opening each reader in a goroutine
	type res struct {
		r   *TSMReader
		err error
	}

	readerC := make(chan *res)
	for i, fn := range files {
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

		// Accumulate file store size stat
		if fi, err := file.Stat(); err == nil {
			atomic.AddInt64(&f.stats.DiskBytes, fi.Size())
		}

		go func(idx int, file *os.File) {
			start := time.Now()
			df, err := NewTSMReader(file)
			f.logger.Printf("%s (#%d) opened in %v", file.Name(), idx, time.Now().Sub(start))

			if err != nil {
				readerC <- &res{r: df, err: fmt.Errorf("error opening memory map for file %s: %v", file.Name(), err)}
				return
			}
			readerC <- &res{r: df}
		}(i, file)
	}

	for range files {
		res := <-readerC
		if res.err != nil {

			return res.err
		}
		f.files = append(f.files, res.r)
	}
	close(readerC)

	sort.Sort(tsmReaders(f.files))
	atomic.StoreInt64(&f.stats.FileCount, int64(len(f.files)))
	return nil
}

func (f *FileStore) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	for _, file := range f.files {
		if f.dereferencer != nil {
			file.deref(f.dereferencer)
		}
		file.Close()
	}

	f.lastFileStats = nil
	f.files = nil
	atomic.StoreInt64(&f.stats.FileCount, 0)
	return nil
}

func (f *FileStore) Read(key string, t int64) ([]Value, error) {
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

func (f *FileStore) KeyCursor(key string, t int64, ascending bool) *KeyCursor {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return newKeyCursor(f, key, t, ascending)
}

func (f *FileStore) Stats() []FileStat {
	f.mu.RLock()
	if len(f.lastFileStats) > 0 {
		defer f.mu.RUnlock()
		return f.lastFileStats
	}
	f.mu.RUnlock()

	// The file stats cache is invalid due to changes to files. Need to
	// recalculate.
	f.mu.Lock()

	// If lastFileStats's capacity is far away from the number of entries
	// we need to add, then we'll reallocate.
	if cap(f.lastFileStats) < len(f.files)/2 {
		f.lastFileStats = make([]FileStat, 0, len(f.files))
	}

	for _, fd := range f.files {
		f.lastFileStats = append(f.lastFileStats, fd.Stats())
	}
	defer f.mu.Unlock()
	return f.lastFileStats
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

		tsm, err := NewTSMReader(fd)
		if err != nil {
			return err
		}
		updated = append(updated, tsm)
	}

	// We need to prune our set of active files now
	var active, inuse []TSMFile
	for _, file := range updated {
		keep := true
		for _, remove := range oldFiles {
			if remove == file.Path() {
				keep = false

				// If queries are running against this file, then we need to move it out of the
				// way and let them complete.  We'll then delete the original file to avoid
				// blocking callers upstream.  If the process crashes, the temp file is
				// cleaned up at startup automatically.
				if file.InUse() {
					// Copy all the tombstones related to this TSM file
					var deletes []string
					for _, t := range file.TombstoneFiles() {
						deletes = append(deletes, t.Path)
					}
					deletes = append(deletes, file.Path())

					// Rename the TSM file used by this reader
					tempPath := file.Path() + ".tmp"
					if err := file.Rename(tempPath); err != nil {
						return err
					}

					// Remove the old file and tombstones.  We can't use the normal TSMReader.Remove()
					// because it now refers to our temp file which we can't remove.
					for _, f := range deletes {
						if err := os.RemoveAll(f); err != nil {
							return err
						}
					}

					inuse = append(inuse, file)
					continue
				}

				// Remove any mmap references held by the index.
				if f.dereferencer != nil {
					file.deref(f.dereferencer)
				}

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

	if err := syncDir(f.dir); err != nil {
		return err
	}

	// Tell the purger about our in-use files we need to remove
	f.purger.add(inuse)

	f.lastFileStats = nil
	f.files = active
	sort.Sort(tsmReaders(f.files))
	atomic.StoreInt64(&f.stats.FileCount, int64(len(f.files)))

	// Recalculate the disk size stat
	var totalSize int64
	for _, file := range f.files {
		totalSize += int64(file.Size())
	}
	atomic.StoreInt64(&f.stats.DiskBytes, totalSize)

	return nil
}

// LastModified returns the last time the file store was updated with new
// TSM files or a delete
func (f *FileStore) LastModified() time.Time {
	f.mu.RLock()
	defer f.mu.RUnlock()

	return f.lastModified
}

// BlockCount returns number of values stored in the block at location idx
// in the file at path.  If path does not match any file in the store, 0 is
// returned.  If idx is out of range for the number of blocks in the file,
// 0 is returned.
func (f *FileStore) BlockCount(path string, idx int) int {
	f.mu.RLock()
	defer f.mu.RUnlock()

	if idx < 0 {
		return 0
	}

	for _, fd := range f.files {
		if fd.Path() == path {
			iter := fd.BlockIterator()
			for i := 0; i < idx; i++ {
				if !iter.Next() {
					return 0
				}
			}
			_, _, _, _, block, _ := iter.Read()
			return BlockCount(block)
		}
	}
	return 0
}

// walkFiles calls fn for every files in filestore in parallel
func (f *FileStore) walkFiles(fn func(f TSMFile) error) error {
	// Copy the current TSM files to prevent a slow walker from
	// blocking other operations.
	f.mu.RLock()
	files := make([]TSMFile, len(f.files))
	copy(files, f.files)
	f.mu.RUnlock()

	// struct to hold the result of opening each reader in a goroutine
	errC := make(chan error, len(files))
	for _, f := range files {
		go func(tsm TSMFile) {
			if err := fn(tsm); err != nil {
				errC <- fmt.Errorf("file %s: %s", tsm.Path(), err)
				return
			}

			errC <- nil
		}(f)
	}

	for i := 0; i < cap(errC); i++ {
		res := <-errC
		if res != nil {
			return res
		}
	}
	return nil
}

// locations returns the files and index blocks for a key and time.  ascending indicates
// whether the key will be scan in ascending time order or descenging time order.
// This function assumes the read-lock has been taken.
func (f *FileStore) locations(key string, t int64, ascending bool) []*location {
	filesSnapshot := make([]TSMFile, len(f.files))
	for i := range f.files {
		filesSnapshot[i] = f.files[i]
	}

	var entries []IndexEntry
	locations := make([]*location, 0, len(filesSnapshot))
	for _, fd := range filesSnapshot {
		minTime, maxTime := fd.TimeRange()

		tombstones := fd.TombstoneRange(key)
		// If we ascending and the max time of the file is before where we want to start
		// skip it.
		if ascending && maxTime < t {
			continue
			// If we are descending and the min time of the file is after where we want to start,
			// then skip it.
		} else if !ascending && minTime > t {
			continue
		}

		// This file could potential contain points we are looking for so find the blocks for
		// the given key.
		fd.ReadEntries(key, &entries)
		for _, ie := range entries {

			// Skip any blocks only contain values that are tombstoned.
			var skip bool
			for _, t := range tombstones {
				if t.Min <= ie.MinTime && t.Max >= ie.MaxTime {
					skip = true
					break
				}
			}

			if skip {
				continue
			}
			// If we ascending and the max time of a block is before where we are looking, skip
			// it since the data is out of our range
			if ascending && ie.MaxTime < t {
				continue
				// If we descending and the min time of a block is after where we are looking, skip
				// it since the data is out of our range
			} else if !ascending && ie.MinTime > t {
				continue
			}

			location := &location{
				r:     fd,
				entry: ie,
			}

			if ascending {
				// For an ascending cursor, mark everything before the seek time as read
				// so we can filter it out at query time
				location.readMin = math.MinInt64
				location.readMax = t - 1
			} else {
				// For an ascending cursort, mark everything after the seek time as read
				// so we can filter it out at query time
				location.readMin = t + 1
				location.readMax = math.MaxInt64
			}
			// Otherwise, add this file and block location
			locations = append(locations, location)
		}
	}
	return locations
}

// CreateSnapshot will create hardlinks for all tsm and tombstone files
// in the path provided
func (f *FileStore) CreateSnapshot() (string, error) {
	f.traceLogger.Printf("Creating snapshot in %s", f.dir)
	files := f.Files()

	f.mu.Lock()
	f.currentTempDirID += 1
	f.mu.Unlock()

	f.mu.RLock()
	defer f.mu.RUnlock()

	// get a tmp directory name
	tmpPath := fmt.Sprintf("%s/%d.tmp", f.dir, f.currentTempDirID)
	err := os.Mkdir(tmpPath, 0777)
	if err != nil {
		return "", err
	}

	for _, tsmf := range files {
		newpath := filepath.Join(tmpPath, filepath.Base(tsmf.Path()))
		if err := os.Link(tsmf.Path(), newpath); err != nil {
			return "", fmt.Errorf("error creating tsm hard link: %q", err)
		}
		// Check for tombstones and link those as well
		for _, tf := range tsmf.TombstoneFiles() {
			newpath := filepath.Join(tmpPath, filepath.Base(tf.Path))
			if err := os.Link(tf.Path, newpath); err != nil {
				return "", fmt.Errorf("error creating tombstone hard link: %q", err)
			}
		}
	}

	return tmpPath, nil
}

// ParseTSMFileName parses the generation and sequence from a TSM file name.
func ParseTSMFileName(name string) (int, int, error) {
	base := filepath.Base(name)
	idx := strings.Index(base, ".")
	if idx == -1 {
		return 0, 0, fmt.Errorf("file %s is named incorrectly", name)
	}

	id := base[:idx]

	idx = strings.Index(id, "-")
	if idx == -1 {
		return 0, 0, fmt.Errorf("file %s is named incorrectly", name)
	}

	generation, err := strconv.ParseUint(id[:idx], 10, 32)
	sequence, err := strconv.ParseUint(id[idx+1:], 10, 32)

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

	// duplicates is a hint that there are overlapping blocks for this key in
	// multiple files (e.g. points have been overwritten but not fully compacted)
	// If this is true, we need to scan the duplicate blocks and dedup the points
	// as query time until they are compacted.
	duplicates bool

	// The distinct set of TSM files references by the cursor
	refs map[string]TSMFile
}

type location struct {
	r     TSMFile
	entry IndexEntry

	readMin, readMax int64
}

func (l *location) read() bool {
	return l.readMin <= l.entry.MinTime && l.readMax >= l.entry.MaxTime
}

func (l *location) markRead(min, max int64) {
	if min < l.readMin {
		l.readMin = min
	}

	if max > l.readMax {
		l.readMax = max
	}
}

type descLocations []*location

// Sort methods
func (a descLocations) Len() int      { return len(a) }
func (a descLocations) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a descLocations) Less(i, j int) bool {
	if a[i].entry.OverlapsTimeRange(a[j].entry.MinTime, a[j].entry.MaxTime) {
		return a[i].r.Path() < a[j].r.Path()
	}
	return a[i].entry.MaxTime < a[j].entry.MaxTime
}

type ascLocations []*location

// Sort methods
func (a ascLocations) Len() int      { return len(a) }
func (a ascLocations) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ascLocations) Less(i, j int) bool {
	if a[i].entry.OverlapsTimeRange(a[j].entry.MinTime, a[j].entry.MaxTime) {
		return a[i].r.Path() < a[j].r.Path()
	}
	return a[i].entry.MinTime < a[j].entry.MinTime
}

// newKeyCursor returns a new instance of KeyCursor.
// This function assumes the read-lock has been taken.
func newKeyCursor(fs *FileStore, key string, t int64, ascending bool) *KeyCursor {
	c := &KeyCursor{
		key:       key,
		fs:        fs,
		seeks:     fs.locations(key, t, ascending),
		ascending: ascending,
	}
	c.refs = make(map[string]TSMFile, len(c.seeks))

	c.duplicates = c.hasOverlappingBlocks()

	if ascending {
		sort.Sort(ascLocations(c.seeks))
	} else {
		sort.Sort(descLocations(c.seeks))
	}

	// Determine the distinct set of TSM files in use and mark then as in-use
	for _, f := range c.seeks {
		if _, ok := c.refs[f.r.Path()]; !ok {
			f.r.Ref()
			c.refs[f.r.Path()] = f.r
		}
	}

	c.seek(t)
	return c
}

// Close removes all references on the cursor.
func (c *KeyCursor) Close() {
	// Remove all of our in-use references since we're done
	for _, f := range c.refs {
		f.Unref()
	}

	c.buf = nil
	c.seeks = nil
	c.fs = nil
	c.current = nil
}

// hasOverlappingBlocks returns true if blocks have overlapping time ranges.
// This result is computed once and stored as the "duplicates" field.
func (c *KeyCursor) hasOverlappingBlocks() bool {
	if len(c.seeks) == 0 {
		return false
	}

	for i := 1; i < len(c.seeks); i++ {
		prev := c.seeks[i-1]
		cur := c.seeks[i]
		if prev.entry.MaxTime >= cur.entry.MinTime {
			return true
		}
	}
	return false
}

// seek positions the cursor at the given time.
func (c *KeyCursor) seek(t int64) {
	if len(c.seeks) == 0 {
		return
	}
	c.current = nil

	if c.ascending {
		c.seekAscending(t)
	} else {
		c.seekDescending(t)
	}
}

func (c *KeyCursor) seekAscending(t int64) {
	for i, e := range c.seeks {
		if t < e.entry.MinTime || e.entry.Contains(t) {
			// Record the position of the first block matching our seek time
			if len(c.current) == 0 {
				c.pos = i
			}

			c.current = append(c.current, e)

			// Exit if we don't have duplicates.
			// Otherwise, keep looking for additional blocks containing this point.
			if !c.duplicates {
				return
			}
		}
	}
}

func (c *KeyCursor) seekDescending(t int64) {
	for i := len(c.seeks) - 1; i >= 0; i-- {
		e := c.seeks[i]
		if t > e.entry.MaxTime || e.entry.Contains(t) {
			// Record the position of the first block matching our seek time
			if len(c.current) == 0 {
				c.pos = i
			}
			c.current = append(c.current, e)

			// Exit if we don't have duplicates.
			// Otherwise, keep looking for additional blocks containing this point.
			if !c.duplicates {
				return
			}
		}
	}
}

// Next moves the cursor to the next position.
// Data should be read by the ReadBlock functions.
func (c *KeyCursor) Next() {
	if len(c.current) == 0 {
		return
	}
	// Do we still have unread values in the current block
	if !c.current[0].read() {
		return
	}
	c.current = c.current[:0]
	if c.ascending {
		c.nextAscending()
	} else {
		c.nextDescending()
	}
}

func (c *KeyCursor) nextAscending() {
	for {
		c.pos++
		if c.pos >= len(c.seeks) {
			return
		} else if !c.seeks[c.pos].read() {
			break
		}
	}

	// Append the first matching block
	if len(c.current) == 0 {
		c.current = append(c.current, nil)
	} else {
		c.current = c.current[:1]
	}
	c.current[0] = c.seeks[c.pos]

	// We're done if there are no overlapping blocks.
	if !c.duplicates {
		return
	}

	// If we have ovelapping blocks, append all their values so we can dedup
	for i := c.pos + 1; i < len(c.seeks); i++ {
		if c.seeks[i].read() {
			continue
		}

		c.current = append(c.current, c.seeks[i])
	}
}

func (c *KeyCursor) nextDescending() {
	for {
		c.pos--
		if c.pos < 0 {
			return
		} else if !c.seeks[c.pos].read() {
			break
		}
	}

	// Append the first matching block
	if len(c.current) == 0 {
		c.current = make([]*location, 1)
	} else {
		c.current = c.current[:1]
	}
	c.current[0] = c.seeks[c.pos]

	// We're done if there are no overlapping blocks.
	if !c.duplicates {
		return
	}

	// If we have ovelapping blocks, append all their values so we can dedup
	for i := c.pos; i >= 0; i-- {
		if c.seeks[i].read() {
			continue
		}
		c.current = append(c.current, c.seeks[i])
	}
}

func (c *KeyCursor) filterFloatValues(tombstones []TimeRange, values FloatValues) FloatValues {
	for _, t := range tombstones {
		values = values.Exclude(t.Min, t.Max)
	}
	return values
}

func (c *KeyCursor) filterIntegerValues(tombstones []TimeRange, values IntegerValues) IntegerValues {
	for _, t := range tombstones {
		values = values.Exclude(t.Min, t.Max)
	}
	return values
}

func (c *KeyCursor) filterStringValues(tombstones []TimeRange, values StringValues) StringValues {
	for _, t := range tombstones {
		values = values.Exclude(t.Min, t.Max)
	}
	return values
}

func (c *KeyCursor) filterBooleanValues(tombstones []TimeRange, values BooleanValues) BooleanValues {
	for _, t := range tombstones {
		values = values.Exclude(t.Min, t.Max)
	}
	return values
}

type purger struct {
	mu        sync.RWMutex
	fileStore *FileStore
	files     map[string]TSMFile
	running   bool

	logger *log.Logger
}

func (p *purger) add(files []TSMFile) {
	p.mu.Lock()
	for _, f := range files {
		p.files[f.Path()] = f
	}
	p.mu.Unlock()
	p.purge()
}

func (p *purger) purge() {
	p.mu.Lock()
	if p.running {
		p.mu.Unlock()
		return
	}
	p.running = true
	p.mu.Unlock()

	go func() {
		for {
			p.mu.Lock()
			for k, v := range p.files {
				if !v.InUse() {
					// Remove any mmap references held by the index.
					if p.fileStore.dereferencer != nil {
						v.deref(p.fileStore.dereferencer)
					}

					if err := v.Close(); err != nil {
						p.logger.Printf("purge: close file: %v", err)
						continue
					}

					if err := v.Remove(); err != nil {
						p.logger.Printf("purge: remove file: %v", err)
						continue
					}
					delete(p.files, k)
				}
			}

			if len(p.files) == 0 {
				p.running = false
				p.mu.Unlock()
				return
			}

			p.mu.Unlock()
			time.Sleep(time.Second)
		}
	}()
}

type tsmReaders []TSMFile

func (a tsmReaders) Len() int           { return len(a) }
func (a tsmReaders) Less(i, j int) bool { return a[i].Path() < a[j].Path() }
func (a tsmReaders) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
