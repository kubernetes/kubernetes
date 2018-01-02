package tsm1 // import "github.com/influxdata/influxdb/tsdb/engine/tsm1"

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/tsdb"
)

//go:generate tmpl -data=@iterator.gen.go.tmpldata iterator.gen.go.tmpl
//go:generate tmpl -data=@file_store.gen.go.tmpldata file_store.gen.go.tmpl
//go:generate tmpl -data=@encoding.gen.go.tmpldata encoding.gen.go.tmpl

func init() {
	tsdb.RegisterEngine("tsm1", NewEngine)
}

// Ensure Engine implements the interface.
var _ tsdb.Engine = &Engine{}

const (
	// keyFieldSeparator separates the series key from the field name in the composite key
	// that identifies a specific field in series
	keyFieldSeparator = "#!~#"
)

// Statistics gathered by the engine.
const (
	statCacheCompactions        = "cacheCompactions"
	statCacheCompactionsActive  = "cacheCompactionsActive"
	statCacheCompactionError    = "cacheCompactionErr"
	statCacheCompactionDuration = "cacheCompactionDuration"

	statTSMLevel1Compactions        = "tsmLevel1Compactions"
	statTSMLevel1CompactionsActive  = "tsmLevel1CompactionsActive"
	statTSMLevel1CompactionError    = "tsmLevel1CompactionErr"
	statTSMLevel1CompactionDuration = "tsmLevel1CompactionDuration"

	statTSMLevel2Compactions        = "tsmLevel2Compactions"
	statTSMLevel2CompactionsActive  = "tsmLevel2CompactionsActive"
	statTSMLevel2CompactionError    = "tsmLevel2CompactionErr"
	statTSMLevel2CompactionDuration = "tsmLevel2CompactionDuration"

	statTSMLevel3Compactions        = "tsmLevel3Compactions"
	statTSMLevel3CompactionsActive  = "tsmLevel3CompactionsActive"
	statTSMLevel3CompactionError    = "tsmLevel3CompactionErr"
	statTSMLevel3CompactionDuration = "tsmLevel3CompactionDuration"

	statTSMOptimizeCompactions        = "tsmOptimizeCompactions"
	statTSMOptimizeCompactionsActive  = "tsmOptimizeCompactionsActive"
	statTSMOptimizeCompactionError    = "tsmOptimizeCompactionErr"
	statTSMOptimizeCompactionDuration = "tsmOptimizeCompactionDuration"

	statTSMFullCompactions        = "tsmFullCompactions"
	statTSMFullCompactionsActive  = "tsmFullCompactionsActive"
	statTSMFullCompactionError    = "tsmFullCompactionErr"
	statTSMFullCompactionDuration = "tsmFullCompactionDuration"
)

// Engine represents a storage engine with compressed blocks.
type Engine struct {
	mu sync.RWMutex

	// The following group of fields is used to track the state of level compactions within the
	// Engine. The WaitGroup is used to monitor the compaction goroutines, the 'done' channel is
	// used to signal those goroutines to shutdown. Every request to disable level compactions will
	// call 'Wait' on 'wg', with the first goroutine to arrive (levelWorkers == 0 while holding the
	// lock) will close the done channel and re-assign 'nil' to the variable. Re-enabling will
	// decrease 'levelWorkers', and when it decreases to zero, level compactions will be started
	// back up again.

	wg           sync.WaitGroup // waitgroup for active level compaction goroutines
	done         chan struct{}  // channel to signal level compactions to stop
	levelWorkers int            // Number of "workers" that expect compactions to be in a disabled state

	snapDone chan struct{}  // channel to signal snapshot compactions to stop
	snapWG   sync.WaitGroup // waitgroup for running snapshot compactions

	id           uint64
	path         string
	logger       *log.Logger // Logger to be used for important messages
	traceLogger  *log.Logger // Logger to be used when trace-logging is on.
	logOutput    io.Writer   // Writer to be logger and traceLogger if active.
	traceLogging bool

	// TODO(benbjohnson): Index needs to be moved entirely into engine.
	index             *tsdb.DatabaseIndex
	measurementFields map[string]*tsdb.MeasurementFields

	WAL            *WAL
	Cache          *Cache
	Compactor      *Compactor
	CompactionPlan CompactionPlanner
	FileStore      *FileStore

	MaxPointsPerBlock int

	// CacheFlushMemorySizeThreshold specifies the minimum size threshodl for
	// the cache when the engine should write a snapshot to a TSM file
	CacheFlushMemorySizeThreshold uint64

	// CacheFlushWriteColdDuration specifies the length of time after which if
	// no writes have been committed to the WAL, the engine will write
	// a snapshot of the cache to a TSM file
	CacheFlushWriteColdDuration time.Duration

	// Controls whether to enabled compactions when the engine is open
	enableCompactionsOnOpen bool

	stats *EngineStatistics
}

// NewEngine returns a new instance of Engine.
func NewEngine(id uint64, path string, walPath string, opt tsdb.EngineOptions) tsdb.Engine {
	w := NewWAL(walPath)
	fs := NewFileStore(path)
	cache := NewCache(uint64(opt.Config.CacheMaxMemorySize), path)

	c := &Compactor{
		Dir:       path,
		FileStore: fs,
	}

	e := &Engine{
		id:           id,
		path:         path,
		logger:       log.New(os.Stderr, "[tsm1] ", log.LstdFlags),
		traceLogger:  log.New(ioutil.Discard, "[tsm1] ", log.LstdFlags),
		logOutput:    os.Stderr,
		traceLogging: opt.Config.TraceLoggingEnabled,

		measurementFields: make(map[string]*tsdb.MeasurementFields),

		WAL:   w,
		Cache: cache,

		FileStore: fs,
		Compactor: c,
		CompactionPlan: &DefaultPlanner{
			FileStore:                    fs,
			CompactFullWriteColdDuration: time.Duration(opt.Config.CompactFullWriteColdDuration),
		},

		CacheFlushMemorySizeThreshold: opt.Config.CacheSnapshotMemorySize,
		CacheFlushWriteColdDuration:   time.Duration(opt.Config.CacheSnapshotWriteColdDuration),
		enableCompactionsOnOpen:       true,
		stats: &EngineStatistics{},
	}

	if e.traceLogging {
		e.traceLogger.SetOutput(e.logOutput)
		fs.enableTraceLogging(true)
		w.enableTraceLogging(true)
	}

	return e
}

func (e *Engine) SetEnabled(enabled bool) {
	e.enableCompactionsOnOpen = enabled
	e.SetCompactionsEnabled(enabled)
}

// SetCompactionsEnabled enables compactions on the engine.  When disabled
// all running compactions are aborted and new compactions stop running.
func (e *Engine) SetCompactionsEnabled(enabled bool) {
	if enabled {
		e.enableSnapshotCompactions()
		e.enableLevelCompactions(false)
	} else {
		e.disableSnapshotCompactions()
		e.disableLevelCompactions(false)
	}
}

// enableLevelCompactions will request that level compactions start back up again
//
// 'wait' signifies that a corresponding call to disableLevelCompactions(true) was made at some
// point, and the associated task that required disabled compactions is now complete
func (e *Engine) enableLevelCompactions(wait bool) {
	e.mu.Lock()
	if wait {
		e.levelWorkers -= 1
	}
	if e.levelWorkers != 0 || e.done != nil {
		// still waiting on more workers or already enabled
		e.mu.Unlock()
		return
	}

	// last one to enable, start things back up
	e.Compactor.EnableCompactions()
	quit := make(chan struct{})
	e.done = quit

	e.wg.Add(4)
	e.mu.Unlock()

	go func() { defer e.wg.Done(); e.compactTSMFull(quit) }()
	go func() { defer e.wg.Done(); e.compactTSMLevel(true, 1, quit) }()
	go func() { defer e.wg.Done(); e.compactTSMLevel(true, 2, quit) }()
	go func() { defer e.wg.Done(); e.compactTSMLevel(false, 3, quit) }()
}

// disableLevelCompactions will stop level compactions before returning
//
// If 'wait' is set to true, then a corresponding call to enableLevelCompactions(true) will be
// required before level compactions will start back up again
func (e *Engine) disableLevelCompactions(wait bool) {
	e.mu.Lock()
	old := e.levelWorkers
	if wait {
		e.levelWorkers += 1
	}

	if old == 0 && e.done != nil {
		// Prevent new compactions from starting
		e.Compactor.DisableCompactions()

		// Stop all background compaction goroutines
		close(e.done)
		e.done = nil
	}

	e.mu.Unlock()
	e.wg.Wait()

	if old == 0 { // first to disable should cleanup
		if err := e.cleanup(); err != nil {
			e.logger.Printf("error cleaning up temp file: %v", err)
		}
	}
}

func (e *Engine) enableSnapshotCompactions() {
	e.mu.Lock()
	if e.snapDone != nil {
		e.mu.Unlock()
		return
	}

	e.Compactor.EnableSnapshots()
	quit := make(chan struct{})
	e.snapDone = quit
	e.snapWG.Add(1)
	e.mu.Unlock()

	go func() { defer e.snapWG.Done(); e.compactCache(quit) }()
}

func (e *Engine) disableSnapshotCompactions() {
	e.mu.Lock()

	if e.snapDone != nil {
		e.Compactor.DisableSnapshots()
		close(e.snapDone)
		e.snapDone = nil
	}

	e.mu.Unlock()
	e.snapWG.Wait()
}

// Path returns the path the engine was opened with.
func (e *Engine) Path() string { return e.path }

// Index returns the database index.
func (e *Engine) Index() *tsdb.DatabaseIndex {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.index
}

// MeasurementFields returns the measurement fields for a measurement.
func (e *Engine) MeasurementFields(measurement string) *tsdb.MeasurementFields {
	e.mu.RLock()
	m := e.measurementFields[measurement]
	e.mu.RUnlock()

	if m != nil {
		return m
	}

	e.mu.Lock()
	m = e.measurementFields[measurement]
	if m == nil {
		m = tsdb.NewMeasurementFields()
		e.measurementFields[measurement] = m
	}
	e.mu.Unlock()
	return m
}

// Format returns the format type of this engine
func (e *Engine) Format() tsdb.EngineFormat {
	return tsdb.TSM1Format
}

// EngineStatistics maintains statistics for the engine.
type EngineStatistics struct {
	CacheCompactions        int64 // Counter of cache compactions that have ever run.
	CacheCompactionsActive  int64 // Gauge of cache compactions currently running.
	CacheCompactionErrors   int64 // Counter of cache compactions that have failed due to error.
	CacheCompactionDuration int64 // Counter of number of wall nanoseconds spent in cache compactions.

	TSMCompactions        [3]int64 // Counter of TSM compactions (by level) that have ever run.
	TSMCompactionsActive  [3]int64 // Gauge of TSM compactions (by level) currently running.
	TSMCompactionErrors   [3]int64 // Counter of TSM compcations (by level) that have failed due to error.
	TSMCompactionDuration [3]int64 // Counter of number of wall nanoseconds spent in TSM compactions (by level).

	TSMOptimizeCompactions        int64 // Counter of optimize compactions that have ever run.
	TSMOptimizeCompactionsActive  int64 // Gauge of optimize compactions currently running.
	TSMOptimizeCompactionErrors   int64 // Counter of optimize compactions that have failed due to error.
	TSMOptimizeCompactionDuration int64 // Counter of number of wall nanoseconds spent in optimize compactions.

	TSMFullCompactions        int64 // Counter of full compactions that have ever run.
	TSMFullCompactionsActive  int64 // Gauge of full compactions currently running.
	TSMFullCompactionErrors   int64 // Counter of full compactions that have failed due to error.
	TSMFullCompactionDuration int64 // Counter of number of wall nanoseconds spent in full compactions.
}

// Statistics returns statistics for periodic monitoring.
func (e *Engine) Statistics(tags map[string]string) []models.Statistic {
	statistics := make([]models.Statistic, 0, 4)
	statistics = append(statistics, models.Statistic{
		Name: "tsm1_engine",
		Tags: tags,
		Values: map[string]interface{}{
			statCacheCompactions:        atomic.LoadInt64(&e.stats.CacheCompactions),
			statCacheCompactionsActive:  atomic.LoadInt64(&e.stats.CacheCompactionsActive),
			statCacheCompactionError:    atomic.LoadInt64(&e.stats.CacheCompactionErrors),
			statCacheCompactionDuration: atomic.LoadInt64(&e.stats.CacheCompactionDuration),

			statTSMLevel1Compactions:        atomic.LoadInt64(&e.stats.TSMCompactions[0]),
			statTSMLevel1CompactionsActive:  atomic.LoadInt64(&e.stats.TSMCompactionsActive[0]),
			statTSMLevel1CompactionError:    atomic.LoadInt64(&e.stats.TSMCompactionErrors[0]),
			statTSMLevel1CompactionDuration: atomic.LoadInt64(&e.stats.TSMCompactionDuration[0]),

			statTSMLevel2Compactions:        atomic.LoadInt64(&e.stats.TSMCompactions[1]),
			statTSMLevel2CompactionsActive:  atomic.LoadInt64(&e.stats.TSMCompactionsActive[1]),
			statTSMLevel2CompactionError:    atomic.LoadInt64(&e.stats.TSMCompactionErrors[1]),
			statTSMLevel2CompactionDuration: atomic.LoadInt64(&e.stats.TSMCompactionDuration[1]),

			statTSMLevel3Compactions:        atomic.LoadInt64(&e.stats.TSMCompactions[2]),
			statTSMLevel3CompactionsActive:  atomic.LoadInt64(&e.stats.TSMCompactionsActive[2]),
			statTSMLevel3CompactionError:    atomic.LoadInt64(&e.stats.TSMCompactionErrors[2]),
			statTSMLevel3CompactionDuration: atomic.LoadInt64(&e.stats.TSMCompactionDuration[2]),

			statTSMOptimizeCompactions:        atomic.LoadInt64(&e.stats.TSMOptimizeCompactions),
			statTSMOptimizeCompactionsActive:  atomic.LoadInt64(&e.stats.TSMOptimizeCompactionsActive),
			statTSMOptimizeCompactionError:    atomic.LoadInt64(&e.stats.TSMOptimizeCompactionErrors),
			statTSMOptimizeCompactionDuration: atomic.LoadInt64(&e.stats.TSMOptimizeCompactionDuration),

			statTSMFullCompactions:        atomic.LoadInt64(&e.stats.TSMFullCompactions),
			statTSMFullCompactionsActive:  atomic.LoadInt64(&e.stats.TSMFullCompactionsActive),
			statTSMFullCompactionError:    atomic.LoadInt64(&e.stats.TSMFullCompactionErrors),
			statTSMFullCompactionDuration: atomic.LoadInt64(&e.stats.TSMFullCompactionDuration),
		},
	})
	statistics = append(statistics, e.Cache.Statistics(tags)...)
	statistics = append(statistics, e.FileStore.Statistics(tags)...)
	statistics = append(statistics, e.WAL.Statistics(tags)...)
	return statistics
}

// Open opens and initializes the engine.
func (e *Engine) Open() error {
	if err := os.MkdirAll(e.path, 0777); err != nil {
		return err
	}

	if err := e.cleanup(); err != nil {
		return err
	}

	if err := e.WAL.Open(); err != nil {
		return err
	}

	if err := e.FileStore.Open(); err != nil {
		return err
	}

	if err := e.reloadCache(); err != nil {
		return err
	}

	e.Compactor.Open()

	if e.enableCompactionsOnOpen {
		e.SetCompactionsEnabled(true)
	}

	return nil
}

// Close closes the engine. Subsequent calls to Close are a nop.
func (e *Engine) Close() error {
	e.SetCompactionsEnabled(false)

	// Lock now and close everything else down.
	e.mu.Lock()
	defer e.mu.Unlock()
	e.done = nil // Ensures that the channel will not be closed again.

	if err := e.FileStore.Close(); err != nil {
		return err
	}
	return e.WAL.Close()
}

// SetLogOutput sets the logger used for all messages. It is safe for concurrent
// use.
func (e *Engine) SetLogOutput(w io.Writer) {
	e.logger.SetOutput(w)

	// Set the trace logger's output only if trace logging is enabled.
	if e.traceLogging {
		e.traceLogger.SetOutput(w)
	}

	e.WAL.SetLogOutput(w)
	e.FileStore.SetLogOutput(w)

	e.mu.Lock()
	e.logOutput = w
	e.mu.Unlock()
}

// LoadMetadataIndex loads the shard metadata into memory.
func (e *Engine) LoadMetadataIndex(shardID uint64, index *tsdb.DatabaseIndex) error {
	now := time.Now()

	// Save reference to index for iterator creation.
	e.index = index
	e.FileStore.dereferencer = index

	if err := e.FileStore.WalkKeys(func(key []byte, typ byte) error {
		fieldType, err := tsmFieldTypeToInfluxQLDataType(typ)
		if err != nil {
			return err
		}

		if err := e.addToIndexFromKey(shardID, key, fieldType, index); err != nil {
			return err
		}
		return nil
	}); err != nil {
		return err
	}

	// load metadata from the Cache
	e.Cache.RLock() // shouldn't need the lock, but just to be safe
	defer e.Cache.RUnlock()

	for key, entry := range e.Cache.Store() {

		fieldType, err := entry.values.InfluxQLType()
		if err != nil {
			e.logger.Printf("error getting the data type of values for key %s: %s", key, err.Error())
			continue
		}

		if err := e.addToIndexFromKey(shardID, []byte(key), fieldType, index); err != nil {
			return err
		}
	}

	e.traceLogger.Printf("Meta data index for shard %d loaded in %v", shardID, time.Since(now))
	return nil
}

// Backup will write a tar archive of any TSM files modified since the passed
// in time to the passed in writer. The basePath will be prepended to the names
// of the files in the archive. It will force a snapshot of the WAL first
// then perform the backup with a read lock against the file store. This means
// that new TSM files will not be able to be created in this shard while the
// backup is running. For shards that are still acively getting writes, this
// could cause the WAL to backup, increasing memory usage and evenutally rejecting writes.
func (e *Engine) Backup(w io.Writer, basePath string, since time.Time) error {
	path, err := e.CreateSnapshot()
	if err != nil {
		return err
	}

	// Remove the temporary snapshot dir
	defer os.RemoveAll(path)

	snapDir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer snapDir.Close()

	snapshotFiles, err := snapDir.Readdir(0)
	if err != nil {
		return err
	}

	var files []os.FileInfo
	// grab all the files and tombstones that have a modified time after since
	for _, f := range snapshotFiles {
		if f.ModTime().UnixNano() > since.UnixNano() {
			files = append(files, f)
		}
	}

	if len(files) == 0 {
		return nil
	}

	tw := tar.NewWriter(w)
	defer tw.Close()

	for _, f := range files {
		if err := e.writeFileToBackup(f, basePath, filepath.Join(path, f.Name()), tw); err != nil {
			return err
		}
	}

	return nil
}

// writeFileToBackup will copy the file into the tar archive. Files will use the shardRelativePath
// in their names. This should be the <db>/<retention policy>/<id> part of the path
func (e *Engine) writeFileToBackup(f os.FileInfo, shardRelativePath, fullPath string, tw *tar.Writer) error {
	h := &tar.Header{
		Name:    filepath.Join(shardRelativePath, f.Name()),
		ModTime: f.ModTime(),
		Size:    f.Size(),
		Mode:    int64(f.Mode()),
	}
	if err := tw.WriteHeader(h); err != nil {
		return err
	}
	fr, err := os.Open(fullPath)
	if err != nil {
		return err
	}

	defer fr.Close()

	_, err = io.CopyN(tw, fr, h.Size)

	return err
}

// Restore will read a tar archive generated by Backup().
// Only files that match basePath will be copied into the directory. This obtains
// a write lock so no operations can be performed while restoring.
func (e *Engine) Restore(r io.Reader, basePath string) error {
	// Copy files from archive while under lock to prevent reopening.
	if err := func() error {
		e.mu.Lock()
		defer e.mu.Unlock()

		tr := tar.NewReader(r)
		for {
			if err := e.readFileFromBackup(tr, basePath); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
		}

		return syncDir(e.path)
	}(); err != nil {
		return err
	}

	return nil
}

// readFileFromBackup copies the next file from the archive into the shard.
// The file is skipped if it does not have a matching shardRelativePath prefix.
func (e *Engine) readFileFromBackup(tr *tar.Reader, shardRelativePath string) error {
	// Read next archive file.
	hdr, err := tr.Next()
	if err != nil {
		return err
	}

	// Skip file if it does not have a matching prefix.
	if !filepath.HasPrefix(hdr.Name, shardRelativePath) {
		return nil
	}
	path, err := filepath.Rel(shardRelativePath, hdr.Name)
	if err != nil {
		return err
	}

	destPath := filepath.Join(e.path, path)
	tmp := destPath + ".tmp"

	// Create new file on disk.
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	// Copy from archive to the file.
	if _, err := io.CopyN(f, tr, hdr.Size); err != nil {
		return err
	}

	// Sync to disk & close.
	if err := f.Sync(); err != nil {
		return err
	}

	if err := f.Close(); err != nil {
		return err
	}

	return renameFile(tmp, destPath)
}

// addToIndexFromKey will pull the measurement name, series key, and field name from a composite key and add it to the
// database index and measurement fields
func (e *Engine) addToIndexFromKey(shardID uint64, key []byte, fieldType influxql.DataType, index *tsdb.DatabaseIndex) error {
	seriesKey, field := SeriesAndFieldFromCompositeKey(key)
	measurement := tsdb.MeasurementFromSeriesKey(string(seriesKey))

	m := index.CreateMeasurementIndexIfNotExists(measurement)
	m.SetFieldName(field)

	mf := e.measurementFields[measurement]
	if mf == nil {
		mf = tsdb.NewMeasurementFields()
		e.measurementFields[measurement] = mf
	}

	if err := mf.CreateFieldIfNotExists(field, fieldType, false); err != nil {
		return err
	}

	// Have we already indexed this series?
	ss := index.SeriesBytes(seriesKey)
	if ss != nil {
		// Add this shard to the existing series
		ss.AssignShard(shardID)
		return nil
	}

	// ignore error because ParseKey returns "missing fields" and we don't have
	// fields (in line protocol format) in the series key
	_, tags, _ := models.ParseKey(seriesKey)

	s := tsdb.NewSeries(string(seriesKey), tags)
	index.CreateSeriesIndexIfNotExists(measurement, s)
	s.AssignShard(shardID)

	return nil
}

// WritePoints writes metadata and point data into the engine.
// Returns an error if new points are added to an existing key.
func (e *Engine) WritePoints(points []models.Point) error {
	values := make(map[string][]Value, len(points))
	var keyBuf []byte
	var baseLen int
	for _, p := range points {
		keyBuf = append(keyBuf[:0], p.Key()...)
		keyBuf = append(keyBuf, keyFieldSeparator...)
		baseLen = len(keyBuf)
		iter := p.FieldIterator()
		t := p.Time().UnixNano()
		for iter.Next() {
			keyBuf = append(keyBuf[:baseLen], iter.FieldKey()...)
			var v Value
			switch iter.Type() {
			case models.Float:
				v = NewFloatValue(t, iter.FloatValue())
			case models.Integer:
				v = NewIntegerValue(t, iter.IntegerValue())
			case models.String:
				v = NewStringValue(t, iter.StringValue())
			case models.Boolean:
				v = NewBooleanValue(t, iter.BooleanValue())
			default:
				return fmt.Errorf("unknown field type for %s: %s", string(iter.FieldKey()), p.String())
			}
			values[string(keyBuf)] = append(values[string(keyBuf)], v)
		}
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// first try to write to the cache
	err := e.Cache.WriteMulti(values)
	if err != nil {
		return err
	}

	_, err = e.WAL.WritePoints(values)
	return err
}

// ContainsSeries returns a map of keys indicating whether the key exists and
// has values or not.
func (e *Engine) ContainsSeries(keys []string) (map[string]bool, error) {
	// keyMap is used to see if a given key exists.  keys
	// are the measurement + tagset (minus separate & field)
	keyMap := map[string]bool{}
	for _, k := range keys {
		keyMap[k] = false
	}

	for _, k := range e.Cache.Keys() {
		seriesKey, _ := SeriesAndFieldFromCompositeKey([]byte(k))
		keyMap[string(seriesKey)] = true
	}

	if err := e.FileStore.WalkKeys(func(k []byte, _ byte) error {
		seriesKey, _ := SeriesAndFieldFromCompositeKey(k)
		if _, ok := keyMap[string(seriesKey)]; ok {
			keyMap[string(seriesKey)] = true
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return keyMap, nil
}

// DeleteSeries removes all series keys from the engine.
func (e *Engine) DeleteSeries(seriesKeys []string) error {
	return e.DeleteSeriesRange(seriesKeys, math.MinInt64, math.MaxInt64)
}

// DeleteSeriesRange removes the values between min and max (inclusive) from all series.
func (e *Engine) DeleteSeriesRange(seriesKeys []string, min, max int64) error {
	if len(seriesKeys) == 0 {
		return nil
	}

	// Disable and abort running compactions so that tombstones added existing tsm
	// files don't get removed.  This would cause deleted measurements/series to
	// re-appear once the compaction completed.  We only disable the level compactions
	// so that snapshotting does not stop while writing out tombstones.  If it is stopped,
	// and writing tombstones takes a long time, writes can get rejected due to the cache
	// filling up.
	e.disableLevelCompactions(true)
	defer e.enableLevelCompactions(true)

	// keyMap is used to see if a given key should be deleted.  seriesKey
	// are the measurement + tagset (minus separate & field)
	keyMap := make(map[string]struct{}, len(seriesKeys))
	for _, k := range seriesKeys {
		keyMap[k] = struct{}{}
	}

	deleteKeys := make([]string, 0, len(seriesKeys))
	// go through the keys in the file store
	if err := e.FileStore.WalkKeys(func(k []byte, _ byte) error {
		seriesKey, _ := SeriesAndFieldFromCompositeKey(k)
		// Keep track if we've added this key since WalkKeys can return keys
		// we've seen before
		key := string(k)
		if _, ok := keyMap[string(seriesKey)]; ok {
			i := sort.SearchStrings(deleteKeys, key)
			if i == len(deleteKeys) {
				deleteKeys = append(deleteKeys, key)
			} else if key != deleteKeys[i] {
				deleteKeys = append(deleteKeys, key)
				copy(deleteKeys[i+1:], deleteKeys[i:])
				deleteKeys[i] = key
			}
		}
		return nil
	}); err != nil {
		return err
	}

	if err := e.FileStore.DeleteRange(deleteKeys, min, max); err != nil {
		return err
	}

	// find the keys in the cache and remove them
	walKeys := deleteKeys[:0]
	e.Cache.RLock()
	s := e.Cache.Store()
	for k, _ := range s {
		seriesKey, _ := SeriesAndFieldFromCompositeKey([]byte(k))
		if _, ok := keyMap[string(seriesKey)]; ok {
			walKeys = append(walKeys, k)
		}
	}
	e.Cache.RUnlock()

	e.Cache.DeleteRange(walKeys, min, max)

	// delete from the WAL
	_, err := e.WAL.DeleteRange(walKeys, min, max)

	return err
}

// DeleteMeasurement deletes a measurement and all related series.
func (e *Engine) DeleteMeasurement(name string, seriesKeys []string) error {
	e.mu.Lock()
	delete(e.measurementFields, name)
	e.mu.Unlock()

	return e.DeleteSeries(seriesKeys)
}

// SeriesCount returns the number of series buckets on the shard.
func (e *Engine) SeriesCount() (n int, err error) {
	return e.index.SeriesN(), nil
}

func (e *Engine) WriteTo(w io.Writer) (n int64, err error) { panic("not implemented") }

// WriteSnapshot will snapshot the cache and write a new TSM file with its contents, releasing the snapshot when done.
func (e *Engine) WriteSnapshot() error {
	// Lock and grab the cache snapshot along with all the closed WAL
	// filenames associated with the snapshot

	var started *time.Time

	defer func() {
		if started != nil {
			e.Cache.UpdateCompactTime(time.Now().Sub(*started))
			e.logger.Printf("Snapshot for path %s written in %v", e.path, time.Since(*started))
		}
	}()

	closedFiles, snapshot, err := func() ([]string, *Cache, error) {
		e.mu.Lock()
		defer e.mu.Unlock()

		now := time.Now()
		started = &now

		if err := e.WAL.CloseSegment(); err != nil {
			return nil, nil, err
		}

		segments, err := e.WAL.ClosedSegments()
		if err != nil {
			return nil, nil, err
		}

		snapshot, err := e.Cache.Snapshot()
		if err != nil {
			return nil, nil, err
		}

		return segments, snapshot, nil
	}()

	if err != nil {
		return err
	}

	// The snapshotted cache may have duplicate points and unsorted data.  We need to deduplicate
	// it before writing the snapshot.  This can be very expensive so it's done while we are not
	// holding the engine write lock.
	dedup := time.Now()
	snapshot.Deduplicate()
	e.traceLogger.Printf("Snapshot for path %s deduplicated in %v", e.path, time.Since(dedup))

	return e.writeSnapshotAndCommit(closedFiles, snapshot)
}

// CreateSnapshot will create a temp directory that holds
// temporary hardlinks to the underylyng shard files
func (e *Engine) CreateSnapshot() (string, error) {
	if err := e.WriteSnapshot(); err != nil {
		return "", err
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.FileStore.CreateSnapshot()
}

// writeSnapshotAndCommit will write the passed cache to a new TSM file and remove the closed WAL segments
func (e *Engine) writeSnapshotAndCommit(closedFiles []string, snapshot *Cache) (err error) {

	defer func() {
		if err != nil {
			e.Cache.ClearSnapshot(false)
		}
	}()
	// write the new snapshot files
	newFiles, err := e.Compactor.WriteSnapshot(snapshot)
	if err != nil {
		e.logger.Printf("error writing snapshot from compactor: %v", err)
		return err
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// update the file store with these new files
	if err := e.FileStore.Replace(nil, newFiles); err != nil {
		e.logger.Printf("error adding new TSM files from snapshot: %v", err)
		return err
	}

	// clear the snapshot from the in-memory cache, then the old WAL files
	e.Cache.ClearSnapshot(true)

	if err := e.WAL.Remove(closedFiles); err != nil {
		e.logger.Printf("error removing closed wal segments: %v", err)
	}

	return nil
}

// compactCache continually checks if the WAL cache should be written to disk
func (e *Engine) compactCache(quit <-chan struct{}) {
	for {
		select {
		case <-quit:
			return

		default:
			e.Cache.UpdateAge()
			if e.ShouldCompactCache(e.WAL.LastWriteTime()) {
				start := time.Now()
				e.traceLogger.Printf("Compacting cache for %s", e.path)
				err := e.WriteSnapshot()
				if err != nil && err != errCompactionsDisabled {
					e.logger.Printf("error writing snapshot: %v", err)
					atomic.AddInt64(&e.stats.CacheCompactionErrors, 1)
				} else {
					atomic.AddInt64(&e.stats.CacheCompactions, 1)
				}
				atomic.AddInt64(&e.stats.CacheCompactionDuration, time.Since(start).Nanoseconds())
			}
		}
		time.Sleep(time.Second)
	}
}

// ShouldCompactCache returns true if the Cache is over its flush threshold
// or if the passed in lastWriteTime is older than the write cold threshold
func (e *Engine) ShouldCompactCache(lastWriteTime time.Time) bool {
	sz := e.Cache.Size()

	if sz == 0 {
		return false
	}

	return sz > e.CacheFlushMemorySizeThreshold ||
		time.Now().Sub(lastWriteTime) > e.CacheFlushWriteColdDuration
}

func (e *Engine) compactTSMLevel(fast bool, level int, quit <-chan struct{}) {
	for {
		select {
		case <-quit:
			return

		default:
			s := e.levelCompactionStrategy(fast, level)
			if s == nil {
				time.Sleep(time.Second)
				continue
			}

			s.Apply()
		}
	}
}

func (e *Engine) compactTSMFull(quit <-chan struct{}) {
	for {
		select {
		case <-quit:
			return

		default:
			s := e.fullCompactionStrategy()
			if s == nil {
				time.Sleep(time.Second)
				continue
			}

			s.Apply()
		}
	}
}

// compactionStrategy holds the details of what to do in a compaction.
type compactionStrategy struct {
	compactionGroups []CompactionGroup

	fast        bool
	description string

	durationStat *int64
	activeStat   *int64
	successStat  *int64
	errorStat    *int64

	logger    *log.Logger
	compactor *Compactor
	fileStore *FileStore
}

// Apply concurrently compacts all the groups in a compaction strategy.
func (s *compactionStrategy) Apply() {
	start := time.Now()

	var wg sync.WaitGroup
	for i := range s.compactionGroups {
		wg.Add(1)
		go func(groupNum int) {
			defer wg.Done()
			s.compactGroup(groupNum)
		}(i)
	}
	wg.Wait()

	atomic.AddInt64(s.durationStat, time.Since(start).Nanoseconds())
}

// compactGroup executes the compaction strategy against a single CompactionGroup.
func (s *compactionStrategy) compactGroup(groupNum int) {
	group := s.compactionGroups[groupNum]
	start := time.Now()
	s.logger.Printf("beginning %s compaction of group %d, %d TSM files", s.description, groupNum, len(group))
	for i, f := range group {
		s.logger.Printf("compacting %s group (%d) %s (#%d)", s.description, groupNum, f, i)
	}

	files, err := func() ([]string, error) {
		// Count the compaction as active only while the compaction is actually running.
		atomic.AddInt64(s.activeStat, 1)
		defer atomic.AddInt64(s.activeStat, -1)

		if s.fast {
			return s.compactor.CompactFast(group)
		} else {
			return s.compactor.CompactFull(group)
		}
	}()

	if err != nil {
		if err == errCompactionsDisabled || err == errCompactionInProgress {
			s.logger.Printf("aborted %s compaction group (%d). %v", s.description, groupNum, err)

			if err == errCompactionInProgress {
				time.Sleep(time.Second)
			}
			return
		}

		s.logger.Printf("error compacting TSM files: %v", err)
		atomic.AddInt64(s.errorStat, 1)
		time.Sleep(time.Second)
		return
	}

	if err := s.fileStore.Replace(group, files); err != nil {
		s.logger.Printf("error replacing new TSM files: %v", err)
		atomic.AddInt64(s.errorStat, 1)
		time.Sleep(time.Second)
		return
	}

	for i, f := range files {
		s.logger.Printf("compacted %s group (%d) into %s (#%d)", s.description, groupNum, f, i)
	}
	s.logger.Printf("compacted %s %d files into %d files in %s", s.description, len(group), len(files), time.Since(start))
	atomic.AddInt64(s.successStat, 1)
}

// levelCompactionStrategy returns a compactionStrategy for the given level.
// It returns nil if there are no TSM files to compact.
func (e *Engine) levelCompactionStrategy(fast bool, level int) *compactionStrategy {
	compactionGroups := e.CompactionPlan.PlanLevel(level)

	if len(compactionGroups) == 0 {
		return nil
	}

	return &compactionStrategy{
		compactionGroups: compactionGroups,
		logger:           e.logger,
		fileStore:        e.FileStore,
		compactor:        e.Compactor,
		fast:             fast,

		description:  fmt.Sprintf("level %d", level),
		activeStat:   &e.stats.TSMCompactionsActive[level-1],
		successStat:  &e.stats.TSMCompactions[level-1],
		errorStat:    &e.stats.TSMCompactionErrors[level-1],
		durationStat: &e.stats.TSMCompactionDuration[level-1],
	}
}

// fullCompactionStrategy returns a compactionStrategy for higher level generations of TSM files.
// It returns nil if there are no TSM files to compact.
func (e *Engine) fullCompactionStrategy() *compactionStrategy {
	optimize := false
	compactionGroups := e.CompactionPlan.Plan(e.WAL.LastWriteTime())

	if len(compactionGroups) == 0 {
		optimize = true
		compactionGroups = e.CompactionPlan.PlanOptimize()
	}

	if len(compactionGroups) == 0 {
		return nil
	}

	s := &compactionStrategy{
		compactionGroups: compactionGroups,
		logger:           e.logger,
		fileStore:        e.FileStore,
		compactor:        e.Compactor,
		fast:             optimize,
	}

	if optimize {
		s.description = "optimize"
		s.activeStat = &e.stats.TSMOptimizeCompactionsActive
		s.successStat = &e.stats.TSMOptimizeCompactions
		s.errorStat = &e.stats.TSMOptimizeCompactionErrors
		s.durationStat = &e.stats.TSMOptimizeCompactionDuration
	} else {
		s.description = "full"
		s.activeStat = &e.stats.TSMFullCompactionsActive
		s.successStat = &e.stats.TSMFullCompactions
		s.errorStat = &e.stats.TSMFullCompactionErrors
		s.durationStat = &e.stats.TSMFullCompactionDuration
	}

	return s
}

// reloadCache reads the WAL segment files and loads them into the cache.
func (e *Engine) reloadCache() error {
	now := time.Now()
	files, err := segmentFileNames(e.WAL.Path())
	if err != nil {
		return err
	}

	limit := e.Cache.MaxSize()
	defer func() {
		e.Cache.SetMaxSize(limit)
	}()

	// Disable the max size during loading
	e.Cache.SetMaxSize(0)

	loader := NewCacheLoader(files)
	loader.SetLogOutput(e.logOutput)
	if err := loader.Load(e.Cache); err != nil {
		return err
	}

	e.traceLogger.Printf("Reloaded WAL cache %s in %v", e.WAL.Path(), time.Since(now))
	return nil
}

func (e *Engine) cleanup() error {
	allfiles, err := ioutil.ReadDir(e.path)
	if os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}

	for _, f := range allfiles {
		// Check to see if there are any `.tmp` directories that were left over from failed shard snapshots
		if f.IsDir() && strings.HasSuffix(f.Name(), ".tmp") {
			if err := os.RemoveAll(filepath.Join(e.path, f.Name())); err != nil {
				return fmt.Errorf("error removing tmp snapshot directory %q: %s", f.Name(), err)
			}
		}
	}

	return e.cleanupTempTSMFiles()
}

func (e *Engine) cleanupTempTSMFiles() error {
	files, err := filepath.Glob(filepath.Join(e.path, fmt.Sprintf("*.%s", CompactionTempExtension)))
	if err != nil {
		return fmt.Errorf("error getting compaction temp files: %s", err.Error())
	}

	for _, f := range files {
		if err := os.Remove(f); err != nil {
			return fmt.Errorf("error removing temp compaction files: %v", err)
		}
	}
	return nil
}

func (e *Engine) KeyCursor(key string, t int64, ascending bool) *KeyCursor {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.FileStore.KeyCursor(key, t, ascending)
}

func (e *Engine) CreateIterator(opt influxql.IteratorOptions) (influxql.Iterator, error) {
	if call, ok := opt.Expr.(*influxql.Call); ok {
		refOpt := opt
		refOpt.Expr = call.Args[0].(*influxql.VarRef)

		aggregate := true
		if opt.Interval.IsZero() {
			switch call.Name {
			case "first":
				aggregate = false
				refOpt.Limit = 1
				refOpt.Ascending = true
			case "last":
				aggregate = false
				refOpt.Limit = 1
				refOpt.Ascending = false
			}
		}

		inputs, err := e.createVarRefIterator(refOpt, aggregate)
		if err != nil {
			return nil, err
		} else if len(inputs) == 0 {
			return nil, nil
		}

		// Wrap each series in a call iterator.
		for i, input := range inputs {
			if opt.InterruptCh != nil {
				input = influxql.NewInterruptIterator(input, opt.InterruptCh)
			}

			itr, err := influxql.NewCallIterator(input, opt)
			if err != nil {
				return nil, err
			}
			inputs[i] = itr
		}

		return influxql.NewParallelMergeIterator(inputs, opt, runtime.GOMAXPROCS(0)), nil
	}

	itrs, err := e.createVarRefIterator(opt, false)
	if err != nil {
		return nil, err
	}

	itr := influxql.NewSortedMergeIterator(itrs, opt)
	if itr != nil && opt.InterruptCh != nil {
		itr = influxql.NewInterruptIterator(itr, opt.InterruptCh)
	}
	return itr, nil
}

// createVarRefIterator creates an iterator for a variable reference.
// The aggregate argument determines this is being created for an aggregate.
// If this is an aggregate, the limit optimization is disabled temporarily. See #6661.
func (e *Engine) createVarRefIterator(opt influxql.IteratorOptions, aggregate bool) ([]influxql.Iterator, error) {
	ref, _ := opt.Expr.(*influxql.VarRef)

	var itrs []influxql.Iterator
	if err := func() error {
		mms := tsdb.Measurements(e.index.MeasurementsByName(influxql.Sources(opt.Sources).Names()))

		for _, mm := range mms {
			// Determine tagsets for this measurement based on dimensions and filters.
			tagSets, err := mm.TagSets(e.id, opt.Dimensions, opt.Condition)
			if err != nil {
				return err
			}

			// Calculate tag sets and apply SLIMIT/SOFFSET.
			tagSets = influxql.LimitTagSets(tagSets, opt.SLimit, opt.SOffset)

			for _, t := range tagSets {
				inputs, err := e.createTagSetIterators(ref, mm, t, opt)
				if err != nil {
					return err
				}

				if !aggregate && len(inputs) > 0 && (opt.Limit > 0 || opt.Offset > 0) {
					itrs = append(itrs, newLimitIterator(influxql.NewSortedMergeIterator(inputs, opt), opt))
				} else {
					itrs = append(itrs, inputs...)
				}
			}
		}
		return nil
	}(); err != nil {
		influxql.Iterators(itrs).Close()
		return nil, err
	}

	return itrs, nil
}

// createTagSetIterators creates a set of iterators for a tagset.
func (e *Engine) createTagSetIterators(ref *influxql.VarRef, mm *tsdb.Measurement, t *influxql.TagSet, opt influxql.IteratorOptions) ([]influxql.Iterator, error) {
	// Set parallelism by number of logical cpus.
	parallelism := runtime.GOMAXPROCS(0)
	if parallelism > len(t.SeriesKeys) {
		parallelism = len(t.SeriesKeys)
	}

	// Create series key groupings w/ return error.
	groups := make([]struct {
		keys    []string
		filters []influxql.Expr
		itrs    []influxql.Iterator
		err     error
	}, parallelism)

	// Group series keys.
	n := len(t.SeriesKeys) / parallelism
	for i := 0; i < parallelism; i++ {
		group := &groups[i]

		if i < parallelism-1 {
			group.keys = t.SeriesKeys[i*n : (i+1)*n]
			group.filters = t.Filters[i*n : (i+1)*n]
		} else {
			group.keys = t.SeriesKeys[i*n:]
			group.filters = t.Filters[i*n:]
		}

		group.itrs = make([]influxql.Iterator, 0, len(group.keys))
	}

	// Read series groups in parallel.
	var wg sync.WaitGroup
	for i := range groups {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			groups[i].itrs, groups[i].err = e.createTagSetGroupIterators(ref, mm, groups[i].keys, t, groups[i].filters, opt)
		}(i)
	}
	wg.Wait()

	// Determine total number of iterators so we can allocate only once.
	var itrN int
	for _, group := range groups {
		itrN += len(group.itrs)
	}

	// Combine all iterators together and check for errors.
	var err error
	itrs := make([]influxql.Iterator, 0, itrN)
	for _, group := range groups {
		if group.err != nil {
			err = group.err
		}
		itrs = append(itrs, group.itrs...)
	}

	// If an error occurred, make sure we close all created iterators.
	if err != nil {
		influxql.Iterators(itrs).Close()
		return nil, err
	}

	return itrs, nil
}

// createTagSetGroupIterators creates a set of iterators for a subset of a tagset's series.
func (e *Engine) createTagSetGroupIterators(ref *influxql.VarRef, mm *tsdb.Measurement, seriesKeys []string, t *influxql.TagSet, filters []influxql.Expr, opt influxql.IteratorOptions) ([]influxql.Iterator, error) {
	conditionFields := make([]influxql.VarRef, len(influxql.ExprNames(opt.Condition)))

	itrs := make([]influxql.Iterator, 0, len(seriesKeys))
	for i, seriesKey := range seriesKeys {
		fields := 0
		if filters[i] != nil {
			// Retrieve non-time fields from this series filter and filter out tags.
			for _, f := range influxql.ExprNames(filters[i]) {
				conditionFields[fields] = f
				fields++
			}
		}

		itr, err := e.createVarRefSeriesIterator(ref, mm, seriesKey, t, filters[i], conditionFields[:fields], opt)
		if err != nil {
			return itrs, err
		} else if itr == nil {
			continue
		}
		itrs = append(itrs, itr)
	}
	return itrs, nil
}

// createVarRefSeriesIterator creates an iterator for a variable reference for a series.
func (e *Engine) createVarRefSeriesIterator(ref *influxql.VarRef, mm *tsdb.Measurement, seriesKey string, t *influxql.TagSet, filter influxql.Expr, conditionFields []influxql.VarRef, opt influxql.IteratorOptions) (influxql.Iterator, error) {
	tags := influxql.NewTags(e.index.TagsForSeries(seriesKey).Map())

	// Create options specific for this series.
	itrOpt := opt
	itrOpt.Condition = filter

	// Build auxilary cursors.
	// Tag values should be returned if the field doesn't exist.
	var aux []cursorAt
	if len(opt.Aux) > 0 {
		aux = make([]cursorAt, len(opt.Aux))
		for i, ref := range opt.Aux {
			// Create cursor from field if a tag wasn't requested.
			if ref.Type != influxql.Tag {
				cur := e.buildCursor(mm.Name, seriesKey, &ref, opt)
				if cur != nil {
					aux[i] = newBufCursor(cur, opt.Ascending)
					continue
				}

				// If a field was requested, use a nil cursor of the requested type.
				switch ref.Type {
				case influxql.Float, influxql.AnyField:
					aux[i] = &floatNilLiteralCursor{}
					continue
				case influxql.Integer:
					aux[i] = &integerNilLiteralCursor{}
					continue
				case influxql.String:
					aux[i] = &stringNilLiteralCursor{}
					continue
				case influxql.Boolean:
					aux[i] = &booleanNilLiteralCursor{}
					continue
				}
			}

			// If field doesn't exist, use the tag value.
			if v := tags.Value(ref.Val); v == "" {
				// However, if the tag value is blank then return a null.
				aux[i] = &stringNilLiteralCursor{}
			} else {
				aux[i] = &stringLiteralCursor{value: v}
			}
		}
	}

	// Build conditional field cursors.
	// If a conditional field doesn't exist then ignore the series.
	var conds []cursorAt
	if len(conditionFields) > 0 {
		conds = make([]cursorAt, len(conditionFields))
		for i, ref := range conditionFields {
			// Create cursor from field if a tag wasn't requested.
			if ref.Type != influxql.Tag {
				cur := e.buildCursor(mm.Name, seriesKey, &ref, opt)
				if cur != nil {
					conds[i] = newBufCursor(cur, opt.Ascending)
					continue
				}

				// If a field was requested, use a nil cursor of the requested type.
				switch ref.Type {
				case influxql.Float, influxql.AnyField:
					conds[i] = &floatNilLiteralCursor{}
					continue
				case influxql.Integer:
					conds[i] = &integerNilLiteralCursor{}
					continue
				case influxql.String:
					conds[i] = &stringNilLiteralCursor{}
					continue
				case influxql.Boolean:
					conds[i] = &booleanNilLiteralCursor{}
					continue
				}
			}

			// If field doesn't exist, use the tag value.
			if v := tags.Value(ref.Val); v == "" {
				// However, if the tag value is blank then return a null.
				conds[i] = &stringNilLiteralCursor{}
			} else {
				conds[i] = &stringLiteralCursor{value: v}
			}
		}
	}
	condNames := influxql.VarRefs(conditionFields).Strings()

	// Limit tags to only the dimensions selected.
	tags = tags.Subset(opt.Dimensions)

	// If it's only auxiliary fields then it doesn't matter what type of iterator we use.
	if ref == nil {
		return newFloatIterator(mm.Name, tags, itrOpt, nil, aux, conds, condNames), nil
	}

	// Build main cursor.
	cur := e.buildCursor(mm.Name, seriesKey, ref, opt)

	// If the field doesn't exist then don't build an iterator.
	if cur == nil {
		return nil, nil
	}

	switch cur := cur.(type) {
	case floatCursor:
		return newFloatIterator(mm.Name, tags, itrOpt, cur, aux, conds, condNames), nil
	case integerCursor:
		return newIntegerIterator(mm.Name, tags, itrOpt, cur, aux, conds, condNames), nil
	case stringCursor:
		return newStringIterator(mm.Name, tags, itrOpt, cur, aux, conds, condNames), nil
	case booleanCursor:
		return newBooleanIterator(mm.Name, tags, itrOpt, cur, aux, conds, condNames), nil
	default:
		panic("unreachable")
	}
}

// buildCursor creates an untyped cursor for a field.
func (e *Engine) buildCursor(measurement, seriesKey string, ref *influxql.VarRef, opt influxql.IteratorOptions) cursor {
	// Look up fields for measurement.
	e.mu.RLock()
	mf := e.measurementFields[measurement]
	e.mu.RUnlock()

	if mf == nil {
		return nil
	}

	// Find individual field.
	f := mf.Field(ref.Val)
	if f == nil {
		return nil
	}

	// Check if we need to perform a cast. Performing a cast in the
	// engine (if it is possible) is much more efficient than an automatic cast.
	if ref.Type != influxql.Unknown && ref.Type != influxql.AnyField && ref.Type != f.Type {
		switch ref.Type {
		case influxql.Float:
			switch f.Type {
			case influxql.Integer:
				cur := e.buildIntegerCursor(measurement, seriesKey, ref.Val, opt)
				return &floatCastIntegerCursor{cursor: cur}
			}
		case influxql.Integer:
			switch f.Type {
			case influxql.Float:
				cur := e.buildFloatCursor(measurement, seriesKey, ref.Val, opt)
				return &integerCastFloatCursor{cursor: cur}
			}
		}
		return nil
	}

	// Return appropriate cursor based on type.
	switch f.Type {
	case influxql.Float:
		return e.buildFloatCursor(measurement, seriesKey, ref.Val, opt)
	case influxql.Integer:
		return e.buildIntegerCursor(measurement, seriesKey, ref.Val, opt)
	case influxql.String:
		return e.buildStringCursor(measurement, seriesKey, ref.Val, opt)
	case influxql.Boolean:
		return e.buildBooleanCursor(measurement, seriesKey, ref.Val, opt)
	default:
		panic("unreachable")
	}
}

// buildFloatCursor creates a cursor for a float field.
func (e *Engine) buildFloatCursor(measurement, seriesKey, field string, opt influxql.IteratorOptions) floatCursor {
	cacheValues := e.Cache.Values(SeriesFieldKey(seriesKey, field))
	keyCursor := e.KeyCursor(SeriesFieldKey(seriesKey, field), opt.SeekTime(), opt.Ascending)
	return newFloatCursor(opt.SeekTime(), opt.Ascending, cacheValues, keyCursor)
}

// buildIntegerCursor creates a cursor for an integer field.
func (e *Engine) buildIntegerCursor(measurement, seriesKey, field string, opt influxql.IteratorOptions) integerCursor {
	cacheValues := e.Cache.Values(SeriesFieldKey(seriesKey, field))
	keyCursor := e.KeyCursor(SeriesFieldKey(seriesKey, field), opt.SeekTime(), opt.Ascending)
	return newIntegerCursor(opt.SeekTime(), opt.Ascending, cacheValues, keyCursor)
}

// buildStringCursor creates a cursor for a string field.
func (e *Engine) buildStringCursor(measurement, seriesKey, field string, opt influxql.IteratorOptions) stringCursor {
	cacheValues := e.Cache.Values(SeriesFieldKey(seriesKey, field))
	keyCursor := e.KeyCursor(SeriesFieldKey(seriesKey, field), opt.SeekTime(), opt.Ascending)
	return newStringCursor(opt.SeekTime(), opt.Ascending, cacheValues, keyCursor)
}

// buildBooleanCursor creates a cursor for a boolean field.
func (e *Engine) buildBooleanCursor(measurement, seriesKey, field string, opt influxql.IteratorOptions) booleanCursor {
	cacheValues := e.Cache.Values(SeriesFieldKey(seriesKey, field))
	keyCursor := e.KeyCursor(SeriesFieldKey(seriesKey, field), opt.SeekTime(), opt.Ascending)
	return newBooleanCursor(opt.SeekTime(), opt.Ascending, cacheValues, keyCursor)
}

// SeriesFieldKey combine a series key and field name for a unique string to be hashed to a numeric ID
func SeriesFieldKey(seriesKey, field string) string {
	return seriesKey + keyFieldSeparator + field
}

func tsmFieldTypeToInfluxQLDataType(typ byte) (influxql.DataType, error) {
	switch typ {
	case BlockFloat64:
		return influxql.Float, nil
	case BlockInteger:
		return influxql.Integer, nil
	case BlockBoolean:
		return influxql.Boolean, nil
	case BlockString:
		return influxql.String, nil
	default:
		return influxql.Unknown, fmt.Errorf("unknown block type: %v", typ)
	}
}

func SeriesAndFieldFromCompositeKey(key []byte) ([]byte, string) {
	sep := bytes.Index(key, []byte(keyFieldSeparator))
	if sep == -1 {
		// No field???
		return key, ""
	}
	return key[:sep], string(key[sep+len(keyFieldSeparator):])
}
