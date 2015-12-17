package b1

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/boltdb/bolt"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

// Format is the file format name of this engine.
const Format = "b1"

func init() {
	tsdb.RegisterEngine(Format, NewEngine)
}

// topLevelBucketN is the number of non-series buckets in the bolt db.
const topLevelBucketN = 3

var (
	// ErrWALPartitionNotFound returns when flushing a partition that does not exist.
	ErrWALPartitionNotFound = errors.New("wal partition not found")
)

// Ensure Engine implements the interface.
var _ tsdb.Engine = &Engine{}

// Engine represents a version 1 storage engine.
type Engine struct {
	mu sync.RWMutex

	path string   // path to data file
	db   *bolt.DB // underlying database

	cache map[uint8]map[string][][]byte // values by <wal partition,series>

	walSize    int           // approximate size of the WAL, in bytes
	flush      chan struct{} // signals background flush
	flushTimer *time.Timer   // signals time-based flush

	// These coordinate closing and waiting for running goroutines.
	wg      sync.WaitGroup
	closing chan struct{}

	// Used for out-of-band error messages.
	logger *log.Logger

	// The maximum size and time thresholds for flushing the WAL.
	MaxWALSize             int
	WALFlushInterval       time.Duration
	WALPartitionFlushDelay time.Duration

	// The writer used by the logger.
	LogOutput io.Writer
}

// NewEngine returns a new instance of Engine. walPath isn't used in b1, but had
// to be added for later engine versions.
func NewEngine(path string, walPath string, opt tsdb.EngineOptions) tsdb.Engine {
	e := &Engine{
		path:  path,
		flush: make(chan struct{}, 1),

		MaxWALSize:             opt.MaxWALSize,
		WALFlushInterval:       opt.WALFlushInterval,
		WALPartitionFlushDelay: opt.WALPartitionFlushDelay,

		LogOutput: os.Stderr,
	}

	// Initialize all partitions of the cache.
	e.cache = make(map[uint8]map[string][][]byte)
	for i := uint8(0); i < WALPartitionN; i++ {
		e.cache[i] = make(map[string][][]byte)
	}

	return e
}

// Path returns the path the engine was initialized with.
func (e *Engine) Path() string { return e.path }

// PerformMaintenance is for periodic maintenance of the store. A no-op for b1
func (e *Engine) PerformMaintenance() {}

// Format returns the format type of this engine
func (e *Engine) Format() tsdb.EngineFormat {
	return tsdb.B1Format
}

// Open opens and initializes the engine.
func (e *Engine) Open() error {
	if err := func() error {
		e.mu.Lock()
		defer e.mu.Unlock()

		// Open underlying storage.
		db, err := bolt.Open(e.path, 0666, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			return err
		}
		e.db = db

		// Initialize data file.
		if err := e.db.Update(func(tx *bolt.Tx) error {
			_, _ = tx.CreateBucketIfNotExists([]byte("series"))
			_, _ = tx.CreateBucketIfNotExists([]byte("fields"))
			_, _ = tx.CreateBucketIfNotExists([]byte("wal"))

			// Set file format, if not set yet.
			b, _ := tx.CreateBucketIfNotExists([]byte("meta"))
			if v := b.Get([]byte("format")); v == nil {
				if err := b.Put([]byte("format"), []byte(Format)); err != nil {
					return fmt.Errorf("set format: %s", err)
				}
			}

			return nil
		}); err != nil {
			return fmt.Errorf("init: %s", err)
		}

		// Start flush interval timer.
		e.flushTimer = time.NewTimer(e.WALFlushInterval)

		// Initialize logger.
		e.logger = log.New(e.LogOutput, "[b1] ", log.LstdFlags)

		// Start background goroutines.
		e.wg.Add(1)
		e.closing = make(chan struct{})
		go e.autoflusher(e.closing)

		return nil
	}(); err != nil {
		e.close()
		return err
	}

	// Flush on-disk WAL before we return to the caller.
	if err := e.Flush(0); err != nil {
		return fmt.Errorf("flush: %s", err)
	}

	return nil
}

func (e *Engine) Close() error {
	e.mu.Lock()
	err := e.close()
	e.mu.Unlock()

	// Wait for open goroutines to finish.
	e.wg.Wait()
	return err
}

func (e *Engine) close() error {
	if e.db != nil {
		e.db.Close()
	}
	if e.closing != nil {
		close(e.closing)
		e.closing = nil
	}
	return nil
}

// SetLogOutput sets the writer used for log output.
// This must be set before opening the engine.
func (e *Engine) SetLogOutput(w io.Writer) { e.LogOutput = w }

// LoadMetadataIndex loads the shard metadata into memory.
func (e *Engine) LoadMetadataIndex(shard *tsdb.Shard, index *tsdb.DatabaseIndex, measurementFields map[string]*tsdb.MeasurementFields) error {
	return e.db.View(func(tx *bolt.Tx) error {
		// load measurement metadata
		meta := tx.Bucket([]byte("fields"))
		c := meta.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			m := index.CreateMeasurementIndexIfNotExists(string(k))
			mf := &tsdb.MeasurementFields{}
			if err := mf.UnmarshalBinary(v); err != nil {
				return err
			}
			for name, _ := range mf.Fields {
				m.SetFieldName(name)
			}
			mf.Codec = tsdb.NewFieldCodec(mf.Fields)
			measurementFields[m.Name] = mf
		}

		// load series metadata
		meta = tx.Bucket([]byte("series"))
		c = meta.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			series := tsdb.NewSeries("", nil)
			if err := series.UnmarshalBinary(v); err != nil {
				return err
			}
			index.CreateSeriesIndexIfNotExists(tsdb.MeasurementFromSeriesKey(string(k)), series)
		}
		return nil
	})
}

// WritePoints will write the raw data points and any new metadata to the index in the shard
func (e *Engine) WritePoints(points []models.Point, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
	// save to the underlying bolt instance
	if err := e.db.Update(func(tx *bolt.Tx) error {
		// save any new metadata
		if len(seriesToCreate) > 0 {
			b := tx.Bucket([]byte("series"))
			for _, sc := range seriesToCreate {
				data, err := sc.Series.MarshalBinary()
				if err != nil {
					return err
				}
				if err := b.Put([]byte(sc.Series.Key), data); err != nil {
					return err
				}
			}
		}
		if len(measurementFieldsToSave) > 0 {
			b := tx.Bucket([]byte("fields"))
			for name, m := range measurementFieldsToSave {
				data, err := m.MarshalBinary()
				if err != nil {
					return err
				}
				if err := b.Put([]byte(name), data); err != nil {
					return err
				}
			}
		}

		// Write points to WAL bucket.
		wal := tx.Bucket([]byte("wal"))
		for _, p := range points {
			// Retrieve partition bucket.
			key := p.Key()
			b, err := wal.CreateBucketIfNotExists([]byte{WALPartition(key)})
			if err != nil {
				return fmt.Errorf("create WAL partition bucket: %s", err)
			}

			// Generate an autoincrementing index for the WAL partition.
			id, _ := b.NextSequence()

			// Append points sequentially to the WAL bucket.
			v := marshalWALEntry(key, p.UnixNano(), p.Data())
			if err := b.Put(u64tob(id), v); err != nil {
				return fmt.Errorf("put wal: %s", err)
			}
		}

		return nil
	}); err != nil {
		return err
	}

	// If successful then save points to in-memory cache.
	if err := func() error {
		e.mu.Lock()
		defer e.mu.Unlock()

		// tracks which in-memory caches need to be resorted
		resorts := map[uint8]map[string]struct{}{}

		for _, p := range points {
			// Generate in-memory cache entry of <timestamp,data>.
			key, data := p.Key(), p.Data()
			v := make([]byte, 8+len(data))
			binary.BigEndian.PutUint64(v[0:8], uint64(p.UnixNano()))
			copy(v[8:], data)

			// Determine if we are appending.
			partitionID := WALPartition(key)
			a := e.cache[partitionID][string(key)]
			appending := (len(a) == 0 || bytes.Compare(a[len(a)-1], v) == -1)

			// Append to cache list.
			a = append(a, v)

			// If not appending, keep track of cache lists that need to be resorted.
			if !appending {
				series := resorts[partitionID]
				if series == nil {
					series = map[string]struct{}{}
					resorts[partitionID] = series
				}
				series[string(key)] = struct{}{}
			}

			e.cache[partitionID][string(key)] = a

			// Calculate estimated WAL size.
			e.walSize += len(key) + len(v)
		}

		// Sort by timestamp if not appending.
		for partitionID, cache := range resorts {
			for key, _ := range cache {
				sort.Sort(byteSlices(e.cache[partitionID][key]))
			}
		}

		// Check for flush threshold.
		e.triggerAutoFlush()

		return nil
	}(); err != nil {
		return err
	}

	return nil
}

// DeleteSeries deletes the series from the engine.
func (e *Engine) DeleteSeries(keys []string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("series"))
		for _, k := range keys {
			if err := b.Delete([]byte(k)); err != nil {
				return err
			}
			if err := tx.DeleteBucket([]byte(k)); err != nil && err != bolt.ErrBucketNotFound {
				return err
			}
			delete(e.cache[WALPartition([]byte(k))], k)
		}
		return nil
	}); err != nil {
		return err
	}

	return nil
}

// DeleteMeasurement deletes a measurement and all related series.
func (e *Engine) DeleteMeasurement(name string, seriesKeys []string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.db.Update(func(tx *bolt.Tx) error {
		bm := tx.Bucket([]byte("fields"))
		if err := bm.Delete([]byte(name)); err != nil {
			return err
		}
		b := tx.Bucket([]byte("series"))
		for _, k := range seriesKeys {
			if err := b.Delete([]byte(k)); err != nil {
				return err
			}
			if err := tx.DeleteBucket([]byte(k)); err != nil && err != bolt.ErrBucketNotFound {
				return err
			}
			delete(e.cache[WALPartition([]byte(k))], k)
		}

		return nil
	}); err != nil {
		return err
	}

	return nil
}

// Flush writes all points from the write ahead log to the index.
func (e *Engine) Flush(partitionFlushDelay time.Duration) error {
	// Retrieve a list of WAL buckets.
	var partitionIDs []uint8
	if err := e.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket([]byte("wal")).ForEach(func(key, _ []byte) error {
			partitionIDs = append(partitionIDs, uint8(key[0]))
			return nil
		})
	}); err != nil {
		return err
	}

	// Continue flushing until there are no more partition buckets.
	for _, partitionID := range partitionIDs {
		if err := e.FlushPartition(partitionID); err != nil {
			return fmt.Errorf("flush partition: id=%d, err=%s", partitionID, err)
		}

		// Wait momentarily so other threads can process.
		time.Sleep(partitionFlushDelay)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Reset WAL size.
	e.walSize = 0

	// Reset the timer.
	e.flushTimer.Reset(e.WALFlushInterval)

	return nil
}

// FlushPartition flushes a single WAL partition.
func (e *Engine) FlushPartition(partitionID uint8) error {
	startTime := time.Now()

	var pointN int
	if err := e.db.Update(func(tx *bolt.Tx) error {
		e.mu.Lock()
		defer e.mu.Unlock()

		// Retrieve partition bucket. Exit if it doesn't exist.
		pb := tx.Bucket([]byte("wal")).Bucket([]byte{byte(partitionID)})
		if pb == nil {
			return ErrWALPartitionNotFound
		}

		// Iterate over keys in the WAL partition bucket.
		c := pb.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			key, timestamp, data := unmarshalWALEntry(v)

			// Create bucket for entry.
			b, err := tx.CreateBucketIfNotExists(key)
			if err != nil {
				return fmt.Errorf("create bucket: %s", err)
			}

			// Write point to bucket.
			if err := b.Put(u64tob(uint64(timestamp)), data); err != nil {
				return fmt.Errorf("put: %s", err)
			}

			// Remove entry in the WAL.
			if err := c.Delete(); err != nil {
				return fmt.Errorf("delete: %s", err)
			}

			pointN++
		}

		return nil
	}); err != nil {
		return err
	}

	// Reset cache.
	e.mu.Lock()
	e.cache[partitionID] = make(map[string][][]byte)
	e.mu.Unlock()

	if pointN > 0 {
		e.logger.Printf("flush %d points in %.3fs", pointN, time.Since(startTime).Seconds())
	}

	return nil
}

// autoflusher waits for notification of a flush and kicks it off in the background.
// This method runs in a separate goroutine.
func (e *Engine) autoflusher(closing chan struct{}) {
	defer e.wg.Done()

	for {
		// Wait for close or flush signal.
		select {
		case <-closing:
			return
		case <-e.flushTimer.C:
			if err := e.Flush(e.WALPartitionFlushDelay); err != nil {
				e.logger.Printf("flush error: %s", err)
			}
		case <-e.flush:
			if err := e.Flush(e.WALPartitionFlushDelay); err != nil {
				e.logger.Printf("flush error: %s", err)
			}
		}
	}
}

// triggerAutoFlush signals that a flush should occur if the size is above the threshold.
// This function must be called within the context of a lock.
func (e *Engine) triggerAutoFlush() {
	// Ignore if we haven't reached the threshold.
	if e.walSize < e.MaxWALSize {
		return
	}

	// Otherwise send a non-blocking signal.
	select {
	case e.flush <- struct{}{}:
	default:
	}
}

// SeriesCount returns the number of series buckets on the shard.
// This does not include a count from the WAL.
func (e *Engine) SeriesCount() (n int, err error) {
	err = e.db.View(func(tx *bolt.Tx) error {
		return tx.ForEach(func(_ []byte, _ *bolt.Bucket) error {
			n++
			return nil
		})
	})

	// Remove top-level buckets.
	n -= topLevelBucketN

	return
}

// Begin starts a new transaction on the engine.
func (e *Engine) Begin(writable bool) (tsdb.Tx, error) {
	tx, err := e.db.Begin(writable)
	if err != nil {
		return nil, err
	}
	return &Tx{Tx: tx, engine: e}, nil
}

// DB returns the underlying Bolt database.
func (e *Engine) DB() *bolt.DB { return e.db }

// WriteTo writes the length and contents of the engine to w.
func (e *Engine) WriteTo(w io.Writer) (n int64, err error) {
	tx, err := e.db.Begin(false)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	// Write size.
	if err := binary.Write(w, binary.BigEndian, uint64(tx.Size())); err != nil {
		return 0, err
	}

	// Write data.
	n, err = tx.WriteTo(w)
	n += 8 // size header
	return
}

// Tx represents a transaction.
type Tx struct {
	*bolt.Tx
	engine *Engine
}

// Cursor returns an iterator for a key over a single field.
func (tx *Tx) Cursor(series string, fields []string, dec *tsdb.FieldCodec, ascending bool) tsdb.Cursor {
	// Retrieve series bucket.
	b := tx.Bucket([]byte(series))

	tx.engine.mu.RLock()
	defer tx.engine.mu.RUnlock()

	// Ignore if there is no bucket or points in the cache.
	partitionID := WALPartition([]byte(series))
	if b == nil && len(tx.engine.cache[partitionID][series]) == 0 {
		return nil
	}

	// Retrieve a copy of the in-cache points for the series.
	cache := make([][]byte, len(tx.engine.cache[partitionID][series]))
	copy(cache, tx.engine.cache[partitionID][series])

	// Build a cursor that merges the bucket and cache together.
	cur := &Cursor{
		cache:     cache,
		fields:    fields,
		dec:       dec,
		ascending: ascending,
	}
	if b != nil {
		cur.cursor = b.Cursor()
	}

	// If it's a reverse cursor, set the current location to the end.
	if !ascending {
		cur.index = len(cache) - 1
		if cur.cursor != nil {
			cur.cursor.Last()
		}
	}
	return cur
}

// Cursor provides ordered iteration across a series.
type Cursor struct {
	// Bolt cursor and readahead buffer.
	cursor *bolt.Cursor
	buf    struct {
		key, value []byte
	}

	// Fields and codec.
	fields []string
	dec    *tsdb.FieldCodec

	// Cache and current cache index.
	cache [][]byte
	index int

	// Previously read key.
	prev []byte

	// The direction the cursor pointer moves after each call to Next()
	ascending bool
}

func (c *Cursor) Ascending() bool { return c.ascending }

// Seek moves the cursor to a position and returns the closest key/value pair.
func (c *Cursor) SeekTo(seek int64) (key int64, value interface{}) {
	// Seek bolt cursor.
	seekBytes := u64tob(uint64(seek))
	if c.cursor != nil {
		c.buf.key, c.buf.value = c.cursor.Seek(seekBytes)
	}

	// Seek cache index.
	c.index = sort.Search(len(c.cache), func(i int) bool {
		return bytes.Compare(c.cache[i][0:8], seekBytes) != -1
	})

	// Search will return an index after the length of cache if the seek value is greater
	// than all the values.  Clamp it to the end of the cache.
	if !c.ascending && c.index >= len(c.cache) {
		c.index = len(c.cache) - 1
	}

	c.prev = nil
	return c.read()
}

// Next returns the next key/value pair from the cursor.
func (c *Cursor) Next() (key int64, value interface{}) {
	return c.read()
}

// read returns the next key/value in the cursor buffer or cache.
func (c *Cursor) read() (key int64, value interface{}) {
	// Continue skipping ahead through duplicate keys in the cache list.
	var k, v []byte
	for {
		if c.ascending {
			k, v = c.readForward()
		} else {
			k, v = c.readReverse()
		}

		// Exit loop if we're at the end of the cache or the next key is different.
		if k == nil || !bytes.Equal(k, c.prev) {
			break
		}
	}

	// Save key so it's not re-read.
	c.prev = k

	// Exit if no keys left.
	if k == nil {
		return tsdb.EOF, nil
	}

	// Convert key to timestamp.
	key = int64(btou64(k))

	// Decode fields. Optimize for single field, if possible.
	if len(c.fields) == 1 {
		decValue, err := c.dec.DecodeByName(c.fields[0], v)
		if err != nil {
			return key, nil
		}
		return key, decValue
	} else if len(c.fields) > 1 {
		m, err := c.dec.DecodeFieldsWithNames(v)
		if err != nil {
			return key, nil
		}
		return key, m
	} else {
		return key, nil
	}
}

// readForward returns the next key/value from the cursor and moves the current location forward.
func (c *Cursor) readForward() (key, value []byte) {
	// Read next value from the cursor.
	if c.buf.key == nil && c.cursor != nil {
		c.buf.key, c.buf.value = c.cursor.Next()
	}

	// Read from the buffer or cache, which ever is lower.
	if c.buf.key != nil && (c.index >= len(c.cache) || bytes.Compare(c.buf.key, c.cache[c.index][0:8]) == -1) {
		key, value = c.buf.key, c.buf.value
		c.buf.key, c.buf.value = nil, nil
	} else if c.index < len(c.cache) {
		key, value = c.cache[c.index][0:8], c.cache[c.index][8:]
		c.index++
	} else {
		key, value = nil, nil
	}
	return
}

// readReverse returns the next key/value from the cursor and moves the current location backwards.
func (c *Cursor) readReverse() (key, value []byte) {
	// Read prev value from the cursor.
	if c.buf.key == nil && c.cursor != nil {
		c.buf.key, c.buf.value = c.cursor.Prev()
	}

	// Read from the buffer or cache, which ever is lower.
	if c.buf.key != nil && (c.index < 0 || bytes.Compare(c.buf.key, c.cache[c.index][0:8]) == 1) {
		key, value = c.buf.key, c.buf.value
		c.buf.key, c.buf.value = nil, nil
	} else if c.index >= 0 && c.index < len(c.cache) {
		key, value = c.cache[c.index][0:8], c.cache[c.index][8:]
		c.index--
	} else {
		key, value = nil, nil
	}
	return
}

// WALPartitionN is the number of partitions in the write ahead log.
const WALPartitionN = 8

// WALPartition returns the partition number that key belongs to.
func WALPartition(key []byte) uint8 {
	h := fnv.New64a()
	h.Write(key)
	return uint8(h.Sum64() % WALPartitionN)
}

// marshalWALEntry encodes point data into a single byte slice.
//
// The format of the byte slice is:
//
//     uint64 timestamp
//     uint32 key length
//     []byte key
//     []byte data
//
func marshalWALEntry(key []byte, timestamp int64, data []byte) []byte {
	v := make([]byte, 8+4, 8+4+len(key)+len(data))
	binary.BigEndian.PutUint64(v[0:8], uint64(timestamp))
	binary.BigEndian.PutUint32(v[8:12], uint32(len(key)))
	v = append(v, key...)
	v = append(v, data...)
	return v
}

// unmarshalWALEntry decodes a WAL entry into it's separate parts.
// Returned byte slices point to the original slice.
func unmarshalWALEntry(v []byte) (key []byte, timestamp int64, data []byte) {
	keyLen := binary.BigEndian.Uint32(v[8:12])
	key = v[12 : 12+keyLen]
	timestamp = int64(binary.BigEndian.Uint64(v[0:8]))
	data = v[12+keyLen:]
	return
}

// marshalCacheEntry encodes the timestamp and data to a single byte slice.
//
// The format of the byte slice is:
//
//     uint64 timestamp
//     []byte data
//
func marshalCacheEntry(timestamp int64, data []byte) []byte {
	buf := make([]byte, 8, 8+len(data))
	binary.BigEndian.PutUint64(buf[0:8], uint64(timestamp))
	return append(buf, data...)
}

// unmarshalCacheEntry returns the timestamp and data from an encoded byte slice.
func unmarshalCacheEntry(buf []byte) (timestamp int64, data []byte) {
	timestamp = int64(binary.BigEndian.Uint64(buf[0:8]))
	data = buf[8:]
	return
}

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

// btou64 converts an 8-byte slice to a uint64.
func btou64(b []byte) uint64 { return binary.BigEndian.Uint64(b) }

// byteSlices represents a sortable slice of byte slices.
type byteSlices [][]byte

func (a byteSlices) Len() int           { return len(a) }
func (a byteSlices) Less(i, j int) bool { return bytes.Compare(a[i], a[j]) == -1 }
func (a byteSlices) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
