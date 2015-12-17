package bz1

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"expvar"
	"fmt"
	"io"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/boltdb/bolt"
	"github.com/golang/snappy"
	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/tsdb/engine/wal"
)

var (
	// ErrSeriesExists is returned when writing points to an existing series.
	ErrSeriesExists = errors.New("series exists")
)

const (
	// Format is the file format name of this engine.
	Format = "bz1"
)

const (
	statSlowInsert               = "slowInsert"
	statPointsWrite              = "pointsWrite"
	statPointsWriteDedupe        = "pointsWriteDedupe"
	statBlocksWrite              = "blksWrite"
	statBlocksWriteBytes         = "blksWriteBytes"
	statBlocksWriteBytesCompress = "blksWriteBytesC"
)

func init() {
	tsdb.RegisterEngine(Format, NewEngine)
}

const (
	// DefaultBlockSize is the default size of uncompressed points blocks.
	DefaultBlockSize = 4 * 1024 // 4KB
)

// Ensure Engine implements the interface.
var _ tsdb.Engine = &Engine{}

// Engine represents a storage engine with compressed blocks.
type Engine struct {
	mu   sync.Mutex
	path string
	db   *bolt.DB

	// expvar-based statistics collection.
	statMap *expvar.Map

	// Write-ahead log storage.
	WAL WAL

	// Size of uncompressed points to write to a block.
	BlockSize int
}

// WAL represents a write ahead log that can be queried
type WAL interface {
	WritePoints(points []models.Point, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error
	LoadMetadataIndex(index *tsdb.DatabaseIndex, measurementFields map[string]*tsdb.MeasurementFields) error
	DeleteSeries(keys []string) error
	Cursor(series string, fields []string, dec *tsdb.FieldCodec, ascending bool) tsdb.Cursor
	Open() error
	Close() error
	Flush() error
}

// NewEngine returns a new instance of Engine.
func NewEngine(path string, walPath string, opt tsdb.EngineOptions) tsdb.Engine {
	// Configure statistics collection.
	key := fmt.Sprintf("engine:%s:%s", opt.EngineVersion, path)
	tags := map[string]string{"path": path, "version": opt.EngineVersion}
	statMap := influxdb.NewStatistics(key, "engine", tags)

	// create the writer with a directory of the same name as the shard, but with the wal extension
	w := wal.NewLog(walPath)

	w.ReadySeriesSize = opt.Config.WALReadySeriesSize
	w.FlushColdInterval = time.Duration(opt.Config.WALFlushColdInterval)
	w.MaxSeriesSize = opt.Config.WALMaxSeriesSize
	w.CompactionThreshold = opt.Config.WALCompactionThreshold
	w.PartitionSizeThreshold = opt.Config.WALPartitionSizeThreshold
	w.ReadySeriesSize = opt.Config.WALReadySeriesSize
	w.LoggingEnabled = opt.Config.WALLoggingEnabled

	e := &Engine{
		path: path,

		statMap:   statMap,
		BlockSize: DefaultBlockSize,
		WAL:       w,
	}

	w.Index = e

	return e
}

// Path returns the path the engine was opened with.
func (e *Engine) Path() string { return e.path }

// PerformMaintenance is for periodic maintenance of the store. A no-op for bz1
func (e *Engine) PerformMaintenance() {}

// Format returns the format type of this engine
func (e *Engine) Format() tsdb.EngineFormat {
	return tsdb.BZ1Format
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
			_, _ = tx.CreateBucketIfNotExists([]byte("points"))

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

		return nil
	}(); err != nil {
		e.close()
		return err
	}

	return nil
}

// Close closes the engine.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.WAL.Close(); err != nil {
		return err
	}

	return e.close()
}

func (e *Engine) close() error {
	if e.db != nil {
		return e.db.Close()
	}
	return nil
}

// SetLogOutput is a no-op.
func (e *Engine) SetLogOutput(w io.Writer) {}

// LoadMetadataIndex loads the shard metadata into memory.
func (e *Engine) LoadMetadataIndex(shard *tsdb.Shard, index *tsdb.DatabaseIndex, measurementFields map[string]*tsdb.MeasurementFields) error {
	if err := e.db.View(func(tx *bolt.Tx) error {
		// Load measurement metadata
		fields, err := e.readFields(tx)
		if err != nil {
			return err
		}
		for k, mf := range fields {
			m := index.CreateMeasurementIndexIfNotExists(string(k))
			for name, _ := range mf.Fields {
				m.SetFieldName(name)
			}
			mf.Codec = tsdb.NewFieldCodec(mf.Fields)
			measurementFields[m.Name] = mf
		}

		// Load series metadata
		series, err := e.readSeries(tx)
		if err != nil {
			return err
		}

		// Load the series into the in-memory index in sorted order to ensure
		// it's always consistent for testing purposes
		a := make([]string, 0, len(series))
		for k, _ := range series {
			a = append(a, k)
		}
		sort.Strings(a)
		for _, key := range a {
			s := series[key]
			s.InitializeShards()
			index.CreateSeriesIndexIfNotExists(tsdb.MeasurementFromSeriesKey(string(key)), s)
		}
		return nil
	}); err != nil {
		return err
	}

	// now flush the metadata that was in the WAL, but hadn't yet been flushed
	if err := e.WAL.LoadMetadataIndex(index, measurementFields); err != nil {
		return err
	}

	// finally open the WAL up
	return e.WAL.Open()
}

// WritePoints writes metadata and point data into the engine.
// Returns an error if new points are added to an existing key.
func (e *Engine) WritePoints(points []models.Point, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
	// Write points to the WAL.
	if err := e.WAL.WritePoints(points, measurementFieldsToSave, seriesToCreate); err != nil {
		return fmt.Errorf("write points: %s", err)
	}

	return nil
}

// WriteIndex writes marshaled points to the engine's underlying index.
func (e *Engine) WriteIndex(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
	return e.db.Update(func(tx *bolt.Tx) error {
		// Write series & field metadata.
		if err := e.writeNewSeries(tx, seriesToCreate); err != nil {
			return fmt.Errorf("write series: %s", err)
		}
		if err := e.writeNewFields(tx, measurementFieldsToSave); err != nil {
			return fmt.Errorf("write fields: %s", err)
		}

		for key, values := range pointsByKey {
			if err := e.writeIndex(tx, key, values); err != nil {
				return fmt.Errorf("write: key=%x, err=%s", key, err)
			}
		}
		return nil
	})
}

func (e *Engine) writeNewFields(tx *bolt.Tx, measurementFieldsToSave map[string]*tsdb.MeasurementFields) error {
	if len(measurementFieldsToSave) == 0 {
		return nil
	}

	// read in all the previously saved fields
	fields, err := e.readFields(tx)
	if err != nil {
		return err
	}

	// add the new ones or overwrite old ones
	for name, mf := range measurementFieldsToSave {
		fields[name] = mf
	}

	return e.writeFields(tx, fields)
}

func (e *Engine) writeFields(tx *bolt.Tx, fields map[string]*tsdb.MeasurementFields) error {
	// compress and save everything
	data, err := json.Marshal(fields)
	if err != nil {
		return err
	}

	return tx.Bucket([]byte("meta")).Put([]byte("fields"), snappy.Encode(nil, data))
}

func (e *Engine) readFields(tx *bolt.Tx) (map[string]*tsdb.MeasurementFields, error) {
	fields := make(map[string]*tsdb.MeasurementFields)

	b := tx.Bucket([]byte("meta")).Get([]byte("fields"))
	if b == nil {
		return fields, nil
	}

	data, err := snappy.Decode(nil, b)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}

	return fields, nil
}

func (e *Engine) writeNewSeries(tx *bolt.Tx, seriesToCreate []*tsdb.SeriesCreate) error {
	if len(seriesToCreate) == 0 {
		return nil
	}

	// read in previously saved series
	series, err := e.readSeries(tx)
	if err != nil {
		return err
	}

	// add new ones, compress and save
	for _, s := range seriesToCreate {
		series[s.Series.Key] = s.Series
	}

	return e.writeSeries(tx, series)
}

func (e *Engine) writeSeries(tx *bolt.Tx, series map[string]*tsdb.Series) error {
	data, err := json.Marshal(series)
	if err != nil {
		return err
	}

	return tx.Bucket([]byte("meta")).Put([]byte("series"), snappy.Encode(nil, data))
}

func (e *Engine) readSeries(tx *bolt.Tx) (map[string]*tsdb.Series, error) {
	series := make(map[string]*tsdb.Series)

	b := tx.Bucket([]byte("meta")).Get([]byte("series"))
	if b == nil {
		return series, nil
	}

	data, err := snappy.Decode(nil, b)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(data, &series); err != nil {
		return nil, err
	}

	return series, nil
}

// writeIndex writes a set of points for a single key.
func (e *Engine) writeIndex(tx *bolt.Tx, key string, a [][]byte) error {
	// Ignore if there are no points.
	if len(a) == 0 {
		return nil
	}
	e.statMap.Add(statPointsWrite, int64(len(a)))

	// Create or retrieve series bucket.
	bkt, err := tx.Bucket([]byte("points")).CreateBucketIfNotExists([]byte(key))
	if err != nil {
		return fmt.Errorf("create series bucket: %s", err)
	}
	c := bkt.Cursor()

	// Ensure the slice is sorted before retrieving the time range.
	a = tsdb.DedupeEntries(a)
	e.statMap.Add(statPointsWriteDedupe, int64(len(a)))

	// Convert the raw time and byte slices to entries with lengths
	for i, p := range a {
		timestamp := int64(btou64(p[0:8]))
		a[i] = MarshalEntry(timestamp, p[8:])
	}

	// Determine time range of new data.
	tmin, tmax := int64(btou64(a[0][0:8])), int64(btou64(a[len(a)-1][0:8]))

	// If tmin is after the last block then append new blocks.
	//
	// This is the optimized fast path. Otherwise we need to merge the points
	// with existing blocks on disk and rewrite all the blocks for that range.
	if k, v := c.Last(); k == nil {
		bkt.FillPercent = 1.0
		if err := e.writeBlocks(bkt, a); err != nil {
			return fmt.Errorf("new blocks: %s", err)
		}
		return nil

	} else if int64(btou64(v[0:8])) < tmin {
		// Append new blocks if our time range is past the last on-disk time.
		bkt.FillPercent = 1.0
		if err := e.writeBlocks(bkt, a); err != nil {
			return fmt.Errorf("append blocks: %s", err)
		}
		return nil
	}

	// Generate map of inserted keys.
	m := make(map[int64]struct{})
	for _, b := range a {
		m[int64(btou64(b[0:8]))] = struct{}{}
	}

	// If time range overlaps existing blocks then unpack full range and reinsert.
	var existing [][]byte
	for k, v := c.First(); k != nil; k, v = c.Next() {
		// Determine block range.
		bmin, bmax := int64(btou64(k)), int64(btou64(v[0:8]))

		// Skip over all blocks before the time range.
		// Exit once we reach a block that is beyond our time range.
		if bmax < tmin {
			continue
		} else if bmin > tmax {
			break
		}

		// Decode block.
		buf, err := snappy.Decode(nil, v[8:])
		if err != nil {
			return fmt.Errorf("decode block: %s", err)
		}

		// Copy out any entries that aren't being overwritten.
		for _, entry := range SplitEntries(buf) {
			if _, ok := m[int64(btou64(entry[0:8]))]; !ok {
				existing = append(existing, entry)
			}
		}

		// Delete block in database.
		c.Delete()
	}

	// Merge entries before rewriting.
	a = append(existing, a...)
	sort.Sort(tsdb.ByteSlices(a))

	// Rewrite points to new blocks.
	if err := e.writeBlocks(bkt, a); err != nil {
		return fmt.Errorf("rewrite blocks: %s", err)
	}

	return nil
}

// writeBlocks writes point data to the bucket in blocks.
func (e *Engine) writeBlocks(bkt *bolt.Bucket, a [][]byte) error {
	var block []byte

	// Group points into blocks by size.
	tmin, tmax := int64(math.MaxInt64), int64(math.MinInt64)
	for i, p := range a {
		// Update block time range.
		timestamp := int64(btou64(p[0:8]))
		if timestamp < tmin {
			tmin = timestamp
		}
		if timestamp > tmax {
			tmax = timestamp
		}

		// Append point to the end of the block.
		block = append(block, p...)

		// If the block is larger than the target block size or this is the
		// last point then flush the block to the bucket.
		if len(block) >= e.BlockSize || i == len(a)-1 {
			e.statMap.Add(statBlocksWrite, 1)
			e.statMap.Add(statBlocksWriteBytes, int64(len(block)))

			// Encode block in the following format:
			//   tmax int64
			//   data []byte (snappy compressed)
			value := append(u64tob(uint64(tmax)), snappy.Encode(nil, block)...)

			// Write block to the bucket.
			if err := bkt.Put(u64tob(uint64(tmin)), value); err != nil {
				return fmt.Errorf("put: ts=%d-%d, err=%s", tmin, tmax, err)
			}
			e.statMap.Add(statBlocksWriteBytesCompress, int64(len(value)))

			// Reset the block & time range.
			block = nil
			tmin, tmax = int64(math.MaxInt64), int64(math.MinInt64)
		}
	}

	return nil
}

// DeleteSeries deletes the series from the engine.
func (e *Engine) DeleteSeries(keys []string) error {
	// remove it from the WAL first
	if err := e.WAL.DeleteSeries(keys); err != nil {
		return err
	}

	return e.db.Update(func(tx *bolt.Tx) error {
		series, err := e.readSeries(tx)
		if err != nil {
			return err
		}
		for _, k := range keys {
			delete(series, k)
			if err := tx.Bucket([]byte("points")).DeleteBucket([]byte(k)); err != nil && err != bolt.ErrBucketNotFound {
				return fmt.Errorf("delete series data: %s", err)
			}
		}

		return e.writeSeries(tx, series)
	})
}

// DeleteMeasurement deletes a measurement and all related series.
func (e *Engine) DeleteMeasurement(name string, seriesKeys []string) error {
	// remove from the WAL first so it won't get flushed after removing from Bolt
	if err := e.WAL.DeleteSeries(seriesKeys); err != nil {
		return err
	}

	return e.db.Update(func(tx *bolt.Tx) error {
		fields, err := e.readFields(tx)
		if err != nil {
			return err
		}
		delete(fields, name)
		if err := e.writeFields(tx, fields); err != nil {
			return err
		}

		series, err := e.readSeries(tx)
		if err != nil {
			return err
		}
		for _, k := range seriesKeys {
			delete(series, k)
			if err := tx.Bucket([]byte("points")).DeleteBucket([]byte(k)); err != nil && err != bolt.ErrBucketNotFound {
				return fmt.Errorf("delete series data: %s", err)
			}
		}

		return e.writeSeries(tx, series)
	})
}

// SeriesCount returns the number of series buckets on the shard.
func (e *Engine) SeriesCount() (n int, err error) {
	err = e.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket([]byte("points")).Cursor()
		for k, _ := c.First(); k != nil; k, _ = c.Next() {
			n++
		}
		return nil
	})
	return
}

// Begin starts a new transaction on the engine.
func (e *Engine) Begin(writable bool) (tsdb.Tx, error) {
	tx, err := e.db.Begin(writable)
	if err != nil {
		return nil, err
	}
	return &Tx{Tx: tx, engine: e, wal: e.WAL}, nil
}

// Stats returns internal statistics for the engine.
func (e *Engine) Stats() (stats Stats, err error) {
	err = e.db.View(func(tx *bolt.Tx) error {
		stats.Size = tx.Size()
		return nil
	})
	return stats, err
}

// SeriesBucketStats returns internal BoltDB stats for a series bucket.
func (e *Engine) SeriesBucketStats(key string) (stats bolt.BucketStats, err error) {
	err = e.db.View(func(tx *bolt.Tx) error {
		bkt := tx.Bucket([]byte("points")).Bucket([]byte(key))
		if bkt != nil {
			stats = bkt.Stats()
		}
		return nil
	})
	return stats, err
}

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

// Stats represents internal engine statistics.
type Stats struct {
	Size int64 // BoltDB data size
}

// Tx represents a transaction.
type Tx struct {
	*bolt.Tx
	engine *Engine
	wal    WAL
}

// Cursor returns an iterator for a key.
func (tx *Tx) Cursor(series string, fields []string, dec *tsdb.FieldCodec, ascending bool) tsdb.Cursor {
	walCursor := tx.wal.Cursor(series, fields, dec, ascending)

	// Retrieve points bucket. Ignore if there is no bucket.
	b := tx.Bucket([]byte("points")).Bucket([]byte(series))
	if b == nil {
		return walCursor
	}

	c := &Cursor{
		cursor:    b.Cursor(),
		fields:    fields,
		dec:       dec,
		ascending: ascending,
	}

	if !ascending {
		c.last()
	}

	return tsdb.MultiCursor(walCursor, c)
}

// Cursor provides ordered iteration across a series.
type Cursor struct {
	cursor       *bolt.Cursor
	buf          []byte // uncompressed buffer
	off          int    // buffer offset
	ascending    bool
	fieldIndices []int
	index        int

	fields []string
	dec    *tsdb.FieldCodec
}

func (c *Cursor) last() {
	_, v := c.cursor.Last()
	c.setBuf(v)
}

func (c *Cursor) Ascending() bool { return c.ascending }

// Seek moves the cursor to a position and returns the closest key/value pair.
func (c *Cursor) SeekTo(seek int64) (key int64, value interface{}) {
	seekBytes := u64tob(uint64(seek))

	// Move cursor to appropriate block and set to buffer.
	k, v := c.cursor.Seek(seekBytes)
	if v == nil { // get the last block, it might have this time
		_, v = c.cursor.Last()
	} else if seek < int64(btou64(k)) { // the seek key is less than this block, go back one and check
		_, v = c.cursor.Prev()

		// if the previous block max time is less than the seek value, reset to where we were originally
		if v == nil || seek > int64(btou64(v[0:8])) {
			_, v = c.cursor.Seek(seekBytes)
		}
	}
	c.setBuf(v)

	// Read current block up to seek position.
	c.seekBuf(seekBytes)

	// Return current entry.
	return c.read()
}

// seekBuf moves the cursor to a position within the current buffer.
func (c *Cursor) seekBuf(seek []byte) (key, value []byte) {
	for {
		// Slice off the current entry.
		buf := c.buf[c.off:]

		// Exit if current entry's timestamp is on or after the seek.
		if len(buf) == 0 {
			return
		}

		if c.ascending && bytes.Compare(buf[0:8], seek) != -1 {
			return
		} else if !c.ascending && bytes.Compare(buf[0:8], seek) != 1 {
			return
		}

		if c.ascending {
			// Otherwise skip ahead to the next entry.
			c.off += entryHeaderSize + entryDataSize(buf)
		} else {
			c.index -= 1
			if c.index < 0 {
				return
			}
			c.off = c.fieldIndices[c.index]
		}
	}
}

// Next returns the next key/value pair from the cursor.
func (c *Cursor) Next() (key int64, value interface{}) {
	// Ignore if there is no buffer.
	if len(c.buf) == 0 {
		return tsdb.EOF, nil
	}

	if c.ascending {
		// Move forward to next entry.
		c.off += entryHeaderSize + entryDataSize(c.buf[c.off:])
	} else {
		// If we've move past the beginning of buf, grab the previous block
		if c.index < 0 {
			_, v := c.cursor.Prev()
			c.setBuf(v)
		}

		if len(c.fieldIndices) > 0 {
			c.off = c.fieldIndices[c.index]
		}
		c.index -= 1
	}

	// If no items left then read first item from next block.
	if c.off >= len(c.buf) {
		_, v := c.cursor.Next()
		c.setBuf(v)
	}

	return c.read()
}

// setBuf saves a compressed block to the buffer.
func (c *Cursor) setBuf(block []byte) {
	// Clear if the block is empty.
	if len(block) == 0 {
		c.buf, c.off, c.fieldIndices, c.index = c.buf[0:0], 0, c.fieldIndices[0:0], 0
		return
	}

	// Otherwise decode block into buffer.
	// Skip over the first 8 bytes since they are the max timestamp.
	buf, err := snappy.Decode(nil, block[8:])
	if err != nil {
		c.buf = c.buf[0:0]
		log.Printf("block decode error: %s", err)
	}

	if c.ascending {
		c.buf, c.off = buf, 0
	} else {
		c.buf, c.off = buf, 0

		// Buf contains multiple fields packed into a byte slice with timestamp
		// and data lengths.  We need to build an index into this byte slice that
		// tells us where each field block is in buf so we can iterate backward without
		// rescanning the buf each time Next is called.  Forward iteration does not
		// need this because we know the entries lenghth and the header size so we can
		// skip forward that many bytes.
		c.fieldIndices = []int{}
		for {
			if c.off >= len(buf) {
				break
			}

			c.fieldIndices = append(c.fieldIndices, c.off)
			c.off += entryHeaderSize + entryDataSize(buf[c.off:])
		}

		c.off = c.fieldIndices[len(c.fieldIndices)-1]
		c.index = len(c.fieldIndices) - 1
	}
}

// read reads the current key and value from the current block.
func (c *Cursor) read() (key int64, value interface{}) {
	// Return nil if the offset is at the end of the buffer.
	if c.off >= len(c.buf) {
		return tsdb.EOF, nil
	}

	// Otherwise read the current entry.
	buf := c.buf[c.off:]
	dataSize := entryDataSize(buf)

	return wal.DecodeKeyValue(c.fields, c.dec, buf[0:8], buf[entryHeaderSize:entryHeaderSize+dataSize])
}

// MarshalEntry encodes point data into a single byte slice.
//
// The format of the byte slice is:
//
//     uint64 timestamp
//     uint32 data length
//     []byte data
//
func MarshalEntry(timestamp int64, data []byte) []byte {
	v := make([]byte, 8+4, 8+4+len(data))
	binary.BigEndian.PutUint64(v[0:8], uint64(timestamp))
	binary.BigEndian.PutUint32(v[8:12], uint32(len(data)))
	v = append(v, data...)
	return v
}

// UnmarshalEntry decodes an entry into it's separate parts.
// Returns the timestamp, data and the number of bytes read.
// Returned byte slices point to the original slice.
func UnmarshalEntry(v []byte) (timestamp int64, data []byte, n int) {
	timestamp = int64(binary.BigEndian.Uint64(v[0:8]))
	dataLen := binary.BigEndian.Uint32(v[8:12])
	data = v[12+dataLen:]
	return timestamp, data, 12 + int(dataLen)
}

// SplitEntries returns a slice of individual entries from one continuous set.
func SplitEntries(b []byte) [][]byte {
	var a [][]byte
	for {
		// Exit if there's no more data left.
		if len(b) == 0 {
			return a
		}

		// Create slice that points to underlying entry.
		dataSize := entryDataSize(b)
		a = append(a, b[0:entryHeaderSize+dataSize])

		// Move buffer forward.
		b = b[entryHeaderSize+dataSize:]
	}
}

// entryHeaderSize is the number of bytes required for the header.
const entryHeaderSize = 8 + 4

// entryDataSize returns the size of an entry's data field, in bytes.
func entryDataSize(v []byte) int { return int(binary.BigEndian.Uint32(v[8:12])) }

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

// btou64 converts an 8-byte slice into an uint64.
func btou64(b []byte) uint64 { return binary.BigEndian.Uint64(b) }
