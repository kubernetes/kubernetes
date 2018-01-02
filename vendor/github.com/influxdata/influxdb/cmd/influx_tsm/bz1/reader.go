package bz1 // import "github.com/influxdata/influxdb/cmd/influx_tsm/bz1"

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/boltdb/bolt"
	"github.com/golang/snappy"
	"github.com/influxdata/influxdb/cmd/influx_tsm/stats"
	"github.com/influxdata/influxdb/cmd/influx_tsm/tsdb"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

// DefaultChunkSize is the size of chunks read from the bz1 shard
const DefaultChunkSize = 1000

// Reader is used to read all data from a bz1 shard.
type Reader struct {
	path string
	db   *bolt.DB
	tx   *bolt.Tx

	cursors    []*cursor
	currCursor int

	keyBuf   string
	values   []tsm1.Value
	valuePos int

	fields map[string]*tsdb.MeasurementFields
	codecs map[string]*tsdb.FieldCodec

	stats *stats.Stats
}

// NewReader returns a reader for the bz1 shard at path.
func NewReader(path string, stats *stats.Stats, chunkSize int) *Reader {
	r := &Reader{
		path:   path,
		fields: make(map[string]*tsdb.MeasurementFields),
		codecs: make(map[string]*tsdb.FieldCodec),
		stats:  stats,
	}

	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	r.values = make([]tsm1.Value, chunkSize)

	return r
}

// Open opens the reader.
func (r *Reader) Open() error {
	// Open underlying storage.
	db, err := bolt.Open(r.path, 0666, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return err
	}
	r.db = db

	seriesSet := make(map[string]bool)

	if err := r.db.View(func(tx *bolt.Tx) error {
		var data []byte

		meta := tx.Bucket([]byte("meta"))
		if meta == nil {
			// No data in this shard.
			return nil
		}

		pointsBucket := tx.Bucket([]byte("points"))
		if pointsBucket == nil {
			return nil
		}

		if err := pointsBucket.ForEach(func(key, _ []byte) error {
			seriesSet[string(key)] = true
			return nil
		}); err != nil {
			return err
		}

		buf := meta.Get([]byte("fields"))
		if buf == nil {
			// No data in this shard.
			return nil
		}

		data, err = snappy.Decode(nil, buf)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(data, &r.fields); err != nil {
			return err
		}
		return nil
	}); err != nil {
		return err
	}

	// Build the codec for each measurement.
	for k, v := range r.fields {
		r.codecs[k] = tsdb.NewFieldCodec(v.Fields)
	}

	r.tx, err = r.db.Begin(false)
	if err != nil {
		return err
	}

	// Create cursor for each field of each series.
	for s := range seriesSet {
		measurement := tsdb.MeasurementFromSeriesKey(s)
		fields := r.fields[measurement]
		if fields == nil {
			r.stats.IncrFiltered()
			continue
		}
		for _, f := range fields.Fields {
			c := newCursor(r.tx, s, f.Name, r.codecs[measurement])
			if c == nil {
				continue
			}
			c.SeekTo(0)
			r.cursors = append(r.cursors, c)
		}
	}
	sort.Sort(cursors(r.cursors))

	return nil
}

// Next returns whether there is any more data to be read.
func (r *Reader) Next() bool {
	r.valuePos = 0
OUTER:
	for {
		if r.currCursor >= len(r.cursors) {
			// All cursors drained. No more data remains.
			return false
		}

		cc := r.cursors[r.currCursor]
		r.keyBuf = tsm1.SeriesFieldKey(cc.series, cc.field)

		for {
			k, v := cc.Next()
			if k == -1 {
				// Go to next cursor and try again.
				r.currCursor++
				if r.valuePos == 0 {
					// The previous cursor had no data. Instead of returning
					// just go immediately to the next cursor.
					continue OUTER
				}
				// There is some data available. Indicate that it should be read.
				return true
			}

			if f, ok := v.(float64); ok {
				if math.IsInf(f, 0) {
					r.stats.AddPointsRead(1)
					r.stats.IncrInf()
					continue
				}

				if math.IsNaN(f) {
					r.stats.AddPointsRead(1)
					r.stats.IncrNaN()
					continue
				}
			}

			r.values[r.valuePos] = tsm1.NewValue(k, v)
			r.valuePos++

			if r.valuePos >= len(r.values) {
				return true
			}
		}
	}
}

// Read returns the next chunk of data in the shard, converted to tsm1 values. Data is
// emitted completely for every field, in every series, before the next field is processed.
// Data from Read() adheres to the requirements for writing to tsm1 shards
func (r *Reader) Read() (string, []tsm1.Value, error) {
	return r.keyBuf, r.values[:r.valuePos], nil
}

// Close closes the reader.
func (r *Reader) Close() error {
	r.tx.Rollback()
	return r.db.Close()
}

// cursor provides ordered iteration across a series.
type cursor struct {
	cursor       *bolt.Cursor
	buf          []byte // uncompressed buffer
	off          int    // buffer offset
	fieldIndices []int
	index        int

	series string
	field  string
	dec    *tsdb.FieldCodec

	keyBuf int64
	valBuf interface{}
}

// newCursor returns an instance of a bz1 cursor.
func newCursor(tx *bolt.Tx, series string, field string, dec *tsdb.FieldCodec) *cursor {
	// Retrieve points bucket. Ignore if there is no bucket.
	b := tx.Bucket([]byte("points")).Bucket([]byte(series))
	if b == nil {
		return nil
	}

	return &cursor{
		cursor: b.Cursor(),
		series: series,
		field:  field,
		dec:    dec,
		keyBuf: -2,
	}
}

// Seek moves the cursor to a position.
func (c *cursor) SeekTo(seek int64) {
	var seekBytes [8]byte
	binary.BigEndian.PutUint64(seekBytes[:], uint64(seek))

	// Move cursor to appropriate block and set to buffer.
	k, v := c.cursor.Seek(seekBytes[:])
	if v == nil { // get the last block, it might have this time
		_, v = c.cursor.Last()
	} else if seek < int64(binary.BigEndian.Uint64(k)) { // the seek key is less than this block, go back one and check
		_, v = c.cursor.Prev()

		// if the previous block max time is less than the seek value, reset to where we were originally
		if v == nil || seek > int64(binary.BigEndian.Uint64(v[0:8])) {
			_, v = c.cursor.Seek(seekBytes[:])
		}
	}
	c.setBuf(v)

	// Read current block up to seek position.
	c.seekBuf(seekBytes[:])

	// Return current entry.
	c.keyBuf, c.valBuf = c.read()
}

// seekBuf moves the cursor to a position within the current buffer.
func (c *cursor) seekBuf(seek []byte) (key, value []byte) {
	for {
		// Slice off the current entry.
		buf := c.buf[c.off:]

		// Exit if current entry's timestamp is on or after the seek.
		if len(buf) == 0 {
			return
		}

		if bytes.Compare(buf[0:8], seek) != -1 {
			return
		}

		c.off += entryHeaderSize + entryDataSize(buf)
	}
}

// Next returns the next key/value pair from the cursor. If there are no values
// remaining, -1 is returned.
func (c *cursor) Next() (int64, interface{}) {
	for {
		k, v := func() (int64, interface{}) {
			if c.keyBuf != -2 {
				k, v := c.keyBuf, c.valBuf
				c.keyBuf = -2
				return k, v
			}

			// Ignore if there is no buffer.
			if len(c.buf) == 0 {
				return -1, nil
			}

			// Move forward to next entry.
			c.off += entryHeaderSize + entryDataSize(c.buf[c.off:])

			// If no items left then read first item from next block.
			if c.off >= len(c.buf) {
				_, v := c.cursor.Next()
				c.setBuf(v)
			}

			return c.read()
		}()

		if k != -1 && v == nil {
			// There is a point in the series at the next timestamp,
			// but not for this cursor's field. Go to the next point.
			continue
		}
		return k, v
	}
}

// setBuf saves a compressed block to the buffer.
func (c *cursor) setBuf(block []byte) {
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
		fmt.Printf("block decode error: %s\n", err)
	}

	c.buf, c.off = buf, 0
}

// read reads the current key and value from the current block.
func (c *cursor) read() (key int64, value interface{}) {
	// Return nil if the offset is at the end of the buffer.
	if c.off >= len(c.buf) {
		return -1, nil
	}

	// Otherwise read the current entry.
	buf := c.buf[c.off:]
	dataSize := entryDataSize(buf)

	return tsdb.DecodeKeyValue(c.field, c.dec, buf[0:8], buf[entryHeaderSize:entryHeaderSize+dataSize])
}

// Sort bz1 cursors in correct order for writing to TSM files.

type cursors []*cursor

func (a cursors) Len() int      { return len(a) }
func (a cursors) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a cursors) Less(i, j int) bool {
	if a[i].series == a[j].series {
		return a[i].field < a[j].field
	}
	return a[i].series < a[j].series
}

// entryHeaderSize is the number of bytes required for the header.
const entryHeaderSize = 8 + 4

// entryDataSize returns the size of an entry's data field, in bytes.
func entryDataSize(v []byte) int { return int(binary.BigEndian.Uint32(v[8:12])) }
