package tsdb

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/tsdb/internal"

	"github.com/boltdb/bolt"
	"github.com/gogo/protobuf/proto"
)

var (
	// ErrFieldOverflow is returned when too many fields are created on a measurement.
	ErrFieldOverflow = errors.New("field overflow")

	// ErrFieldTypeConflict is returned when a new field already exists with a different type.
	ErrFieldTypeConflict = errors.New("field type conflict")

	// ErrFieldNotFound is returned when a field cannot be found.
	ErrFieldNotFound = errors.New("field not found")

	// ErrFieldUnmappedID is returned when the system is presented, during decode, with a field ID
	// there is no mapping for.
	ErrFieldUnmappedID = errors.New("field ID not mapped")

	// ErrWALPartitionNotFound is returns when flushing a WAL partition that
	// does not exist.
	ErrWALPartitionNotFound = errors.New("wal partition not found")
)

// topLevelBucketN is the number of non-series buckets in the bolt db.
const topLevelBucketN = 3

// Shard represents a self-contained time series database. An inverted index of
// the measurement and tag data is kept along with the raw time series data.
// Data can be split across many shards. The query engine in TSDB is responsible
// for combining the output of many shards into a single query result.
type Shard struct {
	db    *bolt.DB // underlying data store
	index *DatabaseIndex
	path  string
	cache map[uint8]map[string][][]byte // values by <wal partition,series>

	walSize    int           // approximate size of the WAL, in bytes
	flush      chan struct{} // signals background flush
	flushTimer *time.Timer   // signals time-based flush

	mu                sync.RWMutex
	measurementFields map[string]*measurementFields // measurement name to their fields

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

// NewShard returns a new initialized Shard
func NewShard(index *DatabaseIndex, path string) *Shard {
	s := &Shard{
		index:             index,
		path:              path,
		flush:             make(chan struct{}, 1),
		measurementFields: make(map[string]*measurementFields),

		MaxWALSize:             DefaultMaxWALSize,
		WALFlushInterval:       DefaultWALFlushInterval,
		WALPartitionFlushDelay: DefaultWALPartitionFlushDelay,

		LogOutput: os.Stderr,
	}

	// Initialize all partitions of the cache.
	s.cache = make(map[uint8]map[string][][]byte)
	for i := uint8(0); i < WALPartitionN; i++ {
		s.cache[i] = make(map[string][][]byte)
	}

	return s
}

// Path returns the path set on the shard when it was created.
func (s *Shard) Path() string { return s.path }

// open initializes and opens the shard's store.
func (s *Shard) Open() error {
	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Return if the shard is already open
		if s.db != nil {
			return nil
		}

		// Open store on shard.
		store, err := bolt.Open(s.path, 0666, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			return err
		}
		s.db = store

		// Initialize store.
		if err := s.db.Update(func(tx *bolt.Tx) error {
			_, _ = tx.CreateBucketIfNotExists([]byte("series"))
			_, _ = tx.CreateBucketIfNotExists([]byte("fields"))
			_, _ = tx.CreateBucketIfNotExists([]byte("wal"))

			return nil
		}); err != nil {
			return fmt.Errorf("init: %s", err)
		}

		if err := s.loadMetadataIndex(); err != nil {
			return fmt.Errorf("load metadata index: %s", err)
		}

		// Initialize logger.
		s.logger = log.New(s.LogOutput, "[shard] ", log.LstdFlags)

		// Start flush interval timer.
		s.flushTimer = time.NewTimer(s.WALFlushInterval)

		// Start background goroutines.
		s.wg.Add(1)
		s.closing = make(chan struct{})
		go s.autoflusher(s.closing)

		return nil
	}(); err != nil {
		s.close()
		return err
	}

	// Flush on-disk WAL before we return to the caller.
	if err := s.Flush(0); err != nil {
		return fmt.Errorf("flush: %s", err)
	}

	return nil
}

// Close shuts down the shard's store.
func (s *Shard) Close() error {
	s.mu.Lock()
	err := s.close()
	s.mu.Unlock()

	// Wait for open goroutines to finish.
	s.wg.Wait()

	return err
}

func (s *Shard) close() error {
	if s.db != nil {
		s.db.Close()
	}
	if s.closing != nil {
		close(s.closing)
		s.closing = nil
	}
	return nil
}

// TODO: this is temporarily exported to make tx.go work. When the query engine gets refactored
// into the tsdb package this should be removed. No one outside tsdb should know the underlying store.
func (s *Shard) DB() *bolt.DB {
	return s.db
}

// TODO: this is temporarily exported to make tx.go work. When the query engine gets refactored
// into the tsdb package this should be removed. No one outside tsdb should know the underlying field encoding scheme.
func (s *Shard) FieldCodec(measurementName string) *FieldCodec {
	s.mu.RLock()
	defer s.mu.RUnlock()
	m := s.measurementFields[measurementName]
	if m == nil {
		return nil
	}
	return m.codec
}

// struct to hold information for a field to create on a measurement
type fieldCreate struct {
	measurement string
	field       *field
}

// struct to hold information for a series to create
type seriesCreate struct {
	measurement string
	series      *Series
}

// WritePoints will write the raw data points and any new metadata to the index in the shard
func (s *Shard) WritePoints(points []Point) error {
	seriesToCreate, fieldsToCreate, err := s.validateSeriesAndFields(points)
	if err != nil {
		return err
	}

	// add any new series to the in-memory index
	if len(seriesToCreate) > 0 {
		s.index.mu.Lock()
		for _, ss := range seriesToCreate {
			s.index.createSeriesIndexIfNotExists(ss.measurement, ss.series)
		}
		s.index.mu.Unlock()
	}

	// add any new fields and keep track of what needs to be saved
	measurementFieldsToSave, err := s.createFieldsAndMeasurements(fieldsToCreate)
	if err != nil {
		return err
	}

	// make sure all data is encoded before attempting to save to bolt
	for _, p := range points {
		// marshal the raw data if it hasn't been marshaled already
		if p.Data() == nil {
			// this was populated earlier, don't need to validate that it's there.
			s.mu.RLock()
			mf := s.measurementFields[p.Name()]
			s.mu.RUnlock()

			// If a measurement is dropped while writes for it are in progress, this could be nil
			if mf == nil {
				return ErrFieldNotFound
			}

			data, err := mf.codec.EncodeFields(p.Fields())
			if err != nil {
				return err
			}
			p.SetData(data)
		}
	}

	// save to the underlying bolt instance
	if err := s.db.Update(func(tx *bolt.Tx) error {
		// save any new metadata
		if len(seriesToCreate) > 0 {
			b := tx.Bucket([]byte("series"))
			for _, sc := range seriesToCreate {
				data, err := sc.series.MarshalBinary()
				if err != nil {
					return err
				}
				if err := b.Put([]byte(sc.series.Key), data); err != nil {
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
		s.mu.Lock()
		defer s.mu.Unlock()

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
			a := s.cache[partitionID][string(key)]
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

			s.cache[partitionID][string(key)] = a

			// Calculate estimated WAL size.
			s.walSize += len(key) + len(v)
		}

		// Sort by timestamp if not appending.
		for partitionID, cache := range resorts {
			for key, _ := range cache {
				sort.Sort(byteSlices(s.cache[partitionID][key]))
			}
		}

		// Check for flush threshold.
		s.triggerAutoFlush()

		return nil
	}(); err != nil {
		return err
	}

	return nil
}

// Flush writes all points from the write ahead log to the index.
func (s *Shard) Flush(partitionFlushDelay time.Duration) error {
	// Retrieve a list of WAL buckets.
	var partitionIDs []uint8
	if err := s.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket([]byte("wal")).ForEach(func(key, _ []byte) error {
			partitionIDs = append(partitionIDs, uint8(key[0]))
			return nil
		})
	}); err != nil {
		return err
	}

	// Continue flushing until there are no more partition buckets.
	for _, partitionID := range partitionIDs {
		if err := s.FlushPartition(partitionID); err != nil {
			return fmt.Errorf("flush partition: id=%d, err=%s", partitionID, err)
		}

		// Wait momentarily so other threads can process.
		time.Sleep(partitionFlushDelay)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Reset WAL size.
	s.walSize = 0

	// Reset the timer.
	s.flushTimer.Reset(s.WALFlushInterval)

	return nil
}

// FlushPartition flushes a single WAL partition.
func (s *Shard) FlushPartition(partitionID uint8) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	startTime := time.Now()

	var pointN int
	if err := s.db.Update(func(tx *bolt.Tx) error {
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
	s.cache[partitionID] = make(map[string][][]byte)

	if pointN > 0 {
		s.logger.Printf("flush %d points in %.3fs", pointN, time.Since(startTime).Seconds())
	}

	return nil
}

// autoflusher waits for notification of a flush and kicks it off in the background.
// This method runs in a separate goroutine.
func (s *Shard) autoflusher(closing chan struct{}) {
	defer s.wg.Done()

	for {
		// Wait for close or flush signal.
		select {
		case <-closing:
			return
		case <-s.flushTimer.C:
			if err := s.Flush(s.WALPartitionFlushDelay); err != nil {
				s.logger.Printf("flush error: %s", err)
			}
		case <-s.flush:
			if err := s.Flush(s.WALPartitionFlushDelay); err != nil {
				s.logger.Printf("flush error: %s", err)
			}
		}
	}
}

// triggerAutoFlush signals that a flush should occur if the size is above the threshold.
// This function must be called within the context of a lock.
func (s *Shard) triggerAutoFlush() {
	// Ignore if we haven't reached the threshold.
	if s.walSize < s.MaxWALSize {
		return
	}

	// Otherwise send a non-blocking signal.
	select {
	case s.flush <- struct{}{}:
	default:
	}
}

func (s *Shard) ValidateAggregateFieldsInStatement(measurementName string, stmt *influxql.SelectStatement) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	validateType := func(aname, fname string, t influxql.DataType) error {
		if t != influxql.Float && t != influxql.Integer {
			return fmt.Errorf("aggregate '%s' requires numerical field values. Field '%s' is of type %s",
				aname, fname, t)
		}
		return nil
	}

	m := s.measurementFields[measurementName]
	if m == nil {
		return fmt.Errorf("measurement not found: %s", measurementName)
	}

	// If a numerical aggregate is requested, ensure it is only performed on numeric data or on a
	// nested aggregate on numeric data.
	for _, a := range stmt.FunctionCalls() {
		// Check for fields like `derivative(mean(value), 1d)`
		var nested *influxql.Call = a
		if fn, ok := nested.Args[0].(*influxql.Call); ok {
			nested = fn
		}

		switch lit := nested.Args[0].(type) {
		case *influxql.VarRef:
			if influxql.IsNumeric(nested) {
				f := m.Fields[lit.Val]
				if err := validateType(a.Name, f.Name, f.Type); err != nil {
					return err
				}
			}
		case *influxql.Distinct:
			if nested.Name != "count" {
				return fmt.Errorf("aggregate call didn't contain a field %s", a.String())
			}
			if influxql.IsNumeric(nested) {
				f := m.Fields[lit.Val]
				if err := validateType(a.Name, f.Name, f.Type); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("aggregate call didn't contain a field %s", a.String())
		}
	}

	return nil
}

// deleteSeries deletes the buckets and the metadata for the given series keys
func (s *Shard) deleteSeries(keys []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("series"))
		for _, k := range keys {
			if err := b.Delete([]byte(k)); err != nil {
				return err
			}
			if err := tx.DeleteBucket([]byte(k)); err != nil && err != bolt.ErrBucketNotFound {
				return err
			}
			delete(s.cache[WALPartition([]byte(k))], k)
		}
		return nil
	}); err != nil {
		return err
	}

	return nil
}

// deleteMeasurement deletes the measurement field encoding information and all underlying series from the shard
func (s *Shard) deleteMeasurement(name string, seriesKeys []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.db.Update(func(tx *bolt.Tx) error {
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
			delete(s.cache[WALPartition([]byte(k))], k)
		}

		return nil
	}); err != nil {
		return err
	}

	// Remove entry from shard index.
	delete(s.measurementFields, name)
	return nil
}

func (s *Shard) createFieldsAndMeasurements(fieldsToCreate []*fieldCreate) (map[string]*measurementFields, error) {
	if len(fieldsToCreate) == 0 {
		return nil, nil
	}

	s.index.mu.Lock()
	s.mu.Lock()
	defer s.index.mu.Unlock()
	defer s.mu.Unlock()

	// add fields
	measurementsToSave := make(map[string]*measurementFields)
	for _, f := range fieldsToCreate {

		m := s.measurementFields[f.measurement]
		if m == nil {
			m = measurementsToSave[f.measurement]
			if m == nil {
				m = &measurementFields{Fields: make(map[string]*field)}
			}
			s.measurementFields[f.measurement] = m
		}

		measurementsToSave[f.measurement] = m

		// add the field to the in memory index
		if err := m.createFieldIfNotExists(f.field.Name, f.field.Type); err != nil {
			return nil, err
		}

		// ensure the measurement is in the index and the field is there
		measurement := s.index.createMeasurementIndexIfNotExists(f.measurement)
		measurement.fieldNames[f.field.Name] = struct{}{}
	}

	return measurementsToSave, nil
}

// validateSeriesAndFields checks which series and fields are new and whose metadata should be saved and indexed
func (s *Shard) validateSeriesAndFields(points []Point) ([]*seriesCreate, []*fieldCreate, error) {
	var seriesToCreate []*seriesCreate
	var fieldsToCreate []*fieldCreate

	// get the mutex for the in memory index, which is shared across shards
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// get the shard mutex for locally defined fields
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, p := range points {
		// see if the series should be added to the index
		if ss := s.index.series[string(p.Key())]; ss == nil {
			series := &Series{Key: string(p.Key()), Tags: p.Tags()}
			seriesToCreate = append(seriesToCreate, &seriesCreate{p.Name(), series})
		}

		// see if the field definitions need to be saved to the shard
		mf := s.measurementFields[p.Name()]
		if mf == nil {
			for name, value := range p.Fields() {
				fieldsToCreate = append(fieldsToCreate, &fieldCreate{p.Name(), &field{Name: name, Type: influxql.InspectDataType(value)}})
			}
			continue // skip validation since all fields are new
		}

		// validate field types and encode data
		for name, value := range p.Fields() {
			if f := mf.Fields[name]; f != nil {
				// Field present in shard metadata, make sure there is no type conflict.
				if f.Type != influxql.InspectDataType(value) {
					return nil, nil, fmt.Errorf("field type conflict: input field \"%s\" on measurement \"%s\" is type %T, already exists as type %s", name, p.Name(), value, f.Type)
				}

				continue // Field is present, and it's of the same type. Nothing more to do.
			}

			fieldsToCreate = append(fieldsToCreate, &fieldCreate{p.Name(), &field{Name: name, Type: influxql.InspectDataType(value)}})
		}
	}

	return seriesToCreate, fieldsToCreate, nil
}

// loadsMetadataIndex loads the shard metadata into memory. This should only be called by Open
func (s *Shard) loadMetadataIndex() error {
	return s.db.View(func(tx *bolt.Tx) error {
		s.index.mu.Lock()
		defer s.index.mu.Unlock()

		// load measurement metadata
		meta := tx.Bucket([]byte("fields"))
		c := meta.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			m := s.index.createMeasurementIndexIfNotExists(string(k))
			mf := &measurementFields{}
			if err := mf.UnmarshalBinary(v); err != nil {
				return err
			}
			for name, _ := range mf.Fields {
				m.fieldNames[name] = struct{}{}
			}
			mf.codec = newFieldCodec(mf.Fields)
			s.measurementFields[m.Name] = mf
		}

		// load series metadata
		meta = tx.Bucket([]byte("series"))
		c = meta.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			series := &Series{}
			if err := series.UnmarshalBinary(v); err != nil {
				return err
			}
			s.index.createSeriesIndexIfNotExists(measurementFromSeriesKey(string(k)), series)
		}
		return nil
	})
}

// SeriesCount returns the number of series buckets on the shard.
// This does not include a count from the WAL.
func (s *Shard) SeriesCount() (n int, err error) {
	err = s.db.View(func(tx *bolt.Tx) error {
		return tx.ForEach(func(_ []byte, _ *bolt.Bucket) error {
			n++
			return nil
		})
	})

	// Remove top-level buckets.
	n -= topLevelBucketN

	return
}

type measurementFields struct {
	Fields map[string]*field `json:"fields"`
	codec  *FieldCodec
}

// MarshalBinary encodes the object to a binary format.
func (m *measurementFields) MarshalBinary() ([]byte, error) {
	var pb internal.MeasurementFields
	for _, f := range m.Fields {
		id := int32(f.ID)
		name := f.Name
		t := int32(f.Type)
		pb.Fields = append(pb.Fields, &internal.Field{ID: &id, Name: &name, Type: &t})
	}
	return proto.Marshal(&pb)
}

// UnmarshalBinary decodes the object from a binary format.
func (m *measurementFields) UnmarshalBinary(buf []byte) error {
	var pb internal.MeasurementFields
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}
	m.Fields = make(map[string]*field)
	for _, f := range pb.Fields {
		m.Fields[f.GetName()] = &field{ID: uint8(f.GetID()), Name: f.GetName(), Type: influxql.DataType(f.GetType())}
	}
	return nil
}

// createFieldIfNotExists creates a new field with an autoincrementing ID.
// Returns an error if 255 fields have already been created on the measurement or
// the fields already exists with a different type.
func (m *measurementFields) createFieldIfNotExists(name string, typ influxql.DataType) error {
	// Ignore if the field already exists.
	if f := m.Fields[name]; f != nil {
		if f.Type != typ {
			return ErrFieldTypeConflict
		}
		return nil
	}

	// Only 255 fields are allowed. If we go over that then return an error.
	if len(m.Fields)+1 > math.MaxUint8 {
		return ErrFieldOverflow
	}

	// Create and append a new field.
	f := &field{
		ID:   uint8(len(m.Fields) + 1),
		Name: name,
		Type: typ,
	}
	m.Fields[name] = f
	m.codec = newFieldCodec(m.Fields)

	return nil
}

// Field represents a series field.
type field struct {
	ID   uint8             `json:"id,omitempty"`
	Name string            `json:"name,omitempty"`
	Type influxql.DataType `json:"type,omitempty"`
}

// FieldCodec provides encoding and decoding functionality for the fields of a given
// Measurement. It is a distinct type to avoid locking writes on this node while
// potentially long-running queries are executing.
//
// It is not affected by changes to the Measurement object after codec creation.
// TODO: this shouldn't be exported. nothing outside the shard should know about field encodings.
//       However, this is here until tx.go and the engine get refactored into tsdb.
type FieldCodec struct {
	fieldsByID   map[uint8]*field
	fieldsByName map[string]*field
}

// NewFieldCodec returns a FieldCodec for the given Measurement. Must be called with
// a RLock that protects the Measurement.
func newFieldCodec(fields map[string]*field) *FieldCodec {
	fieldsByID := make(map[uint8]*field, len(fields))
	fieldsByName := make(map[string]*field, len(fields))
	for _, f := range fields {
		fieldsByID[f.ID] = f
		fieldsByName[f.Name] = f
	}
	return &FieldCodec{fieldsByID: fieldsByID, fieldsByName: fieldsByName}
}

// EncodeFields converts a map of values with string keys to a byte slice of field
// IDs and values.
//
// If a field exists in the codec, but its type is different, an error is returned. If
// a field is not present in the codec, the system panics.
func (f *FieldCodec) EncodeFields(values map[string]interface{}) ([]byte, error) {
	// Allocate byte slice
	b := make([]byte, 0, 10)

	for k, v := range values {
		field := f.fieldsByName[k]
		if field == nil {
			panic(fmt.Sprintf("field does not exist for %s", k))
		} else if influxql.InspectDataType(v) != field.Type {
			return nil, fmt.Errorf("field \"%s\" is type %T, mapped as type %s", k, v, field.Type)
		}

		var buf []byte

		switch field.Type {
		case influxql.Float:
			value := v.(float64)
			buf = make([]byte, 9)
			binary.BigEndian.PutUint64(buf[1:9], math.Float64bits(value))
		case influxql.Integer:
			var value uint64
			switch v.(type) {
			case int:
				value = uint64(v.(int))
			case int32:
				value = uint64(v.(int32))
			case int64:
				value = uint64(v.(int64))
			default:
				panic(fmt.Sprintf("invalid integer type: %T", v))
			}
			buf = make([]byte, 9)
			binary.BigEndian.PutUint64(buf[1:9], value)
		case influxql.Boolean:
			value := v.(bool)

			// Only 1 byte need for a boolean.
			buf = make([]byte, 2)
			if value {
				buf[1] = byte(1)
			}
		case influxql.String:
			value := v.(string)
			if len(value) > maxStringLength {
				value = value[:maxStringLength]
			}
			// Make a buffer for field ID (1 bytes), the string length (2 bytes), and the string.
			buf = make([]byte, len(value)+3)

			// Set the string length, then copy the string itself.
			binary.BigEndian.PutUint16(buf[1:3], uint16(len(value)))
			for i, c := range []byte(value) {
				buf[i+3] = byte(c)
			}
		default:
			panic(fmt.Sprintf("unsupported value type during encode fields: %T", v))
		}

		// Always set the field ID as the leading byte.
		buf[0] = field.ID

		// Append temp buffer to the end.
		b = append(b, buf...)
	}

	return b, nil
}

// TODO: this shouldn't be exported. remove when tx.go and engine.go get refactored into tsdb
func (f *FieldCodec) FieldIDByName(s string) (uint8, error) {
	fi := f.fieldsByName[s]
	if fi == nil {
		return 0, ErrFieldNotFound
	}
	return fi.ID, nil
}

// DecodeFields decodes a byte slice into a set of field ids and values.
func (f *FieldCodec) DecodeFields(b []byte) (map[uint8]interface{}, error) {
	if len(b) == 0 {
		return nil, nil
	}

	// Create a map to hold the decoded data.
	values := make(map[uint8]interface{}, 0)

	for {
		if len(b) < 1 {
			// No more bytes.
			break
		}

		// First byte is the field identifier.
		fieldID := b[0]
		field := f.fieldsByID[fieldID]
		if field == nil {
			// See note in DecodeByID() regarding field-mapping failures.
			return nil, ErrFieldUnmappedID
		}

		var value interface{}
		switch field.Type {
		case influxql.Float:
			value = math.Float64frombits(binary.BigEndian.Uint64(b[1:9]))
			// Move bytes forward.
			b = b[9:]
		case influxql.Integer:
			value = int64(binary.BigEndian.Uint64(b[1:9]))
			// Move bytes forward.
			b = b[9:]
		case influxql.Boolean:
			if b[1] == 1 {
				value = true
			} else {
				value = false
			}
			// Move bytes forward.
			b = b[2:]
		case influxql.String:
			size := binary.BigEndian.Uint16(b[1:3])
			value = string(b[3 : size+3])
			// Move bytes forward.
			b = b[size+3:]
		default:
			panic(fmt.Sprintf("unsupported value type during decode fields: %T", f.fieldsByID[fieldID]))
		}

		values[fieldID] = value

	}

	return values, nil
}

// DecodeFieldsWithNames decodes a byte slice into a set of field names and values
// TODO: shouldn't be exported. refactor engine
func (f *FieldCodec) DecodeFieldsWithNames(b []byte) (map[string]interface{}, error) {
	fields, err := f.DecodeFields(b)
	if err != nil {
		return nil, err
	}
	m := make(map[string]interface{})
	for id, v := range fields {
		field := f.fieldsByID[id]
		if field != nil {
			m[field.Name] = v
		}
	}
	return m, nil
}

// DecodeByID scans a byte slice for a field with the given ID, converts it to its
// expected type, and return that value.
// TODO: shouldn't be exported. refactor engine
func (f *FieldCodec) DecodeByID(targetID uint8, b []byte) (interface{}, error) {
	if len(b) == 0 {
		return 0, ErrFieldNotFound
	}

	for {
		if len(b) < 1 {
			// No more bytes.
			break
		}
		field, ok := f.fieldsByID[b[0]]
		if !ok {
			// This can happen, though is very unlikely. If this node receives encoded data, to be written
			// to disk, and is queried for that data before its metastore is updated, there will be no field
			// mapping for the data during decode. All this can happen because data is encoded by the node
			// that first received the write request, not the node that actually writes the data to disk.
			// So if this happens, the read must be aborted.
			return 0, ErrFieldUnmappedID
		}

		var value interface{}
		switch field.Type {
		case influxql.Float:
			// Move bytes forward.
			value = math.Float64frombits(binary.BigEndian.Uint64(b[1:9]))
			b = b[9:]
		case influxql.Integer:
			value = int64(binary.BigEndian.Uint64(b[1:9]))
			b = b[9:]
		case influxql.Boolean:
			if b[1] == 1 {
				value = true
			} else {
				value = false
			}
			// Move bytes forward.
			b = b[2:]
		case influxql.String:
			size := binary.BigEndian.Uint16(b[1:3])
			value = string(b[3 : 3+size])
			// Move bytes forward.
			b = b[size+3:]
		default:
			panic(fmt.Sprintf("unsupported value type during decode by id: %T", field.Type))
		}

		if field.ID == targetID {
			return value, nil
		}
	}

	return 0, ErrFieldNotFound
}

// DecodeByName scans a byte slice for a field with the given name, converts it to its
// expected type, and return that value.
func (f *FieldCodec) DecodeByName(name string, b []byte) (interface{}, error) {
	if fi := f.fieldByName(name); fi == nil {
		return 0, ErrFieldNotFound
	} else {
		return f.DecodeByID(fi.ID, b)
	}
}

// FieldByName returns the field by its name. It will return a nil if not found
func (f *FieldCodec) fieldByName(name string) *field {
	return f.fieldsByName[name]
}

// mustMarshal encodes a value to JSON.
// This will panic if an error occurs. This should only be used internally when
// an invalid marshal will cause corruption and a panic is appropriate.
func mustMarshalJSON(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic("marshal: " + err.Error())
	}
	return b
}

// mustUnmarshalJSON decodes a value from JSON.
// This will panic if an error occurs. This should only be used internally when
// an invalid unmarshal will cause corruption and a panic is appropriate.
func mustUnmarshalJSON(b []byte, v interface{}) {
	if err := json.Unmarshal(b, v); err != nil {
		panic("unmarshal: " + err.Error())
	}
}

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
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

// byteSlices represents a sortable slice of byte slices.
type byteSlices [][]byte

func (a byteSlices) Len() int           { return len(a) }
func (a byteSlices) Less(i, j int) bool { return bytes.Compare(a[i], a[j]) == -1 }
func (a byteSlices) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// shardCursor provides ordered iteration across a Bolt bucket and shard cache.
type shardCursor struct {
	// Bolt cursor and readahead buffer.
	cursor *bolt.Cursor
	buf    struct {
		key, value []byte
	}

	// Cache and current cache index.
	cache [][]byte
	index int
}

// Seek moves the cursor to a position and returns the closest key/value pair.
func (sc *shardCursor) Seek(seek []byte) (key, value []byte) {
	// Seek bolt cursor.
	if sc.cursor != nil {
		sc.buf.key, sc.buf.value = sc.cursor.Seek(seek)
	}

	// Seek cache index.
	sc.index = sort.Search(len(sc.cache), func(i int) bool {
		return bytes.Compare(sc.cache[i][0:8], seek) != -1
	})

	return sc.read()
}

// Next returns the next key/value pair from the cursor.
func (sc *shardCursor) Next() (key, value []byte) {
	// Read next bolt key/value if not bufferred.
	if sc.buf.key == nil && sc.cursor != nil {
		sc.buf.key, sc.buf.value = sc.cursor.Next()
	}

	return sc.read()
}

// read returns the next key/value in the cursor buffer or cache.
func (sc *shardCursor) read() (key, value []byte) {
	// If neither a buffer or cache exists then return nil.
	if sc.buf.key == nil && sc.index >= len(sc.cache) {
		return nil, nil
	}

	// Use the buffer if it exists and there's no cache or if it is lower than the cache.
	if sc.buf.key != nil && (sc.index >= len(sc.cache) || bytes.Compare(sc.buf.key, sc.cache[sc.index][0:8]) == -1) {
		key, value = sc.buf.key, sc.buf.value
		sc.buf.key, sc.buf.value = nil, nil
		return
	}

	// Otherwise read from the cache.
	// Continue skipping ahead through duplicate keys in the cache list.
	for {
		// Read the current cache key/value pair.
		key, value = sc.cache[sc.index][0:8], sc.cache[sc.index][8:]
		sc.index++

		// Exit loop if we're at the end of the cache or the next key is different.
		if sc.index >= len(sc.cache) || !bytes.Equal(key, sc.cache[sc.index][0:8]) {
			break
		}
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
