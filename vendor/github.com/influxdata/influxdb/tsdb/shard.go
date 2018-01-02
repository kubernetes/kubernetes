package tsdb

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	internal "github.com/influxdata/influxdb/tsdb/internal"
)

// monitorStatInterval is the interval at which the shard is inspected
// for the purpose of determining certain monitoring statistics.
const monitorStatInterval = 30 * time.Second

const (
	statWriteReq           = "writeReq"
	statWriteReqOK         = "writeReqOk"
	statWriteReqErr        = "writeReqErr"
	statSeriesCreate       = "seriesCreate"
	statFieldsCreate       = "fieldsCreate"
	statWritePointsErr     = "writePointsErr"
	statWritePointsDropped = "writePointsDropped"
	statWritePointsOK      = "writePointsOk"
	statWriteBytes         = "writeBytes"
	statDiskBytes          = "diskBytes"
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

	// ErrEngineClosed is returned when a caller attempts indirectly to
	// access the shard's underlying engine.
	ErrEngineClosed = errors.New("engine is closed")

	// ErrShardDisabled is returned when a the shard is not available for
	// queries or writes.
	ErrShardDisabled = errors.New("shard is disabled")
)

var (
	// Static objects to prevent small allocs.
	timeBytes = []byte("time")
)

// A ShardError implements the error interface, and contains extra
// context about the shard that generated the error.
type ShardError struct {
	id  uint64
	Err error
}

// NewShardError returns a new ShardError.
func NewShardError(id uint64, err error) error {
	if err == nil {
		return nil
	}
	return ShardError{id: id, Err: err}
}

func (e ShardError) Error() string {
	return fmt.Sprintf("[shard %d] %s", e.id, e.Err)
}

// PartialWriteError indicates a write request could only write a portion of the
// requested values.
type PartialWriteError struct {
	Reason  string
	Dropped int
}

func (e PartialWriteError) Error() string {
	return fmt.Sprintf("%s dropped=%d", e.Reason, e.Dropped)
}

// Shard represents a self-contained time series database. An inverted index of
// the measurement and tag data is kept along with the raw time series data.
// Data can be split across many shards. The query engine in TSDB is responsible
// for combining the output of many shards into a single query result.
type Shard struct {
	index   *DatabaseIndex
	path    string
	walPath string
	id      uint64

	database        string
	retentionPolicy string

	options EngineOptions

	mu      sync.RWMutex
	engine  Engine
	closing chan struct{}
	enabled bool

	// expvar-based stats.
	stats       *ShardStatistics
	defaultTags models.StatisticTags

	logger *log.Logger
	// used by logger. Referenced so it can be passed down to new caches.
	logOutput io.Writer

	EnableOnOpen bool
}

// NewShard returns a new initialized Shard. walPath doesn't apply to the b1 type index
func NewShard(id uint64, index *DatabaseIndex, path string, walPath string, options EngineOptions) *Shard {
	db, rp := DecodeStorePath(path)
	s := &Shard{
		index:   index,
		id:      id,
		path:    path,
		walPath: walPath,
		options: options,
		closing: make(chan struct{}),

		stats: &ShardStatistics{},
		defaultTags: models.StatisticTags{
			"path":            path,
			"walPath":         walPath,
			"id":              fmt.Sprintf("%d", id),
			"database":        db,
			"retentionPolicy": rp,
			"engine":          options.EngineVersion,
		},

		database:        db,
		retentionPolicy: rp,

		logger:       log.New(os.Stderr, "[shard] ", log.LstdFlags),
		logOutput:    os.Stderr,
		EnableOnOpen: true,
	}
	return s
}

// SetLogOutput sets the writer to which log output will be written. It is safe
// for concurrent use.
func (s *Shard) SetLogOutput(w io.Writer) {
	s.logger.SetOutput(w)
	if err := s.ready(); err == nil {
		s.engine.SetLogOutput(w)
	}

	s.mu.Lock()
	s.logOutput = w
	s.mu.Unlock()
}

// SetEnabled enables the shard for queries and write.  When disabled, all
// writes and queries return an error and compactions are stopped for the shard.
func (s *Shard) SetEnabled(enabled bool) {
	s.mu.Lock()
	// Prevent writes and queries
	s.enabled = enabled
	if s.engine != nil {
		// Disable background compactions and snapshotting
		s.engine.SetEnabled(enabled)
	}
	s.mu.Unlock()
}

// ShardStatistics maintains statistics for a shard.
type ShardStatistics struct {
	WriteReq           int64
	WriteReqOK         int64
	WriteReqErr        int64
	SeriesCreated      int64
	FieldsCreated      int64
	WritePointsErr     int64
	WritePointsDropped int64
	WritePointsOK      int64
	BytesWritten       int64
	DiskBytes          int64
}

// Statistics returns statistics for periodic monitoring.
func (s *Shard) Statistics(tags map[string]string) []models.Statistic {
	if err := s.ready(); err != nil {
		return nil
	}

	seriesN, _ := s.engine.SeriesCount()
	tags = s.defaultTags.Merge(tags)
	statistics := []models.Statistic{{
		Name: "shard",
		Tags: tags,
		Values: map[string]interface{}{
			statWriteReq:           atomic.LoadInt64(&s.stats.WriteReq),
			statWriteReqOK:         atomic.LoadInt64(&s.stats.WriteReqOK),
			statWriteReqErr:        atomic.LoadInt64(&s.stats.WriteReqErr),
			statSeriesCreate:       seriesN,
			statFieldsCreate:       atomic.LoadInt64(&s.stats.FieldsCreated),
			statWritePointsErr:     atomic.LoadInt64(&s.stats.WritePointsErr),
			statWritePointsDropped: atomic.LoadInt64(&s.stats.WritePointsDropped),
			statWritePointsOK:      atomic.LoadInt64(&s.stats.WritePointsOK),
			statWriteBytes:         atomic.LoadInt64(&s.stats.BytesWritten),
			statDiskBytes:          atomic.LoadInt64(&s.stats.DiskBytes),
		},
	}}
	statistics = append(statistics, s.engine.Statistics(tags)...)
	return statistics
}

// Path returns the path set on the shard when it was created.
func (s *Shard) Path() string { return s.path }

// Open initializes and opens the shard's store.
func (s *Shard) Open() error {
	if err := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()

		// Return if the shard is already open
		if s.engine != nil {
			return nil
		}

		// Initialize underlying engine.
		e, err := NewEngine(s.id, s.path, s.walPath, s.options)
		if err != nil {
			return err
		}

		// Set log output on the engine.
		e.SetLogOutput(s.logOutput)

		// Disable compactions while loading the index
		e.SetEnabled(false)

		// Open engine.
		if err := e.Open(); err != nil {
			return err
		}

		// Load metadata index.
		start := time.Now()
		if err := e.LoadMetadataIndex(s.id, s.index); err != nil {
			return err
		}

		count := s.index.SeriesShardN(s.id)
		atomic.AddInt64(&s.stats.SeriesCreated, int64(count))

		s.engine = e

		s.logger.Printf("%s database index loaded in %s", s.path, time.Now().Sub(start))

		go s.monitor()

		return nil
	}(); err != nil {
		s.close()
		return NewShardError(s.id, err)
	}

	if s.EnableOnOpen {
		// enable writes, queries and compactions
		s.SetEnabled(true)
	}

	return nil
}

// UnloadIndex removes all references to this shard from the DatabaseIndex
func (s *Shard) UnloadIndex() {
	// Don't leak our shard ID and series keys in the index
	s.index.RemoveShard(s.id)
}

// Close shuts down the shard's store.
func (s *Shard) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.close()
}

func (s *Shard) close() error {
	if s.engine == nil {
		return nil
	}

	// Close the closing channel at most once.
	select {
	case <-s.closing:
	default:
		close(s.closing)
	}

	// Don't leak our shard ID and series keys in the index
	s.UnloadIndex()

	err := s.engine.Close()
	if err == nil {
		s.engine = nil
	}
	return err
}

// ready determines if the Shard is ready for queries or writes.
// It returns nil if ready, otherwise ErrShardClosed or ErrShardDiabled
func (s *Shard) ready() error {
	var err error

	s.mu.RLock()
	if s.engine == nil {
		err = ErrEngineClosed
	} else if !s.enabled {
		err = ErrShardDisabled
	}
	s.mu.RUnlock()
	return err
}

// DiskSize returns the size on disk of this shard
func (s *Shard) DiskSize() (int64, error) {
	var size int64
	err := filepath.Walk(s.path, func(_ string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !fi.IsDir() {
			size += fi.Size()
		}
		return err
	})
	if err != nil {
		return 0, err
	}

	err = filepath.Walk(s.walPath, func(_ string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !fi.IsDir() {
			size += fi.Size()
		}
		return err
	})

	return size, err
}

// FieldCreate holds information for a field to create on a measurement
type FieldCreate struct {
	Measurement string
	Field       *Field
}

// SeriesCreate holds information for a series to create
type SeriesCreate struct {
	Measurement string
	Series      *Series
}

// WritePoints will write the raw data points and any new metadata to the index in the shard
func (s *Shard) WritePoints(points []models.Point) error {
	if err := s.ready(); err != nil {
		return err
	}

	var writeError error

	s.mu.RLock()
	defer s.mu.RUnlock()

	atomic.AddInt64(&s.stats.WriteReq, 1)

	points, fieldsToCreate, err := s.validateSeriesAndFields(points)
	if err != nil {
		if _, ok := err.(PartialWriteError); !ok {
			return err
		}
		// There was a partial write (points dropped), hold onto the error to return
		// to the caller, but continue on writing the remaining points.
		writeError = err
	}
	atomic.AddInt64(&s.stats.FieldsCreated, int64(len(fieldsToCreate)))

	// add any new fields and keep track of what needs to be saved
	if err := s.createFieldsAndMeasurements(fieldsToCreate); err != nil {
		return err
	}

	// Write to the engine.
	if err := s.engine.WritePoints(points); err != nil {
		atomic.AddInt64(&s.stats.WritePointsErr, int64(len(points)))
		atomic.AddInt64(&s.stats.WriteReqErr, 1)
		return fmt.Errorf("engine: %s", err)
	}
	atomic.AddInt64(&s.stats.WritePointsOK, int64(len(points)))
	atomic.AddInt64(&s.stats.WriteReqOK, 1)

	return writeError
}

// ContainsSeries determines if the shard contains the provided series keys. The
// returned map contains all the provided keys that are in the shard, and the
// value for each key will be true if the shard has values for that key.
func (s *Shard) ContainsSeries(seriesKeys []string) (map[string]bool, error) {
	if err := s.ready(); err != nil {
		return nil, err
	}

	return s.engine.ContainsSeries(seriesKeys)
}

// DeleteSeries deletes a list of series.
func (s *Shard) DeleteSeries(seriesKeys []string) error {
	if err := s.ready(); err != nil {
		return err
	}
	if err := s.engine.DeleteSeries(seriesKeys); err != nil {
		return err
	}
	return nil
}

// DeleteSeriesRange deletes all values from for seriesKeys between min and max (inclusive)
func (s *Shard) DeleteSeriesRange(seriesKeys []string, min, max int64) error {
	if err := s.ready(); err != nil {
		return err
	}

	if err := s.engine.DeleteSeriesRange(seriesKeys, min, max); err != nil {
		return err
	}

	return nil
}

// DeleteMeasurement deletes a measurement and all underlying series.
func (s *Shard) DeleteMeasurement(name string, seriesKeys []string) error {
	if err := s.ready(); err != nil {
		return err
	}

	if err := s.engine.DeleteMeasurement(name, seriesKeys); err != nil {
		return err
	}

	return nil
}

func (s *Shard) createFieldsAndMeasurements(fieldsToCreate []*FieldCreate) error {
	if len(fieldsToCreate) == 0 {
		return nil
	}

	// add fields
	for _, f := range fieldsToCreate {
		m := s.engine.MeasurementFields(f.Measurement)

		// Add the field to the in memory index
		if err := m.CreateFieldIfNotExists(f.Field.Name, f.Field.Type, false); err != nil {
			return err
		}

		// ensure the measurement is in the index and the field is there
		measurement := s.index.CreateMeasurementIndexIfNotExists(f.Measurement)
		measurement.SetFieldName(f.Field.Name)
	}

	return nil
}

// validateSeriesAndFields checks which series and fields are new and whose metadata should be saved and indexed
func (s *Shard) validateSeriesAndFields(points []models.Point) ([]models.Point, []*FieldCreate, error) {
	var (
		fieldsToCreate []*FieldCreate
		err            error
		dropped, n     int
		reason         string
	)
	if s.options.Config.MaxValuesPerTag > 0 {
		// Validate that all the new points would not exceed any limits, if so, we drop them
		// and record why/increment counters
		for i, p := range points {
			tags := p.Tags()
			m := s.index.Measurement(p.Name())

			// Measurement doesn't exist yet, can't check the limit
			if m != nil {
				var dropPoint bool
				for _, tag := range tags {
					// If the tag value already exists, skip the limit check
					if m.HasTagKeyValue(tag.Key, tag.Value) {
						continue
					}

					n := m.CardinalityBytes(tag.Key)
					if n >= s.options.Config.MaxValuesPerTag {
						dropPoint = true
						reason = fmt.Sprintf("max-values-per-tag limit exceeded (%d/%d): measurement=%q tag=%q value=%q",
							n, s.options.Config.MaxValuesPerTag, m.Name, tag.Key, tag.Value)
						break
					}
				}
				if dropPoint {
					atomic.AddInt64(&s.stats.WritePointsDropped, 1)
					dropped++

					// This causes n below to not be increment allowing the point to be dropped
					continue
				}
			}
			points[n] = points[i]
			n++
		}
		points = points[:n]
	}

	// get the shard mutex for locally defined fields
	n = 0
	for i, p := range points {
		// verify the tags and fields
		tags := p.Tags()
		if v := tags.Get(timeBytes); v != nil {
			s.logger.Printf("dropping tag 'time' from '%s'\n", p.PrecisionString(""))
			tags.Delete(timeBytes)
			p.SetTags(tags)
		}

		var validField bool
		iter := p.FieldIterator()
		for iter.Next() {
			if bytes.Equal(iter.FieldKey(), timeBytes) {
				s.logger.Printf("dropping field 'time' from '%s'\n", p.PrecisionString(""))
				iter.Delete()
				continue
			}
			validField = true
		}

		if !validField {
			continue
		}

		iter.Reset()

		// see if the series should be added to the index
		ss := s.index.SeriesBytes(p.Key())
		if ss == nil {
			if s.options.Config.MaxSeriesPerDatabase > 0 && s.index.SeriesN()+1 > s.options.Config.MaxSeriesPerDatabase {
				atomic.AddInt64(&s.stats.WritePointsDropped, 1)
				dropped++
				reason = fmt.Sprintf("max-series-per-database limit exceeded: db=%s (%d/%d)",
					s.database, s.index.SeriesN(), s.options.Config.MaxSeriesPerDatabase)
				continue
			}

			ss = s.index.CreateSeriesIndexIfNotExists(p.Name(), NewSeries(string(p.Key()), tags))
			atomic.AddInt64(&s.stats.SeriesCreated, 1)
		}

		if !ss.Assigned(s.id) {
			ss.AssignShard(s.id)
		}

		// see if the field definitions need to be saved to the shard
		mf := s.engine.MeasurementFields(p.Name())

		if mf == nil {
			var createType influxql.DataType
			for iter.Next() {
				switch iter.Type() {
				case models.Float:
					createType = influxql.Float
				case models.Integer:
					createType = influxql.Integer
				case models.String:
					createType = influxql.String
				case models.Boolean:
					createType = influxql.Boolean
				default:
					continue
				}
				fieldsToCreate = append(fieldsToCreate, &FieldCreate{p.Name(), &Field{Name: string(iter.FieldKey()), Type: createType}})
			}
			continue // skip validation since all fields are new
		}

		iter.Reset()

		// validate field types and encode data
		for iter.Next() {
			var fieldType influxql.DataType
			switch iter.Type() {
			case models.Float:
				fieldType = influxql.Float
			case models.Integer:
				fieldType = influxql.Integer
			case models.Boolean:
				fieldType = influxql.Boolean
			case models.String:
				fieldType = influxql.String
			default:
				continue
			}
			if f := mf.FieldBytes(iter.FieldKey()); f != nil {
				// Field present in shard metadata, make sure there is no type conflict.
				if f.Type != fieldType {
					return points, nil, fmt.Errorf("%s: input field \"%s\" on measurement \"%s\" is type %s, already exists as type %s", ErrFieldTypeConflict, iter.FieldKey(), p.Name(), fieldType, f.Type)
				}

				continue // Field is present, and it's of the same type. Nothing more to do.
			}

			fieldsToCreate = append(fieldsToCreate, &FieldCreate{p.Name(), &Field{Name: string(iter.FieldKey()), Type: fieldType}})
		}
		points[n] = points[i]
		n++
	}
	points = points[:n]

	if dropped > 0 {
		err = PartialWriteError{Reason: reason, Dropped: dropped}
	}

	return points, fieldsToCreate, err
}

// SeriesCount returns the number of series buckets on the shard.
func (s *Shard) SeriesCount() (int, error) {
	if err := s.ready(); err != nil {
		return 0, err
	}
	return s.engine.SeriesCount()
}

// WriteTo writes the shard's data to w.
func (s *Shard) WriteTo(w io.Writer) (int64, error) {
	if err := s.ready(); err != nil {
		return 0, err
	}
	n, err := s.engine.WriteTo(w)
	atomic.AddInt64(&s.stats.BytesWritten, int64(n))
	return n, err
}

// CreateIterator returns an iterator for the data in the shard.
func (s *Shard) CreateIterator(opt influxql.IteratorOptions) (influxql.Iterator, error) {
	if err := s.ready(); err != nil {
		return nil, err
	}

	if influxql.Sources(opt.Sources).HasSystemSource() {
		return s.createSystemIterator(opt)
	}
	opt.Sources = influxql.Sources(opt.Sources).Filter(s.database, s.retentionPolicy)
	return s.engine.CreateIterator(opt)
}

// createSystemIterator returns an iterator for a system source.
func (s *Shard) createSystemIterator(opt influxql.IteratorOptions) (influxql.Iterator, error) {
	// Only support a single system source.
	if len(opt.Sources) > 1 {
		return nil, errors.New("cannot select from multiple system sources")
	}

	m := opt.Sources[0].(*influxql.Measurement)
	switch m.Name {
	case "_fieldKeys":
		return NewFieldKeysIterator(s, opt)
	case "_series":
		return NewSeriesIterator(s, opt)
	case "_tagKeys":
		return NewTagKeysIterator(s, opt)
	default:
		return nil, fmt.Errorf("unknown system source: %s", m.Name)
	}
}

// FieldDimensions returns unique sets of fields and dimensions across a list of sources.
func (s *Shard) FieldDimensions(sources influxql.Sources) (fields map[string]influxql.DataType, dimensions map[string]struct{}, err error) {
	if err := s.ready(); err != nil {
		return nil, nil, err
	}

	if sources.HasSystemSource() {
		// Only support a single system source.
		if len(sources) > 1 {
			return nil, nil, errors.New("cannot select from multiple system sources")
		}

		switch m := sources[0].(type) {
		case *influxql.Measurement:
			switch m.Name {
			case "_fieldKeys":
				return map[string]influxql.DataType{
					"fieldKey":  influxql.String,
					"fieldType": influxql.String,
				}, nil, nil
			case "_series":
				return map[string]influxql.DataType{
					"key": influxql.String,
				}, nil, nil
			case "_tagKeys":
				return map[string]influxql.DataType{
					"tagKey": influxql.String,
				}, nil, nil
			}
		}
		return nil, nil, nil
	}

	fields = make(map[string]influxql.DataType)
	dimensions = make(map[string]struct{})

	for _, src := range sources {
		switch m := src.(type) {
		case *influxql.Measurement:
			// Retrieve measurement.
			mm := s.index.Measurement(m.Name)
			if mm == nil {
				continue
			}

			// Append fields and dimensions.
			mf := s.engine.MeasurementFields(m.Name)
			if mf != nil {
				for name, typ := range mf.FieldSet() {
					fields[name] = typ
				}
			}
			for _, key := range mm.TagKeys() {
				dimensions[key] = struct{}{}
			}
		}
	}

	return
}

// ExpandSources expands regex sources and removes duplicates.
// NOTE: sources must be normalized (db and rp set) before calling this function.
func (s *Shard) ExpandSources(sources influxql.Sources) (influxql.Sources, error) {
	// Use a map as a set to prevent duplicates.
	set := map[string]influxql.Source{}

	// Iterate all sources, expanding regexes when they're found.
	for _, source := range sources {
		switch src := source.(type) {
		case *influxql.Measurement:
			// Add non-regex measurements directly to the set.
			if src.Regex == nil {
				set[src.String()] = src
				continue
			}

			// Loop over matching measurements.
			for _, m := range s.index.MeasurementsByRegex(src.Regex.Val) {
				other := &influxql.Measurement{
					Database:        src.Database,
					RetentionPolicy: src.RetentionPolicy,
					Name:            m.Name,
				}
				set[other.String()] = other
			}

		default:
			return nil, fmt.Errorf("expandSources: unsupported source type: %T", source)
		}
	}

	// Convert set to sorted slice.
	names := make([]string, 0, len(set))
	for name := range set {
		names = append(names, name)
	}
	sort.Strings(names)

	// Convert set to a list of Sources.
	expanded := make(influxql.Sources, 0, len(set))
	for _, name := range names {
		expanded = append(expanded, set[name])
	}

	return expanded, nil
}

// Restore restores data to the underlying engine for the shard.
// The shard is reopened after restore.
func (s *Shard) Restore(r io.Reader, basePath string) error {
	s.mu.Lock()

	// Restore to engine.
	if err := s.engine.Restore(r, basePath); err != nil {
		s.mu.Unlock()
		return err
	}

	s.mu.Unlock()

	// Close shard.
	if err := s.Close(); err != nil {
		return err
	}

	// Reopen engine.
	return s.Open()
}

// CreateSnapshot will return a path to a temp directory
// containing hard links to the underlying shard files
func (s *Shard) CreateSnapshot() (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.engine.CreateSnapshot()
}

func (s *Shard) monitor() {
	t := time.NewTicker(monitorStatInterval)
	defer t.Stop()
	t2 := time.NewTicker(time.Minute)
	defer t2.Stop()
	for {
		select {
		case <-s.closing:
			return
		case <-t.C:
			size, err := s.DiskSize()
			if err != nil {
				s.logger.Printf("Error collecting shard size: %v", err)
				continue
			}
			atomic.StoreInt64(&s.stats.DiskBytes, size)
		case <-t2.C:
			if s.options.Config.MaxValuesPerTag == 0 {
				continue
			}

			for _, m := range s.index.Measurements() {
				for _, k := range m.TagKeys() {
					n := m.Cardinality(k)
					perc := int(float64(n) / float64(s.options.Config.MaxValuesPerTag) * 100)
					if perc > 100 {
						perc = 100
					}

					// Log at 80, 85, 90-100% levels
					if perc == 80 || perc == 85 || perc >= 90 {
						s.logger.Printf("WARN: %d%% of max-values-per-tag limit exceeded: (%d/%d), db=%s shard=%d measurement=%s tag=%s",
							perc, n, s.options.Config.MaxValuesPerTag, s.database, s.id, m.Name, k)
					}
				}
			}
		}
	}
}

// Shards represents a sortable list of shards.
type Shards []*Shard

func (a Shards) Len() int           { return len(a) }
func (a Shards) Less(i, j int) bool { return a[i].id < a[j].id }
func (a Shards) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// MeasurementFields holds the fields of a measurement and their codec.
type MeasurementFields struct {
	mu sync.RWMutex

	fields map[string]*Field
}

// NewMeasurementFields returns an initialised *MeasurementFields value.
func NewMeasurementFields() *MeasurementFields {
	return &MeasurementFields{fields: make(map[string]*Field)}
}

// MarshalBinary encodes the object to a binary format.
func (m *MeasurementFields) MarshalBinary() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var pb internal.MeasurementFields
	for _, f := range m.fields {
		id := int32(f.ID)
		name := f.Name
		t := int32(f.Type)
		pb.Fields = append(pb.Fields, &internal.Field{ID: &id, Name: &name, Type: &t})
	}
	return proto.Marshal(&pb)
}

// UnmarshalBinary decodes the object from a binary format.
func (m *MeasurementFields) UnmarshalBinary(buf []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var pb internal.MeasurementFields
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}
	m.fields = make(map[string]*Field, len(pb.Fields))
	for _, f := range pb.Fields {
		m.fields[f.GetName()] = &Field{ID: uint8(f.GetID()), Name: f.GetName(), Type: influxql.DataType(f.GetType())}
	}
	return nil
}

// CreateFieldIfNotExists creates a new field with an autoincrementing ID.
// Returns an error if 255 fields have already been created on the measurement or
// the fields already exists with a different type.
func (m *MeasurementFields) CreateFieldIfNotExists(name string, typ influxql.DataType, limitCount bool) error {
	m.mu.RLock()

	// Ignore if the field already exists.
	if f := m.fields[name]; f != nil {
		if f.Type != typ {
			m.mu.RUnlock()
			return ErrFieldTypeConflict
		}
		m.mu.RUnlock()
		return nil
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()
	if f := m.fields[name]; f != nil {
		return nil
	}

	// Create and append a new field.
	f := &Field{
		ID:   uint8(len(m.fields) + 1),
		Name: name,
		Type: typ,
	}
	m.fields[name] = f

	return nil
}

// Field returns the field for name, or nil if there is no field for name.
func (m *MeasurementFields) Field(name string) *Field {
	m.mu.RLock()
	f := m.fields[name]
	m.mu.RUnlock()
	return f
}

// FieldBytes returns the field for name, or nil if there is no field for name.
// FieldBytes should be preferred to Field when the caller has a []byte, because
// it avoids a string allocation, which can't be avoided if the caller converts
// the []byte to a string and calls Field.
func (m *MeasurementFields) FieldBytes(name []byte) *Field {
	m.mu.RLock()
	f := m.fields[string(name)]
	m.mu.RUnlock()
	return f
}

// FieldSet returns the set of fields and their types for the measurement.
func (m *MeasurementFields) FieldSet() map[string]influxql.DataType {
	m.mu.RLock()
	defer m.mu.RUnlock()

	fields := make(map[string]influxql.DataType)
	for name, f := range m.fields {
		fields[name] = f.Type
	}
	return fields
}

// Field represents a series field.
type Field struct {
	ID   uint8             `json:"id,omitempty"`
	Name string            `json:"name,omitempty"`
	Type influxql.DataType `json:"type,omitempty"`
}

// shardIteratorCreator creates iterators for a local shard.
// This simply wraps the shard so that Close() does not close the underlying shard.
type shardIteratorCreator struct {
	sh         *Shard
	maxSeriesN int
}

func (ic *shardIteratorCreator) Close() error { return nil }

func (ic *shardIteratorCreator) CreateIterator(opt influxql.IteratorOptions) (influxql.Iterator, error) {
	itr, err := ic.sh.CreateIterator(opt)
	if err != nil {
		return nil, err
	} else if itr == nil {
		return nil, nil
	}

	// Enforce series limit at creation time.
	if ic.maxSeriesN > 0 {
		stats := itr.Stats()
		if stats.SeriesN > ic.maxSeriesN {
			itr.Close()
			return nil, fmt.Errorf("max-select-series limit exceeded: (%d/%d)", stats.SeriesN, ic.maxSeriesN)
		}
	}

	return itr, nil
}
func (ic *shardIteratorCreator) FieldDimensions(sources influxql.Sources) (fields map[string]influxql.DataType, dimensions map[string]struct{}, err error) {
	return ic.sh.FieldDimensions(sources)
}
func (ic *shardIteratorCreator) ExpandSources(sources influxql.Sources) (influxql.Sources, error) {
	return ic.sh.ExpandSources(sources)
}

// NewFieldKeysIterator returns an iterator that can be iterated over to
// retrieve field keys.
func NewFieldKeysIterator(sh *Shard, opt influxql.IteratorOptions) (influxql.Iterator, error) {
	itr := &fieldKeysIterator{sh: sh}

	// Retrieve measurements from shard. Filter if condition specified.
	if opt.Condition == nil {
		itr.mms = sh.index.Measurements()
	} else {
		mms, _, err := sh.index.measurementsByExpr(opt.Condition)
		if err != nil {
			return nil, err
		}
		itr.mms = mms
	}

	// Sort measurements by name.
	sort.Sort(itr.mms)

	return itr, nil
}

// fieldKeysIterator iterates over measurements and gets field keys from each measurement.
type fieldKeysIterator struct {
	sh  *Shard
	mms Measurements // remaining measurements
	buf struct {
		mm     *Measurement // current measurement
		fields []Field      // current measurement's fields
	}
}

// Stats returns stats about the points processed.
func (itr *fieldKeysIterator) Stats() influxql.IteratorStats { return influxql.IteratorStats{} }

// Close closes the iterator.
func (itr *fieldKeysIterator) Close() error { return nil }

// Next emits the next tag key name.
func (itr *fieldKeysIterator) Next() (*influxql.FloatPoint, error) {
	for {
		// If there are no more keys then move to the next measurements.
		if len(itr.buf.fields) == 0 {
			if len(itr.mms) == 0 {
				return nil, nil
			}

			itr.buf.mm = itr.mms[0]
			mf := itr.sh.engine.MeasurementFields(itr.buf.mm.Name)
			if mf != nil {
				fset := mf.FieldSet()
				if len(fset) == 0 {
					itr.mms = itr.mms[1:]
					continue
				}

				keys := make([]string, 0, len(fset))
				for k := range fset {
					keys = append(keys, k)
				}
				sort.Strings(keys)

				itr.buf.fields = make([]Field, len(keys))
				for i, name := range keys {
					itr.buf.fields[i] = Field{Name: name, Type: fset[name]}
				}
			}
			itr.mms = itr.mms[1:]
			continue
		}

		// Return next key.
		field := itr.buf.fields[0]
		p := &influxql.FloatPoint{
			Name: itr.buf.mm.Name,
			Aux:  []interface{}{field.Name, field.Type.String()},
		}
		itr.buf.fields = itr.buf.fields[1:]

		return p, nil
	}
}

// seriesIterator emits series ids.
type seriesIterator struct {
	mms  Measurements
	keys struct {
		buf []string
		i   int
	}

	point influxql.FloatPoint // reusable point
	opt   influxql.IteratorOptions
}

// NewSeriesIterator returns a new instance of SeriesIterator.
func NewSeriesIterator(sh *Shard, opt influxql.IteratorOptions) (influxql.Iterator, error) {
	// Only equality operators are allowed.
	var err error
	influxql.WalkFunc(opt.Condition, func(n influxql.Node) {
		switch n := n.(type) {
		case *influxql.BinaryExpr:
			switch n.Op {
			case influxql.EQ, influxql.NEQ, influxql.EQREGEX, influxql.NEQREGEX,
				influxql.OR, influxql.AND:
			default:
				err = errors.New("invalid tag comparison operator")
			}
		}
	})
	if err != nil {
		return nil, err
	}

	// Read and sort all measurements.
	mms := sh.index.Measurements()
	sort.Sort(mms)

	return &seriesIterator{
		mms: mms,
		point: influxql.FloatPoint{
			Aux: make([]interface{}, len(opt.Aux)),
		},
		opt: opt,
	}, nil
}

// Stats returns stats about the points processed.
func (itr *seriesIterator) Stats() influxql.IteratorStats { return influxql.IteratorStats{} }

// Close closes the iterator.
func (itr *seriesIterator) Close() error { return nil }

// Next emits the next point in the iterator.
func (itr *seriesIterator) Next() (*influxql.FloatPoint, error) {
	for {
		// Load next measurement's keys if there are no more remaining.
		if itr.keys.i >= len(itr.keys.buf) {
			if err := itr.nextKeys(); err != nil {
				return nil, err
			}
			if len(itr.keys.buf) == 0 {
				return nil, nil
			}
		}

		// Read the next key.
		key := itr.keys.buf[itr.keys.i]
		itr.keys.i++

		// Write auxiliary fields.
		for i, f := range itr.opt.Aux {
			switch f.Val {
			case "key":
				itr.point.Aux[i] = key
			}
		}
		return &itr.point, nil
	}
}

// nextKeys reads all keys for the next measurement.
func (itr *seriesIterator) nextKeys() error {
	for {
		// Ensure previous keys are cleared out.
		itr.keys.i, itr.keys.buf = 0, itr.keys.buf[:0]

		// Read next measurement.
		if len(itr.mms) == 0 {
			return nil
		}
		mm := itr.mms[0]
		itr.mms = itr.mms[1:]

		// Read all series keys.
		ids, err := mm.seriesIDsAllOrByExpr(itr.opt.Condition)
		if err != nil {
			return err
		} else if len(ids) == 0 {
			continue
		}
		itr.keys.buf = mm.AppendSeriesKeysByID(itr.keys.buf, ids)
		sort.Strings(itr.keys.buf)

		return nil
	}
}

// NewTagKeysIterator returns a new instance of TagKeysIterator.
func NewTagKeysIterator(sh *Shard, opt influxql.IteratorOptions) (influxql.Iterator, error) {
	fn := func(m *Measurement) []string {
		return m.TagKeys()
	}
	return newMeasurementKeysIterator(sh, fn, opt)
}

// tagValuesIterator emits key/tag values
type tagValuesIterator struct {
	series []*Series // remaining series
	keys   []string  // tag keys to select from a series
	fields []string  // fields to emit (key or value)
	buf    struct {
		s    *Series  // current series
		keys []string // current tag's keys
	}
}

// NewTagValuesIterator returns a new instance of TagValuesIterator.
func NewTagValuesIterator(sh *Shard, opt influxql.IteratorOptions) (influxql.Iterator, error) {
	if opt.Condition == nil {
		return nil, errors.New("a condition is required")
	}

	measurementExpr := influxql.CloneExpr(opt.Condition)
	measurementExpr = influxql.Reduce(influxql.RewriteExpr(measurementExpr, func(e influxql.Expr) influxql.Expr {
		switch e := e.(type) {
		case *influxql.BinaryExpr:
			switch e.Op {
			case influxql.EQ, influxql.NEQ, influxql.EQREGEX, influxql.NEQREGEX:
				tag, ok := e.LHS.(*influxql.VarRef)
				if !ok || tag.Val != "_name" {
					return nil
				}
			}
		}
		return e
	}), nil)

	mms, ok, err := sh.index.measurementsByExpr(measurementExpr)
	if err != nil {
		return nil, err
	} else if !ok {
		mms = sh.index.Measurements()
		sort.Sort(mms)
	}

	// If there are no measurements, return immediately.
	if len(mms) == 0 {
		return &tagValuesIterator{}, nil
	}

	filterExpr := influxql.CloneExpr(opt.Condition)
	filterExpr = influxql.Reduce(influxql.RewriteExpr(filterExpr, func(e influxql.Expr) influxql.Expr {
		switch e := e.(type) {
		case *influxql.BinaryExpr:
			switch e.Op {
			case influxql.EQ, influxql.NEQ, influxql.EQREGEX, influxql.NEQREGEX:
				tag, ok := e.LHS.(*influxql.VarRef)
				if !ok || strings.HasPrefix(tag.Val, "_") {
					return nil
				}
			}
		}
		return e
	}), nil)

	var series []*Series
	keys := newStringSet()
	for _, mm := range mms {
		ss, ok, err := mm.TagKeysByExpr(opt.Condition)
		if err != nil {
			return nil, err
		} else if !ok {
			keys.add(mm.TagKeys()...)
		} else {
			keys = keys.union(ss)
		}

		ids, err := mm.seriesIDsAllOrByExpr(filterExpr)
		if err != nil {
			return nil, err
		}

		for _, id := range ids {
			series = append(series, mm.SeriesByID(id))
		}
	}

	return &tagValuesIterator{
		series: series,
		keys:   keys.list(),
		fields: influxql.VarRefs(opt.Aux).Strings(),
	}, nil
}

// Stats returns stats about the points processed.
func (itr *tagValuesIterator) Stats() influxql.IteratorStats { return influxql.IteratorStats{} }

// Close closes the iterator.
func (itr *tagValuesIterator) Close() error { return nil }

// Next emits the next point in the iterator.
func (itr *tagValuesIterator) Next() (*influxql.FloatPoint, error) {
	for {
		// If there are no more values then move to the next key.
		if len(itr.buf.keys) == 0 {
			if len(itr.series) == 0 {
				return nil, nil
			}

			itr.buf.s = itr.series[0]
			itr.buf.keys = itr.keys
			itr.series = itr.series[1:]
			continue
		}

		key := itr.buf.keys[0]
		value := itr.buf.s.Tags.GetString(key)
		if value == "" {
			itr.buf.keys = itr.buf.keys[1:]
			continue
		}

		// Prepare auxiliary fields.
		auxFields := make([]interface{}, len(itr.fields))
		for i, f := range itr.fields {
			switch f {
			case "_tagKey":
				auxFields[i] = key
			case "value":
				auxFields[i] = value
			}
		}

		// Return next key.
		p := &influxql.FloatPoint{
			Name: itr.buf.s.measurement.Name,
			Aux:  auxFields,
		}
		itr.buf.keys = itr.buf.keys[1:]

		return p, nil
	}
}

// measurementKeyFunc is the function called by measurementKeysIterator.
type measurementKeyFunc func(m *Measurement) []string

func newMeasurementKeysIterator(sh *Shard, fn measurementKeyFunc, opt influxql.IteratorOptions) (*measurementKeysIterator, error) {
	itr := &measurementKeysIterator{fn: fn}

	// Retrieve measurements from shard. Filter if condition specified.
	if opt.Condition == nil {
		itr.mms = sh.index.Measurements()
	} else {
		mms, _, err := sh.index.measurementsByExpr(opt.Condition)
		if err != nil {
			return nil, err
		}
		itr.mms = mms
	}

	// Sort measurements by name.
	sort.Sort(itr.mms)

	return itr, nil
}

// measurementKeysIterator iterates over measurements and gets keys from each measurement.
type measurementKeysIterator struct {
	mms Measurements // remaining measurements
	buf struct {
		mm   *Measurement // current measurement
		keys []string     // current measurement's keys
	}
	fn measurementKeyFunc
}

// Stats returns stats about the points processed.
func (itr *measurementKeysIterator) Stats() influxql.IteratorStats { return influxql.IteratorStats{} }

// Close closes the iterator.
func (itr *measurementKeysIterator) Close() error { return nil }

// Next emits the next tag key name.
func (itr *measurementKeysIterator) Next() (*influxql.FloatPoint, error) {
	for {
		// If there are no more keys then move to the next measurements.
		if len(itr.buf.keys) == 0 {
			if len(itr.mms) == 0 {
				return nil, nil
			}

			itr.buf.mm = itr.mms[0]
			itr.buf.keys = itr.fn(itr.buf.mm)
			itr.mms = itr.mms[1:]
			continue
		}

		// Return next key.
		p := &influxql.FloatPoint{
			Name: itr.buf.mm.Name,
			Aux:  []interface{}{itr.buf.keys[0]},
		}
		itr.buf.keys = itr.buf.keys[1:]

		return p, nil
	}
}
