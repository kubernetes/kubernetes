package tsdb

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
)

var (
	// ErrFormatNotFound is returned when no format can be determined from a path.
	ErrFormatNotFound = errors.New("format not found")

	// ErrUnknownEngineFormat is returned when the engine format is
	// unknown. ErrUnknownEngineFormat is currently returned if a format
	// other than tsm1 is encountered.
	ErrUnknownEngineFormat = errors.New("unknown engine format")
)

// Engine represents a swappable storage engine for the shard.
type Engine interface {
	Open() error
	Close() error

	SetLogOutput(io.Writer)
	LoadMetadataIndex(shardID uint64, index *DatabaseIndex) error

	Backup(w io.Writer, basePath string, since time.Time) error
	Restore(r io.Reader, basePath string) error

	CreateIterator(opt influxql.IteratorOptions) (influxql.Iterator, error)
	WritePoints(points []models.Point) error
	ContainsSeries(keys []string) (map[string]bool, error)
	DeleteSeries(keys []string) error
	DeleteSeriesRange(keys []string, min, max int64) error
	DeleteMeasurement(name string, seriesKeys []string) error
	SeriesCount() (n int, err error)
	MeasurementFields(measurement string) *MeasurementFields
	CreateSnapshot() (string, error)
	SetEnabled(enabled bool)

	// Format will return the format for the engine
	Format() EngineFormat

	// Statistics will return statistics relevant to this engine.
	Statistics(tags map[string]string) []models.Statistic

	io.WriterTo
}

// EngineFormat represents the format for an engine.
type EngineFormat int

const (
	// TSM1Format is the format used by the tsm1 engine.
	TSM1Format EngineFormat = 2
)

// NewEngineFunc creates a new engine.
type NewEngineFunc func(id uint64, path string, walPath string, options EngineOptions) Engine

// newEngineFuncs is a lookup of engine constructors by name.
var newEngineFuncs = make(map[string]NewEngineFunc)

// RegisterEngine registers a storage engine initializer by name.
func RegisterEngine(name string, fn NewEngineFunc) {
	if _, ok := newEngineFuncs[name]; ok {
		panic("engine already registered: " + name)
	}
	newEngineFuncs[name] = fn
}

// RegisteredEngines returns the slice of currently registered engines.
func RegisteredEngines() []string {
	a := make([]string, 0, len(newEngineFuncs))
	for k := range newEngineFuncs {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

// NewEngine returns an instance of an engine based on its format.
// If the path does not exist then the DefaultFormat is used.
func NewEngine(id uint64, path string, walPath string, options EngineOptions) (Engine, error) {
	// Create a new engine
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return newEngineFuncs[options.EngineVersion](id, path, walPath, options), nil
	}

	// If it's a dir then it's a tsm1 engine
	format := DefaultEngine
	if fi, err := os.Stat(path); err != nil {
		return nil, err
	} else if !fi.Mode().IsDir() {
		return nil, ErrUnknownEngineFormat
	} else {
		format = "tsm1"
	}

	// Lookup engine by format.
	fn := newEngineFuncs[format]
	if fn == nil {
		return nil, fmt.Errorf("invalid engine format: %q", format)
	}

	return fn(id, path, walPath, options), nil
}

// EngineOptions represents the options used to initialize the engine.
type EngineOptions struct {
	EngineVersion string
	ShardID       uint64

	Config Config
}

// NewEngineOptions returns the default options.
func NewEngineOptions() EngineOptions {
	return EngineOptions{
		EngineVersion: DefaultEngine,
		Config:        NewConfig(),
	}
}
