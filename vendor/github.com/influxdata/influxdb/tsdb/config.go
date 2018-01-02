package tsdb

import (
	"errors"
	"fmt"
	"time"

	"github.com/influxdata/influxdb/toml"
)

const (
	// DefaultEngine is the default engine for new shards
	DefaultEngine = "tsm1"

	// tsdb/engine/wal configuration options

	// Default settings for TSM

	// DefaultCacheMaxMemorySize is the maximum size a shard's cache can
	// reach before it starts rejecting writes.
	DefaultCacheMaxMemorySize = 1024 * 1024 * 1024 // 1GB

	// DefaultCacheSnapshotMemorySize is the size at which the engine will
	// snapshot the cache and write it to a TSM file, freeing up memory
	DefaultCacheSnapshotMemorySize = 25 * 1024 * 1024 // 25MB

	// DefaultCacheSnapshotWriteColdDuration is the length of time at which
	// the engine will snapshot the cache and write it to a new TSM file if
	// the shard hasn't received writes or deletes
	DefaultCacheSnapshotWriteColdDuration = time.Duration(10 * time.Minute)

	// DefaultCompactFullWriteColdDuration is the duration at which the engine
	// will compact all TSM files in a shard if it hasn't received a write or delete
	DefaultCompactFullWriteColdDuration = time.Duration(4 * time.Hour)

	// DefaultMaxPointsPerBlock is the maximum number of points in an encoded
	// block in a TSM file
	DefaultMaxPointsPerBlock = 1000

	// DefaultMaxSeriesPerDatabase is the maximum number of series a node can hold per database.
	DefaultMaxSeriesPerDatabase = 1000000

	// DefaultMaxValuesPerTag is the maximum number of values a tag can have within a measurement.
	DefaultMaxValuesPerTag = 100000
)

// Config holds the configuration for the tsbd package.
type Config struct {
	Dir    string `toml:"dir"`
	Engine string `toml:"-"`

	// General WAL configuration options
	WALDir string `toml:"wal-dir"`

	// Query logging
	QueryLogEnabled bool `toml:"query-log-enabled"`

	// Compaction options for tsm1 (descriptions above with defaults)
	CacheMaxMemorySize             uint64        `toml:"cache-max-memory-size"`
	CacheSnapshotMemorySize        uint64        `toml:"cache-snapshot-memory-size"`
	CacheSnapshotWriteColdDuration toml.Duration `toml:"cache-snapshot-write-cold-duration"`
	CompactFullWriteColdDuration   toml.Duration `toml:"compact-full-write-cold-duration"`

	// Limits

	// MaxSeriesPerDatabase is the maximum number of series a node can hold per database.
	// When this limit is exceeded, writes return a 'max series per database exceeded' error.
	// A value of 0 disables the limit.
	MaxSeriesPerDatabase int `toml:"max-series-per-database"`

	// MaxValuesPerTag is the maximum number of tag values a single tag key can have within
	// a measurement.  When the limit is execeeded, writes return an error.
	// A value of 0 disables the limit.
	MaxValuesPerTag int `toml:"max-values-per-tag"`

	TraceLoggingEnabled bool `toml:"trace-logging-enabled"`
}

// NewConfig returns the default configuration for tsdb.
func NewConfig() Config {
	return Config{
		Engine: DefaultEngine,

		QueryLogEnabled: true,

		CacheMaxMemorySize:             DefaultCacheMaxMemorySize,
		CacheSnapshotMemorySize:        DefaultCacheSnapshotMemorySize,
		CacheSnapshotWriteColdDuration: toml.Duration(DefaultCacheSnapshotWriteColdDuration),
		CompactFullWriteColdDuration:   toml.Duration(DefaultCompactFullWriteColdDuration),

		MaxSeriesPerDatabase: DefaultMaxSeriesPerDatabase,
		MaxValuesPerTag:      DefaultMaxValuesPerTag,

		TraceLoggingEnabled: false,
	}
}

// Validate validates the configuration hold by c.
func (c *Config) Validate() error {
	if c.Dir == "" {
		return errors.New("Data.Dir must be specified")
	} else if c.WALDir == "" {
		return errors.New("Data.WALDir must be specified")
	}

	valid := false
	for _, e := range RegisteredEngines() {
		if e == c.Engine {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("unrecognized engine %s", c.Engine)
	}

	return nil
}
