package collectd

import (
	"time"

	"github.com/influxdata/influxdb/toml"
)

const (
	// DefaultBindAddress is the default port to bind to
	DefaultBindAddress = ":25826"

	// DefaultDatabase is the default DB to write to
	DefaultDatabase = "collectd"

	// DefaultRetentionPolicy is the default retention policy of the writes
	DefaultRetentionPolicy = ""

	// DefaultBatchSize is the default write batch size.
	DefaultBatchSize = 5000

	// DefaultBatchPending is the default number of pending write batches.
	DefaultBatchPending = 10

	// DefaultBatchDuration is the default batch timeout duration.
	DefaultBatchDuration = toml.Duration(10 * time.Second)

	// DefaultTypesDB is the default location of the collectd types db file.
	DefaultTypesDB = "/usr/share/collectd/types.db"

	// DefaultReadBuffer is the default buffer size for the UDP listener.
	// Sets the size of the operating system's receive buffer associated with
	// the UDP traffic. Keep in mind that the OS must be able
	// to handle the number set here or the UDP listener will error and exit.
	//
	// DefaultReadBuffer = 0 means to use the OS default, which is usually too
	// small for high UDP performance.
	//
	// Increasing OS buffer limits:
	//     Linux:      sudo sysctl -w net.core.rmem_max=<read-buffer>
	//     BSD/Darwin: sudo sysctl -w kern.ipc.maxsockbuf=<read-buffer>
	DefaultReadBuffer = 0
)

// Config represents a configuration for the collectd service.
type Config struct {
	Enabled         bool          `toml:"enabled"`
	BindAddress     string        `toml:"bind-address"`
	Database        string        `toml:"database"`
	RetentionPolicy string        `toml:"retention-policy"`
	BatchSize       int           `toml:"batch-size"`
	BatchPending    int           `toml:"batch-pending"`
	BatchDuration   toml.Duration `toml:"batch-timeout"`
	ReadBuffer      int           `toml:"read-buffer"`
	TypesDB         string        `toml:"typesdb"`
}

// NewConfig returns a new instance of Config with defaults.
func NewConfig() Config {
	return Config{
		BindAddress:     DefaultBindAddress,
		Database:        DefaultDatabase,
		RetentionPolicy: DefaultRetentionPolicy,
		ReadBuffer:      DefaultReadBuffer,
		BatchSize:       DefaultBatchSize,
		BatchPending:    DefaultBatchPending,
		BatchDuration:   DefaultBatchDuration,
		TypesDB:         DefaultTypesDB,
	}
}

// WithDefaults takes the given config and returns a new config with any required
// default values set.
func (c *Config) WithDefaults() *Config {
	d := *c
	if d.BindAddress == "" {
		d.BindAddress = DefaultBindAddress
	}
	if d.Database == "" {
		d.Database = DefaultDatabase
	}
	if d.RetentionPolicy == "" {
		d.RetentionPolicy = DefaultRetentionPolicy
	}
	if d.BatchSize == 0 {
		d.BatchSize = DefaultBatchSize
	}
	if d.BatchPending == 0 {
		d.BatchPending = DefaultBatchPending
	}
	if d.BatchDuration == 0 {
		d.BatchDuration = DefaultBatchDuration
	}
	if d.ReadBuffer == 0 {
		d.ReadBuffer = DefaultReadBuffer
	}
	if d.TypesDB == "" {
		d.TypesDB = DefaultTypesDB
	}

	return &d
}
