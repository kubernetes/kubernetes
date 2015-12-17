package opentsdb

import (
	"time"

	"github.com/influxdb/influxdb/toml"
)

const (
	// DefaultBindAddress is the default address that the service binds to.
	DefaultBindAddress = ":4242"

	// DefaultDatabase is the default database used for writes.
	DefaultDatabase = "opentsdb"

	// DefaultRetentionPolicy is the default retention policy used for writes.
	DefaultRetentionPolicy = ""

	// DefaultConsistencyLevel is the default write consistency level.
	DefaultConsistencyLevel = "one"

	// DefaultBatchSize is the default Graphite batch size.
	DefaultBatchSize = 1000

	// DefaultBatchTimeout is the default Graphite batch timeout.
	DefaultBatchTimeout = time.Second

	// DefaultBatchPending is the default number of batches that can be in the queue.
	DefaultBatchPending = 5
)

// Config represents the configuration of the OpenTSDB service.
type Config struct {
	Enabled          bool          `toml:"enabled"`
	BindAddress      string        `toml:"bind-address"`
	Database         string        `toml:"database"`
	RetentionPolicy  string        `toml:"retention-policy"`
	ConsistencyLevel string        `toml:"consistency-level"`
	TLSEnabled       bool          `toml:"tls-enabled"`
	Certificate      string        `toml:"certificate"`
	BatchSize        int           `toml:"batch-size"`
	BatchPending     int           `toml:"batch-pending"`
	BatchTimeout     toml.Duration `toml:"batch-timeout"`
	LogPointErrors   bool          `toml:"log-point-errors"`
}

// NewConfig returns a new config for the service.
func NewConfig() Config {
	return Config{
		BindAddress:      DefaultBindAddress,
		Database:         DefaultDatabase,
		RetentionPolicy:  DefaultRetentionPolicy,
		ConsistencyLevel: DefaultConsistencyLevel,
		TLSEnabled:       false,
		Certificate:      "/etc/ssl/influxdb.pem",
		BatchSize:        DefaultBatchSize,
		BatchPending:     DefaultBatchPending,
		BatchTimeout:     toml.Duration(DefaultBatchTimeout),
		LogPointErrors:   true,
	}
}
