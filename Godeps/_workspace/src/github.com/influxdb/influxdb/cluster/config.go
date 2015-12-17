package cluster

import (
	"time"

	"github.com/influxdb/influxdb/toml"
)

const (
	// DefaultWriteTimeout is the default timeout for a complete write to succeed.
	DefaultWriteTimeout = 5 * time.Second

	// DefaultShardWriterTimeout is the default timeout set on shard writers.
	DefaultShardWriterTimeout = 5 * time.Second

	// DefaultShardMapperTimeout is the default timeout set on shard mappers.
	DefaultShardMapperTimeout = 5 * time.Second
)

// Config represents the configuration for the clustering service.
type Config struct {
	ForceRemoteShardMapping bool          `toml:"force-remote-mapping"`
	WriteTimeout            toml.Duration `toml:"write-timeout"`
	ShardWriterTimeout      toml.Duration `toml:"shard-writer-timeout"`
	ShardMapperTimeout      toml.Duration `toml:"shard-mapper-timeout"`
}

// NewConfig returns an instance of Config with defaults.
func NewConfig() Config {
	return Config{
		WriteTimeout:       toml.Duration(DefaultWriteTimeout),
		ShardWriterTimeout: toml.Duration(DefaultShardWriterTimeout),
		ShardMapperTimeout: toml.Duration(DefaultShardMapperTimeout),
	}
}
