package hh

import (
	"time"

	"github.com/influxdb/influxdb/toml"
)

const (
	// DefaultMaxSize is the default maximum size of all hinted handoff queues in bytes.
	DefaultMaxSize = 1024 * 1024 * 1024

	// DefaultMaxAge is the default maximum amount of time that a hinted handoff write
	// can stay in the queue.  After this time, the write will be purged.
	DefaultMaxAge = 7 * 24 * time.Hour

	// DefaultRetryRateLimit is the default rate that hinted handoffs will be retried.
	// The rate is in bytes per second and applies across all nodes when retried.   A
	// value of 0 disables the rate limit.
	DefaultRetryRateLimit = 0

	// DefaultRetryInterval is the default amout of time the system waits before
	// attempting to flush hinted handoff queues.
	DefaultRetryInterval = time.Second
)

type Config struct {
	Enabled        bool          `toml:"enabled"`
	Dir            string        `toml:"dir"`
	MaxSize        int64         `toml:"max-size"`
	MaxAge         toml.Duration `toml:"max-age"`
	RetryRateLimit int64         `toml:"retry-rate-limit"`
	RetryInterval  toml.Duration `toml:"retry-interval"`
}

func NewConfig() Config {
	return Config{
		Enabled:        true,
		MaxSize:        DefaultMaxSize,
		MaxAge:         toml.Duration(DefaultMaxAge),
		RetryRateLimit: DefaultRetryRateLimit,
		RetryInterval:  toml.Duration(DefaultRetryInterval),
	}
}
