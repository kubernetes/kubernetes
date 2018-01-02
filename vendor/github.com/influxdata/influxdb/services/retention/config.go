package retention

import (
	"time"

	"github.com/influxdata/influxdb/toml"
)

// Config represents the configuration for the retention service.
type Config struct {
	Enabled       bool          `toml:"enabled"`
	CheckInterval toml.Duration `toml:"check-interval"`
}

// NewConfig returns an instance of Config with defaults.
func NewConfig() Config {
	return Config{Enabled: true, CheckInterval: toml.Duration(30 * time.Minute)}
}
