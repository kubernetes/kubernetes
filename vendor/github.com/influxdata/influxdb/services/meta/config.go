package meta

import (
	"errors"
	"time"
)

const (
	// DefaultLeaseDuration is the default duration for leases.
	DefaultLeaseDuration = 60 * time.Second

	// DefaultLoggingEnabled determines if log messages are printed for the meta service
	DefaultLoggingEnabled = true
)

// Config represents the meta configuration.
type Config struct {
	Dir string `toml:"dir"`

	RetentionAutoCreate bool `toml:"retention-autocreate"`
	LoggingEnabled      bool `toml:"logging-enabled"`
}

// NewConfig builds a new configuration with default values.
func NewConfig() *Config {
	return &Config{
		RetentionAutoCreate: true,
		LoggingEnabled:      DefaultLoggingEnabled,
	}
}

func (c *Config) Validate() error {
	if c.Dir == "" {
		return errors.New("Meta.Dir must be specified")
	}
	return nil
}
