package subscriber

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/influxdata/influxdb/toml"
)

const (
	DefaultHTTPTimeout      = 30 * time.Second
	DefaultWriteConcurrency = 40
	DefaultWriteBufferSize  = 1000
)

// Config represents a configuration of the subscriber service.
type Config struct {
	// Whether to enable to Subscriber service
	Enabled bool `toml:"enabled"`

	HTTPTimeout toml.Duration `toml:"http-timeout"`

	// InsecureSkipVerify gets passed to the http client, if true, it will
	// skip https certificate verification. Defaults to false
	InsecureSkipVerify bool `toml:"insecure-skip-verify"`

	// configure the path to the PEM encoded CA certs file. If the
	// empty string, the default system certs will be used
	CaCerts string `toml:"ca-certs"`

	// The number of writer goroutines processing the write channel.
	WriteConcurrency int `toml:"write-concurrency"`

	// The number of in-flight writes buffered in the write channel.
	WriteBufferSize int `toml:"write-buffer-size"`
}

// NewConfig returns a new instance of a subscriber config.
func NewConfig() Config {
	return Config{
		Enabled:            true,
		HTTPTimeout:        toml.Duration(DefaultHTTPTimeout),
		InsecureSkipVerify: false,
		CaCerts:            "",
		WriteConcurrency:   DefaultWriteConcurrency,
		WriteBufferSize:    DefaultWriteBufferSize,
	}
}

func (c Config) Validate() error {
	if c.HTTPTimeout <= 0 {
		return errors.New("http-timeout must be greater than 0")
	}

	if c.CaCerts != "" && !fileExists(c.CaCerts) {
		abspath, err := filepath.Abs(c.CaCerts)
		if err != nil {
			return fmt.Errorf("ca-certs file %s does not exist. Wrapped Error: %v", c.CaCerts, err)
		}
		return fmt.Errorf("ca-certs file %s does not exist", abspath)
	}

	if c.WriteBufferSize <= 0 {
		return errors.New("write-buffer-size must be greater than 0")
	}

	if c.WriteConcurrency <= 0 {
		return errors.New("write-concurrency must be greater than 0")
	}

	return nil
}

func fileExists(fileName string) bool {
	info, err := os.Stat(fileName)
	return err == nil && !info.IsDir()
}
