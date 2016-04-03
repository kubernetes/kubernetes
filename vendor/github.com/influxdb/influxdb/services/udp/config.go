package udp

import "github.com/influxdb/influxdb/toml"

type Config struct {
	Enabled     bool   `toml:"enabled"`
	BindAddress string `toml:"bind-address"`

	Database     string        `toml:"database"`
	BatchSize    int           `toml:"batch-size"`
	BatchTimeout toml.Duration `toml:"batch-timeout"`
}
