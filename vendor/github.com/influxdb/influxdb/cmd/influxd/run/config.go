package run

import (
	"errors"
	"fmt"
	"os/user"
	"path/filepath"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/admin"
	"github.com/influxdb/influxdb/services/collectd"
	"github.com/influxdb/influxdb/services/continuous_querier"
	"github.com/influxdb/influxdb/services/graphite"
	"github.com/influxdb/influxdb/services/hh"
	"github.com/influxdb/influxdb/services/httpd"
	"github.com/influxdb/influxdb/services/monitor"
	"github.com/influxdb/influxdb/services/opentsdb"
	"github.com/influxdb/influxdb/services/precreator"
	"github.com/influxdb/influxdb/services/retention"
	"github.com/influxdb/influxdb/services/udp"
	"github.com/influxdb/influxdb/tsdb"
)

// Config represents the configuration format for the influxd binary.
type Config struct {
	Meta       meta.Config       `toml:"meta"`
	Data       tsdb.Config       `toml:"data"`
	Cluster    cluster.Config    `toml:"cluster"`
	Retention  retention.Config  `toml:"retention"`
	Precreator precreator.Config `toml:"shard-precreation"`

	Admin     admin.Config      `toml:"admin"`
	HTTPD     httpd.Config      `toml:"http"`
	Graphites []graphite.Config `toml:"graphite"`
	Collectd  collectd.Config   `toml:"collectd"`
	OpenTSDB  opentsdb.Config   `toml:"opentsdb"`
	UDP       udp.Config        `toml:"udp"`

	// Snapshot SnapshotConfig `toml:"snapshot"`
	Monitoring      monitor.Config            `toml:"monitoring"`
	ContinuousQuery continuous_querier.Config `toml:"continuous_queries"`

	HintedHandoff hh.Config `toml:"hinted-handoff"`

	// Server reporting
	ReportingDisabled bool `toml:"reporting-disabled"`
}

// NewConfig returns an instance of Config with reasonable defaults.
func NewConfig() *Config {
	c := &Config{}
	c.Meta = meta.NewConfig()
	c.Data = tsdb.NewConfig()
	c.Cluster = cluster.NewConfig()
	c.Precreator = precreator.NewConfig()

	c.Admin = admin.NewConfig()
	c.HTTPD = httpd.NewConfig()
	c.Collectd = collectd.NewConfig()
	c.OpenTSDB = opentsdb.NewConfig()
	c.Graphites = append(c.Graphites, graphite.NewConfig())

	c.Monitoring = monitor.NewConfig()
	c.ContinuousQuery = continuous_querier.NewConfig()
	c.Retention = retention.NewConfig()
	c.HintedHandoff = hh.NewConfig()

	return c
}

// NewDemoConfig returns the config that runs when no config is specified.
func NewDemoConfig() (*Config, error) {
	c := NewConfig()

	// By default, store meta and data files in current users home directory
	u, err := user.Current()
	if err != nil {
		return nil, fmt.Errorf("failed to determine current user for storage")
	}

	c.Meta.Dir = filepath.Join(u.HomeDir, ".influxdb/meta")
	c.Data.Dir = filepath.Join(u.HomeDir, ".influxdb/data")
	c.HintedHandoff.Dir = filepath.Join(u.HomeDir, ".influxdb/hh")

	c.Admin.Enabled = true
	c.Monitoring.Enabled = false

	return c, nil
}

// Validate returns an error if the config is invalid.
func (c *Config) Validate() error {
	if c.Meta.Dir == "" {
		return errors.New("Meta.Dir must be specified")
	} else if c.Data.Dir == "" {
		return errors.New("Data.Dir must be specified")
	} else if c.HintedHandoff.Dir == "" {
		return errors.New("HintedHandoff.Dir must be specified")
	}

	for _, g := range c.Graphites {
		if err := g.Validate(); err != nil {
			return fmt.Errorf("invalid graphite config: %v", err)
		}
	}
	return nil
}
