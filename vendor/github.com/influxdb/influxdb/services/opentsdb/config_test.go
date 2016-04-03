package opentsdb_test

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/opentsdb"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c opentsdb.Config
	if _, err := toml.Decode(`
enabled = true
bind-address = ":9000"
database = "xxx"
consistency-level ="all"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != true {
		t.Fatalf("unexpected enabled: %v", c.Enabled)
	} else if c.BindAddress != ":9000" {
		t.Fatalf("unexpected bind address: %s", c.BindAddress)
	} else if c.Database != "xxx" {
		t.Fatalf("unexpected database: %s", c.Database)
	} else if c.ConsistencyLevel != "all" {
		t.Fatalf("unexpected database: %s", c.ConsistencyLevel)
	}
}
