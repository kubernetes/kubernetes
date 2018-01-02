package collectd_test

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/services/collectd"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c collectd.Config
	if _, err := toml.Decode(`
enabled = true
bind-address = ":9000"
database = "xxx"
typesdb = "yyy"
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
	} else if c.TypesDB != "yyy" {
		t.Fatalf("unexpected types db: %s", c.TypesDB)
	}
}
