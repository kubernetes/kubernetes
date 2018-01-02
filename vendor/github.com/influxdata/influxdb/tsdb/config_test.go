package tsdb_test

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/tsdb"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	c := tsdb.NewConfig()
	if _, err := toml.Decode(`
dir = "/var/lib/influxdb/data"
wal-dir = "/var/lib/influxdb/wal"
`, &c); err != nil {
		t.Fatal(err)
	}

	if err := c.Validate(); err != nil {
		t.Errorf("unexpected validate error: %s", err)
	}

	if got, exp := c.Dir, "/var/lib/influxdb/data"; got != exp {
		t.Errorf("unexpected dir:\n\nexp=%v\n\ngot=%v\n\n", exp, got)
	}
	if got, exp := c.WALDir, "/var/lib/influxdb/wal"; got != exp {
		t.Errorf("unexpected wal-dir:\n\nexp=%v\n\ngot=%v\n\n", exp, got)
	}
}

func TestConfig_Validate_Error(t *testing.T) {
	c := tsdb.NewConfig()
	if err := c.Validate(); err == nil || err.Error() != "Data.Dir must be specified" {
		t.Errorf("unexpected error: %s", err)
	}

	c.Dir = "/var/lib/influxdb/data"
	if err := c.Validate(); err == nil || err.Error() != "Data.WALDir must be specified" {
		t.Errorf("unexpected error: %s", err)
	}

	c.WALDir = "/var/lib/influxdb/wal"
	c.Engine = "fake1"
	if err := c.Validate(); err == nil || err.Error() != "unrecognized engine fake1" {
		t.Errorf("unexpected error: %s", err)
	}
}
