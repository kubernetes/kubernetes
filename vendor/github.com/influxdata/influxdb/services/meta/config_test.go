package meta_test

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/services/meta"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c meta.Config
	if _, err := toml.Decode(`
dir = "/tmp/foo"
logging-enabled = false
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Dir != "/tmp/foo" {
		t.Fatalf("unexpected dir: %s", c.Dir)
	} else if c.LoggingEnabled {
		t.Fatalf("unexpected logging enabled: %v", c.LoggingEnabled)
	}
}
