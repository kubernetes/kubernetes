package coordinator_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/coordinator"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c coordinator.Config
	if _, err := toml.Decode(`
write-timeout = "20s"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if time.Duration(c.WriteTimeout) != 20*time.Second {
		t.Fatalf("unexpected write timeout s: %s", c.WriteTimeout)
	}
}
