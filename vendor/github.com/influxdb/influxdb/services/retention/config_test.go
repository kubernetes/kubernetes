package retention_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/retention"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c retention.Config
	if _, err := toml.Decode(`
enabled = true
check-interval = "1s"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != true {
		t.Fatalf("unexpected enabled state: %v", c.Enabled)
	} else if time.Duration(c.CheckInterval) != time.Second {
		t.Fatalf("unexpected check interval: %v", c.CheckInterval)
	}
}
