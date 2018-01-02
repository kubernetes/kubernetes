package continuous_querier_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/services/continuous_querier"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c continuous_querier.Config
	if _, err := toml.Decode(`
run-interval = "1m"
enabled = true
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if time.Duration(c.RunInterval) != time.Minute {
		t.Fatalf("unexpected run interval: %v", c.RunInterval)
	} else if c.Enabled != true {
		t.Fatalf("unexpected enabled: %v", c.Enabled)
	}
}
