package precreator_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/precreator"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c precreator.Config
	if _, err := toml.Decode(`
enabled = true
check-interval = "2m"
advance-period = "10m"
`, &c); err != nil {

		t.Fatal(err)
	}

	// Validate configuration.
	if !c.Enabled {
		t.Fatalf("unexpected enabled state: %v", c.Enabled)
	} else if time.Duration(c.CheckInterval) != 2*time.Minute {
		t.Fatalf("unexpected check interval: %s", c.CheckInterval)
	} else if time.Duration(c.AdvancePeriod) != 10*time.Minute {
		t.Fatalf("unexpected advance period: %s", c.AdvancePeriod)
	}
}
