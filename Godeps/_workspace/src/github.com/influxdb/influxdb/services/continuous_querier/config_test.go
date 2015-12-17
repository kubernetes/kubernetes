package continuous_querier_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/continuous_querier"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c continuous_querier.Config
	if _, err := toml.Decode(`
recompute-previous-n = 1
recompute-no-older-than = "10s"
compute-runs-per-interval = 2
compute-no-more-than = "20s"
enabled = true
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.RecomputePreviousN != 1 {
		t.Fatalf("unexpected recompute previous n: %d", c.RecomputePreviousN)
	} else if time.Duration(c.RecomputeNoOlderThan) != 10*time.Second {
		t.Fatalf("unexpected recompute no older than: %v", c.RecomputeNoOlderThan)
	} else if c.ComputeRunsPerInterval != 2 {
		t.Fatalf("unexpected compute runs per interval: %d", c.ComputeRunsPerInterval)
	} else if time.Duration(c.ComputeNoMoreThan) != 20*time.Second {
		t.Fatalf("unexpected compute no more than: %v", c.ComputeNoMoreThan)
	} else if c.Enabled != true {
		t.Fatalf("unexpected enabled: %v", c.Enabled)
	}
}
