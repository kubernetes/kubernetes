package hh_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/hh"
)

func TestConfigParse(t *testing.T) {
	// Parse configuration.
	var c hh.Config
	if _, err := toml.Decode(`
enabled = false
retry-interval = "10m"
retry-max-interval = "100m"
max-size=2048
max-age="20m"
retry-rate-limit=1000
purge-interval = "1h"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if exp := true; c.Enabled == true {
		t.Fatalf("unexpected enabled: got %v, exp %v", c.Enabled, exp)
	}

	if exp := 10 * time.Minute; c.RetryInterval.String() != exp.String() {
		t.Fatalf("unexpected retry interval: got %v, exp %v", c.RetryInterval, exp)
	}

	if exp := 100 * time.Minute; c.RetryMaxInterval.String() != exp.String() {
		t.Fatalf("unexpected retry max interval: got %v, exp %v", c.RetryMaxInterval, exp)
	}

	if exp := 20 * time.Minute; c.MaxAge.String() != exp.String() {
		t.Fatalf("unexpected max age: got %v, exp %v", c.MaxAge, exp)
	}

	if exp := int64(2048); c.MaxSize != exp {
		t.Fatalf("unexpected retry interval: got %v, exp %v", c.MaxSize, exp)
	}

	if exp := int64(1000); c.RetryRateLimit != exp {
		t.Fatalf("unexpected retry rate limit: got %v, exp %v", c.RetryRateLimit, exp)
	}

	if exp := time.Hour; c.PurgeInterval.String() != exp.String() {
		t.Fatalf("unexpected purge interval: got %v, exp %v", c.PurgeInterval, exp)
	}

}

func TestDefaultDisabled(t *testing.T) {
	// Parse empty configuration.
	var c hh.Config
	if _, err := toml.Decode(``, &c); err != nil {
		t.Fatal(err)
	}

	if exp := false; c.Enabled == true {
		t.Fatalf("unexpected default Enabled value: got %v, exp %v", c.Enabled, exp)
	}

	// Default configuration.
	c = hh.NewConfig()
	if exp := false; c.Enabled == true {
		t.Fatalf("unexpected default enabled value: got %v, exp %v", c.Enabled, exp)
	}
}
