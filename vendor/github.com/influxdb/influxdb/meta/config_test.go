package meta_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/meta"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c meta.Config
	if _, err := toml.Decode(`
dir = "/tmp/foo"
election-timeout = "10s"
heartbeat-timeout = "20s"
leader-lease-timeout = "30h"
commit-timeout = "40m"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Dir != "/tmp/foo" {
		t.Fatalf("unexpected dir: %s", c.Dir)
	} else if time.Duration(c.ElectionTimeout) != 10*time.Second {
		t.Fatalf("unexpected election timeout: %v", c.ElectionTimeout)
	} else if time.Duration(c.HeartbeatTimeout) != 20*time.Second {
		t.Fatalf("unexpected heartbeat timeout: %v", c.HeartbeatTimeout)
	} else if time.Duration(c.LeaderLeaseTimeout) != 30*time.Hour {
		t.Fatalf("unexpected leader lease timeout: %v", c.LeaderLeaseTimeout)
	} else if time.Duration(c.CommitTimeout) != 40*time.Minute {
		t.Fatalf("unexpected commit timeout: %v", c.CommitTimeout)
	}
}
