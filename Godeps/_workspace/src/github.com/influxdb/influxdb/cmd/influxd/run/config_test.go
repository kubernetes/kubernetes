package run_test

import (
	"os"
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/cmd/influxd/run"
)

// Ensure the configuration can be parsed.
func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c run.Config
	if _, err := toml.Decode(`
[meta]
dir = "/tmp/meta"

[data]
dir = "/tmp/data"

[cluster]

[admin]
bind-address = ":8083"

[http]
bind-address = ":8087"

[[graphite]]
protocol = "udp"

[[graphite]]
protocol = "tcp"

[collectd]
bind-address = ":1000"

[opentsdb]
bind-address = ":2000"

[[udp]]
bind-address = ":4444"

[monitoring]
enabled = true

[subscriber]
enabled = true

[continuous_queries]
enabled = true
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Meta.Dir != "/tmp/meta" {
		t.Fatalf("unexpected meta dir: %s", c.Meta.Dir)
	} else if c.Data.Dir != "/tmp/data" {
		t.Fatalf("unexpected data dir: %s", c.Data.Dir)
	} else if c.Admin.BindAddress != ":8083" {
		t.Fatalf("unexpected admin bind address: %s", c.Admin.BindAddress)
	} else if c.HTTPD.BindAddress != ":8087" {
		t.Fatalf("unexpected api bind address: %s", c.HTTPD.BindAddress)
	} else if len(c.Graphites) != 2 {
		t.Fatalf("unexpected graphites count: %d", len(c.Graphites))
	} else if c.Graphites[0].Protocol != "udp" {
		t.Fatalf("unexpected graphite protocol(0): %s", c.Graphites[0].Protocol)
	} else if c.Graphites[1].Protocol != "tcp" {
		t.Fatalf("unexpected graphite protocol(1): %s", c.Graphites[1].Protocol)
	} else if c.Collectd.BindAddress != ":1000" {
		t.Fatalf("unexpected collectd bind address: %s", c.Collectd.BindAddress)
	} else if c.OpenTSDB.BindAddress != ":2000" {
		t.Fatalf("unexpected opentsdb bind address: %s", c.OpenTSDB.BindAddress)
	} else if c.UDPs[0].BindAddress != ":4444" {
		t.Fatalf("unexpected udp bind address: %s", c.UDPs[0].BindAddress)
	} else if c.Subscriber.Enabled != true {
		t.Fatalf("unexpected subscriber enabled: %v", c.Subscriber.Enabled)
	} else if c.ContinuousQuery.Enabled != true {
		t.Fatalf("unexpected continuous query enabled: %v", c.ContinuousQuery.Enabled)
	}
}

// Ensure the configuration can be parsed.
func TestConfig_Parse_EnvOverride(t *testing.T) {
	// Parse configuration.
	var c run.Config
	if _, err := toml.Decode(`
[meta]
dir = "/tmp/meta"

[data]
dir = "/tmp/data"

[cluster]

[admin]
bind-address = ":8083"

[http]
bind-address = ":8087"

[[graphite]]
protocol = "udp"

[[graphite]]
protocol = "tcp"

[collectd]
bind-address = ":1000"

[opentsdb]
bind-address = ":2000"

[[udp]]
bind-address = ":4444"

[monitoring]
enabled = true

[continuous_queries]
enabled = true
`, &c); err != nil {
		t.Fatal(err)
	}

	if err := os.Setenv("INFLUXDB_UDP_BIND_ADDRESS", ":1234"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := os.Setenv("INFLUXDB_GRAPHITE_1_PROTOCOL", "udp"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := c.ApplyEnvOverrides(); err != nil {
		t.Fatalf("failed to apply env overrides: %v", err)
	}

	if c.UDPs[0].BindAddress != ":4444" {
		t.Fatalf("unexpected udp bind address: %s", c.UDPs[0].BindAddress)
	}

	if c.Graphites[1].Protocol != "udp" {
		t.Fatalf("unexpected graphite protocol(0): %s", c.Graphites[0].Protocol)
	}
}
