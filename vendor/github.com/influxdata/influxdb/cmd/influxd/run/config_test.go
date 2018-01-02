package run_test

import (
	"os"
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/cmd/influxd/run"
)

// Ensure the configuration can be parsed.
func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c run.Config
	if err := c.FromToml(`
[meta]
dir = "/tmp/meta"

[data]
dir = "/tmp/data"

[coordinator]

[admin]
bind-address = ":8083"

[http]
bind-address = ":8087"

[[graphite]]
protocol = "udp"

[[graphite]]
protocol = "tcp"

[[collectd]]
bind-address = ":1000"

[[collectd]]
bind-address = ":1010"

[[opentsdb]]
bind-address = ":2000"

[[opentsdb]]
bind-address = ":2010"

[[opentsdb]]
bind-address = ":2020"

[[udp]]
bind-address = ":4444"

[monitoring]
enabled = true

[subscriber]
enabled = true

[continuous_queries]
enabled = true
`); err != nil {
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
	} else if len(c.GraphiteInputs) != 2 {
		t.Fatalf("unexpected graphiteInputs count: %d", len(c.GraphiteInputs))
	} else if c.GraphiteInputs[0].Protocol != "udp" {
		t.Fatalf("unexpected graphite protocol(0): %s", c.GraphiteInputs[0].Protocol)
	} else if c.GraphiteInputs[1].Protocol != "tcp" {
		t.Fatalf("unexpected graphite protocol(1): %s", c.GraphiteInputs[1].Protocol)
	} else if c.CollectdInputs[0].BindAddress != ":1000" {
		t.Fatalf("unexpected collectd bind address: %s", c.CollectdInputs[0].BindAddress)
	} else if c.CollectdInputs[1].BindAddress != ":1010" {
		t.Fatalf("unexpected collectd bind address: %s", c.CollectdInputs[1].BindAddress)
	} else if c.OpenTSDBInputs[0].BindAddress != ":2000" {
		t.Fatalf("unexpected opentsdb bind address: %s", c.OpenTSDBInputs[0].BindAddress)
	} else if c.OpenTSDBInputs[1].BindAddress != ":2010" {
		t.Fatalf("unexpected opentsdb bind address: %s", c.OpenTSDBInputs[1].BindAddress)
	} else if c.OpenTSDBInputs[2].BindAddress != ":2020" {
		t.Fatalf("unexpected opentsdb bind address: %s", c.OpenTSDBInputs[2].BindAddress)
	} else if c.UDPInputs[0].BindAddress != ":4444" {
		t.Fatalf("unexpected udp bind address: %s", c.UDPInputs[0].BindAddress)
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

[coordinator]

[admin]
bind-address = ":8083"

[http]
bind-address = ":8087"

[[graphite]]
protocol = "udp"

[[graphite]]
protocol = "tcp"

[[collectd]]
bind-address = ":1000"

[[collectd]]
bind-address = ":1010"

[[opentsdb]]
bind-address = ":2000"

[[opentsdb]]
bind-address = ":2010"

[[udp]]
bind-address = ":4444"

[[udp]]

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

	if err := os.Setenv("INFLUXDB_UDP_0_BIND_ADDRESS", ":5555"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := os.Setenv("INFLUXDB_GRAPHITE_1_PROTOCOL", "udp"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := os.Setenv("INFLUXDB_COLLECTD_1_BIND_ADDRESS", ":1020"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := os.Setenv("INFLUXDB_OPENTSDB_0_BIND_ADDRESS", ":2020"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	// uint64 type
	if err := os.Setenv("INFLUXDB_DATA_CACHE_MAX_MEMORY_SIZE", "1000"); err != nil {
		t.Fatalf("failed to set env var: %v", err)
	}

	if err := c.ApplyEnvOverrides(); err != nil {
		t.Fatalf("failed to apply env overrides: %v", err)
	}

	if c.UDPInputs[0].BindAddress != ":5555" {
		t.Fatalf("unexpected udp bind address: %s", c.UDPInputs[0].BindAddress)
	}

	if c.UDPInputs[1].BindAddress != ":1234" {
		t.Fatalf("unexpected udp bind address: %s", c.UDPInputs[1].BindAddress)
	}

	if c.GraphiteInputs[1].Protocol != "udp" {
		t.Fatalf("unexpected graphite protocol: %s", c.GraphiteInputs[1].Protocol)
	}

	if c.CollectdInputs[1].BindAddress != ":1020" {
		t.Fatalf("unexpected collectd bind address: %s", c.CollectdInputs[1].BindAddress)
	}

	if c.OpenTSDBInputs[0].BindAddress != ":2020" {
		t.Fatalf("unexpected opentsdb bind address: %s", c.OpenTSDBInputs[0].BindAddress)
	}

	if c.Data.CacheMaxMemorySize != 1000 {
		t.Fatalf("unexpected cache max memory size: %v", c.Data.CacheMaxMemorySize)
	}
}

func TestConfig_ValidateNoServiceConfigured(t *testing.T) {
	var c run.Config
	if _, err := toml.Decode(`
[meta]
enabled = false

[data]
enabled = false
`, &c); err != nil {
		t.Fatal(err)
	}

	if e := c.Validate(); e == nil {
		t.Fatalf("got nil, expected error")
	}
}

func TestConfig_ValidateMonitorStore_MetaOnly(t *testing.T) {
	c := run.NewConfig()
	if _, err := toml.Decode(`
[monitor]
store-enabled = true

[meta]
dir = "foo"

[data]
enabled = false
`, &c); err != nil {
		t.Fatal(err)
	}

	if err := c.Validate(); err == nil {
		t.Fatalf("got nil, expected error")
	}
}

func TestConfig_DeprecatedOptions(t *testing.T) {
	// Parse configuration.
	var c run.Config
	if err := c.FromToml(`
[cluster]
max-select-point = 100
`); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Coordinator.MaxSelectPointN != 100 {
		t.Fatalf("unexpected coordinator max select points: %d", c.Coordinator.MaxSelectPointN)

	}
}
