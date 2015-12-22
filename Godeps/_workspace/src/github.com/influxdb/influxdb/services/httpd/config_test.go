package httpd_test

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/httpd"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c httpd.Config
	if _, err := toml.Decode(`
enabled = true
bind-address = ":8080"
auth-enabled = true
log-enabled = true
write-tracing = true
pprof-enabled = true
https-enabled = true
https-certificate = "/dev/null"
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != true {
		t.Fatalf("unexpected enabled: %v", c.Enabled)
	} else if c.BindAddress != ":8080" {
		t.Fatalf("unexpected bind address: %s", c.BindAddress)
	} else if c.AuthEnabled != true {
		t.Fatalf("unexpected auth enabled: %v", c.AuthEnabled)
	} else if c.LogEnabled != true {
		t.Fatalf("unexpected log enabled: %v", c.LogEnabled)
	} else if c.WriteTracing != true {
		t.Fatalf("unexpected write tracing: %v", c.WriteTracing)
	} else if c.PprofEnabled != true {
		t.Fatalf("unexpected pprof enabled: %v", c.PprofEnabled)
	} else if c.HTTPSEnabled != true {
		t.Fatalf("unexpected https enabled: %v", c.HTTPSEnabled)
	} else if c.HTTPSCertificate != "/dev/null" {
		t.Fatalf("unexpected https certificate: %v", c.HTTPSCertificate)
	}
}

func TestConfig_WriteTracing(t *testing.T) {
	c := httpd.Config{WriteTracing: true}
	s := httpd.NewService(c)
	if !s.Handler.WriteTrace {
		t.Fatalf("write tracing was not set")
	}
}
