package subscriber_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/services/subscriber"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c subscriber.Config
	if _, err := toml.Decode(`
enabled = false
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != false {
		t.Errorf("unexpected enabled state: %v", c.Enabled)
	}
	if c.InsecureSkipVerify == true {
		t.Errorf("InsecureSkipVerify: expected %v. got %v", false, c.InsecureSkipVerify)
	}
}

func TestConfig_ParseTLSConfig(t *testing.T) {
	abspath, err := filepath.Abs("/path/to/ca-certs.pem")
	if err != nil {
		t.Fatalf("Could not construct absolute path. %v", err)
	}

	// Parse configuration.
	var c subscriber.Config
	if _, err := toml.Decode(fmt.Sprintf(`
http-timeout = "60s"
enabled = true
ca-certs = '%s'
insecure-skip-verify = true
write-buffer-size = 1000
write-concurrency = 10
`, abspath), &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != true {
		t.Errorf("unexpected enabled state: %v", c.Enabled)
	}
	if c.CaCerts != abspath {
		t.Errorf("CaCerts: expected %s. got %s", abspath, c.CaCerts)
	}
	if c.InsecureSkipVerify != true {
		t.Errorf("InsecureSkipVerify: expected %v. got %v", true, c.InsecureSkipVerify)
	}
	err = c.Validate()
	if err == nil {
		t.Errorf("Expected Validation to fail (%s doesn't exist)", abspath)
	}

	if err.Error() != fmt.Sprintf("ca-certs file %s does not exist", abspath) {
		t.Errorf("Expected descriptive validation error. Instead got %v", err)
	}
}

func TestConfig_ParseTLSConfigValidCerts(t *testing.T) {
	tmpfile, err := ioutil.TempFile("", "ca-certs.crt")
	if err != nil {
		t.Fatalf("could not create temp file. error was: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte("=== BEGIN CERTIFICATE ===\n=== END CERTIFICATE ===")); err != nil {
		t.Fatalf("could not write temp file. error was: %v", err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatalf("could not close temp file. error was %v", err)
	}

	// Parse configuration.
	var c subscriber.Config
	if _, err := toml.Decode(fmt.Sprintf(`
http-timeout = "60s"
enabled = true
ca-certs = '%s'
insecure-skip-verify = false
write-buffer-size = 1000
write-concurrency = 10
`, tmpfile.Name()), &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.Enabled != true {
		t.Errorf("unexpected enabled state: %v", c.Enabled)
	}
	if c.CaCerts != tmpfile.Name() {
		t.Errorf("CaCerts: expected %v. got %v", tmpfile.Name(), c.CaCerts)
	}
	if c.InsecureSkipVerify != false {
		t.Errorf("InsecureSkipVerify: expected %v. got %v", false, c.InsecureSkipVerify)
	}
	if err := c.Validate(); err != nil {
		t.Errorf("Expected Validation to succeed. Instead was: %v", err)
	}
}
