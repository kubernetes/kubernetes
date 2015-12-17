package toml_test

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/cmd/influxd/run"
	itoml "github.com/influxdb/influxdb/toml"
)

// Ensure that megabyte sizes can be parsed.
func TestSize_UnmarshalText_MB(t *testing.T) {
	var s itoml.Size
	if err := s.UnmarshalText([]byte("200m")); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if s != 200*(1<<20) {
		t.Fatalf("unexpected size: %d", s)
	}
}

// Ensure that gigabyte sizes can be parsed.
func TestSize_UnmarshalText_GB(t *testing.T) {
	var s itoml.Size
	if err := s.UnmarshalText([]byte("1g")); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if s != 1073741824 {
		t.Fatalf("unexpected size: %d", s)
	}
}

func TestConfig_Encode(t *testing.T) {
	var c run.Config
	c.Cluster.WriteTimeout = itoml.Duration(time.Minute)
	buf := new(bytes.Buffer)
	if err := toml.NewEncoder(buf).Encode(&c); err != nil {
		t.Fatal("Failed to encode: ", err)
	}
	got, search := buf.String(), `write-timeout = "1m0s"`
	if !strings.Contains(got, search) {
		t.Fatalf("Encoding config failed.\nfailed to find %s in:\n%s\n", search, got)
	}
}
