// +build experimental

package daemon

import (
	"testing"

	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
)

type fakeDriver struct{}

func (fakeDriver) Name() string                              { return "fake" }
func (fakeDriver) Create(name string) (volume.Volume, error) { return nil, nil }
func (fakeDriver) Remove(v volume.Volume) error              { return nil }

func TestGetVolumeDriver(t *testing.T) {
	_, err := getVolumeDriver("missing")
	if err == nil {
		t.Fatal("Expected error, was nil")
	}

	volumedrivers.Register(fakeDriver{}, "fake")
	d, err := getVolumeDriver("fake")
	if err != nil {
		t.Fatal(err)
	}
	if d.Name() != "fake" {
		t.Fatalf("Expected fake driver, got %s\n", d.Name())
	}
}

func TestParseBindMount(t *testing.T) {
	cases := []struct {
		bind       string
		driver     string
		expDest    string
		expSource  string
		expName    string
		expDriver  string
		mountLabel string
		expRW      bool
		fail       bool
	}{
		{"/tmp:/tmp", "", "/tmp", "/tmp", "", "", "", true, false},
		{"/tmp:/tmp:ro", "", "/tmp", "/tmp", "", "", "", false, false},
		{"/tmp:/tmp:rw", "", "/tmp", "/tmp", "", "", "", true, false},
		{"/tmp:/tmp:foo", "", "/tmp", "/tmp", "", "", "", false, true},
		{"name:/tmp", "", "/tmp", "", "name", "local", "", true, false},
		{"name:/tmp", "external", "/tmp", "", "name", "external", "", true, false},
		{"name:/tmp:ro", "local", "/tmp", "", "name", "local", "", false, false},
		{"local/name:/tmp:rw", "", "/tmp", "", "local/name", "local", "", true, false},
	}

	for _, c := range cases {
		conf := &runconfig.Config{VolumeDriver: c.driver}
		m, err := parseBindMount(c.bind, c.mountLabel, conf)
		if c.fail {
			if err == nil {
				t.Fatalf("Expected error, was nil, for spec %s\n", c.bind)
			}
			continue
		}

		if m.Destination != c.expDest {
			t.Fatalf("Expected destination %s, was %s, for spec %s\n", c.expDest, m.Destination, c.bind)
		}

		if m.Source != c.expSource {
			t.Fatalf("Expected source %s, was %s, for spec %s\n", c.expSource, m.Source, c.bind)
		}

		if m.Name != c.expName {
			t.Fatalf("Expected name %s, was %s for spec %s\n", c.expName, m.Name, c.bind)
		}

		if m.Driver != c.expDriver {
			t.Fatalf("Expected driver %s, was %s, for spec %s\n", c.expDriver, m.Driver, c.bind)
		}

		if m.RW != c.expRW {
			t.Fatalf("Expected RW %v, was %v for spec %s\n", c.expRW, m.RW, c.bind)
		}
	}
}
