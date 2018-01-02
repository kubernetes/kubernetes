package runconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/docker/api/types/container"
	networktypes "github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/strslice"
)

type f struct {
	file       string
	entrypoint strslice.StrSlice
}

func TestDecodeContainerConfig(t *testing.T) {

	var (
		fixtures []f
		image    string
	)

	//TODO: Should run for Solaris
	if runtime.GOOS == "solaris" {
		t.Skip()
	}

	if runtime.GOOS != "windows" {
		image = "ubuntu"
		fixtures = []f{
			{"fixtures/unix/container_config_1_14.json", strslice.StrSlice{}},
			{"fixtures/unix/container_config_1_17.json", strslice.StrSlice{"bash"}},
			{"fixtures/unix/container_config_1_19.json", strslice.StrSlice{"bash"}},
		}
	} else {
		image = "windows"
		fixtures = []f{
			{"fixtures/windows/container_config_1_19.json", strslice.StrSlice{"cmd"}},
		}
	}

	for _, f := range fixtures {
		b, err := ioutil.ReadFile(f.file)
		if err != nil {
			t.Fatal(err)
		}

		c, h, _, err := DecodeContainerConfig(bytes.NewReader(b))
		if err != nil {
			t.Fatal(fmt.Errorf("Error parsing %s: %v", f, err))
		}

		if c.Image != image {
			t.Fatalf("Expected %s image, found %s\n", image, c.Image)
		}

		if len(c.Entrypoint) != len(f.entrypoint) {
			t.Fatalf("Expected %v, found %v\n", f.entrypoint, c.Entrypoint)
		}

		if h != nil && h.Memory != 1000 {
			t.Fatalf("Expected memory to be 1000, found %d\n", h.Memory)
		}
	}
}

// TestDecodeContainerConfigIsolation validates isolation passed
// to the daemon in the hostConfig structure. Note this is platform specific
// as to what level of container isolation is supported.
func TestDecodeContainerConfigIsolation(t *testing.T) {

	// An Invalid isolation level
	if _, _, _, err := callDecodeContainerConfigIsolation("invalid"); err != nil {
		if !strings.Contains(err.Error(), `Invalid isolation: "invalid"`) {
			t.Fatal(err)
		}
	}

	// Blank isolation (== default)
	if _, _, _, err := callDecodeContainerConfigIsolation(""); err != nil {
		t.Fatal("Blank isolation should have succeeded")
	}

	// Default isolation
	if _, _, _, err := callDecodeContainerConfigIsolation("default"); err != nil {
		t.Fatal("default isolation should have succeeded")
	}

	// Process isolation (Valid on Windows only)
	if runtime.GOOS == "windows" {
		if _, _, _, err := callDecodeContainerConfigIsolation("process"); err != nil {
			t.Fatal("process isolation should have succeeded")
		}
	} else {
		if _, _, _, err := callDecodeContainerConfigIsolation("process"); err != nil {
			if !strings.Contains(err.Error(), `Invalid isolation: "process"`) {
				t.Fatal(err)
			}
		}
	}

	// Hyper-V Containers isolation (Valid on Windows only)
	if runtime.GOOS == "windows" {
		if _, _, _, err := callDecodeContainerConfigIsolation("hyperv"); err != nil {
			t.Fatal("hyperv isolation should have succeeded")
		}
	} else {
		if _, _, _, err := callDecodeContainerConfigIsolation("hyperv"); err != nil {
			if !strings.Contains(err.Error(), `Invalid isolation: "hyperv"`) {
				t.Fatal(err)
			}
		}
	}
}

// callDecodeContainerConfigIsolation is a utility function to call
// DecodeContainerConfig for validating isolation
func callDecodeContainerConfigIsolation(isolation string) (*container.Config, *container.HostConfig, *networktypes.NetworkingConfig, error) {
	var (
		b   []byte
		err error
	)
	w := ContainerConfigWrapper{
		Config: &container.Config{},
		HostConfig: &container.HostConfig{
			NetworkMode: "none",
			Isolation:   container.Isolation(isolation)},
	}
	if b, err = json.Marshal(w); err != nil {
		return nil, nil, nil, fmt.Errorf("Error on marshal %s", err.Error())
	}
	return DecodeContainerConfig(bytes.NewReader(b))
}
