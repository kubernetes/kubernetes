// +build linux

package daemon

import (
	"testing"

	"github.com/docker/docker/runconfig"
)

func TestMergeLxcConfig(t *testing.T) {
	kv := []runconfig.KeyValuePair{
		{"lxc.cgroups.cpuset", "1,2"},
	}
	hostConfig := &runconfig.HostConfig{
		LxcConf: runconfig.NewLxcConfig(kv),
	}

	out, err := mergeLxcConfIntoOptions(hostConfig)
	if err != nil {
		t.Fatalf("Failed to merge Lxc Config: %s", err)
	}

	cpuset := out[0]
	if expected := "cgroups.cpuset=1,2"; cpuset != expected {
		t.Fatalf("expected %s got %s", expected, cpuset)
	}
}
