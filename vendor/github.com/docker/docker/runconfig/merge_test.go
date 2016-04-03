package runconfig

import (
	"testing"

	"github.com/docker/docker/pkg/nat"
)

func TestMerge(t *testing.T) {
	volumesImage := make(map[string]struct{})
	volumesImage["/test1"] = struct{}{}
	volumesImage["/test2"] = struct{}{}
	portsImage := make(nat.PortSet)
	portsImage[newPortNoError("tcp", "1111")] = struct{}{}
	portsImage[newPortNoError("tcp", "2222")] = struct{}{}
	configImage := &Config{
		ExposedPorts: portsImage,
		Env:          []string{"VAR1=1", "VAR2=2"},
		Volumes:      volumesImage,
	}

	portsUser := make(nat.PortSet)
	portsUser[newPortNoError("tcp", "2222")] = struct{}{}
	portsUser[newPortNoError("tcp", "3333")] = struct{}{}
	volumesUser := make(map[string]struct{})
	volumesUser["/test3"] = struct{}{}
	configUser := &Config{
		ExposedPorts: portsUser,
		Env:          []string{"VAR2=3", "VAR3=3"},
		Volumes:      volumesUser,
	}

	if err := Merge(configUser, configImage); err != nil {
		t.Error(err)
	}

	if len(configUser.ExposedPorts) != 3 {
		t.Fatalf("Expected 3 ExposedPorts, 1111, 2222 and 3333, found %d", len(configUser.ExposedPorts))
	}
	for portSpecs := range configUser.ExposedPorts {
		if portSpecs.Port() != "1111" && portSpecs.Port() != "2222" && portSpecs.Port() != "3333" {
			t.Fatalf("Expected 1111 or 2222 or 3333, found %s", portSpecs)
		}
	}
	if len(configUser.Env) != 3 {
		t.Fatalf("Expected 3 env var, VAR1=1, VAR2=3 and VAR3=3, found %d", len(configUser.Env))
	}
	for _, env := range configUser.Env {
		if env != "VAR1=1" && env != "VAR2=3" && env != "VAR3=3" {
			t.Fatalf("Expected VAR1=1 or VAR2=3 or VAR3=3, found %s", env)
		}
	}

	if len(configUser.Volumes) != 3 {
		t.Fatalf("Expected 3 volumes, /test1, /test2 and /test3, found %d", len(configUser.Volumes))
	}
	for v := range configUser.Volumes {
		if v != "/test1" && v != "/test2" && v != "/test3" {
			t.Fatalf("Expected /test1 or /test2 or /test3, found %s", v)
		}
	}

	ports, _, err := nat.ParsePortSpecs([]string{"0000"})
	if err != nil {
		t.Error(err)
	}
	configImage2 := &Config{
		ExposedPorts: ports,
	}

	if err := Merge(configUser, configImage2); err != nil {
		t.Error(err)
	}

	if len(configUser.ExposedPorts) != 4 {
		t.Fatalf("Expected 4 ExposedPorts, 0000, 1111, 2222 and 3333, found %d", len(configUser.ExposedPorts))
	}
	for portSpecs := range configUser.ExposedPorts {
		if portSpecs.Port() != "0" && portSpecs.Port() != "1111" && portSpecs.Port() != "2222" && portSpecs.Port() != "3333" {
			t.Fatalf("Expected %q or %q or %q or %q, found %s", 0, 1111, 2222, 3333, portSpecs)
		}
	}
}
