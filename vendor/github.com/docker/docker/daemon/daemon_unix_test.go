// +build !windows,!solaris

package daemon

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
	"github.com/docker/docker/volume/store"
)

// Unix test as uses settings which are not available on Windows
func TestAdjustCPUShares(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-unix-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	daemon := &Daemon{
		repository: tmp,
		root:       tmp,
	}

	hostConfig := &containertypes.HostConfig{
		Resources: containertypes.Resources{CPUShares: linuxMinCPUShares - 1},
	}
	daemon.adaptContainerSettings(hostConfig, true)
	if hostConfig.CPUShares != linuxMinCPUShares {
		t.Errorf("Expected CPUShares to be %d", linuxMinCPUShares)
	}

	hostConfig.CPUShares = linuxMaxCPUShares + 1
	daemon.adaptContainerSettings(hostConfig, true)
	if hostConfig.CPUShares != linuxMaxCPUShares {
		t.Errorf("Expected CPUShares to be %d", linuxMaxCPUShares)
	}

	hostConfig.CPUShares = 0
	daemon.adaptContainerSettings(hostConfig, true)
	if hostConfig.CPUShares != 0 {
		t.Error("Expected CPUShares to be unchanged")
	}

	hostConfig.CPUShares = 1024
	daemon.adaptContainerSettings(hostConfig, true)
	if hostConfig.CPUShares != 1024 {
		t.Error("Expected CPUShares to be unchanged")
	}
}

// Unix test as uses settings which are not available on Windows
func TestAdjustCPUSharesNoAdjustment(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-unix-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	daemon := &Daemon{
		repository: tmp,
		root:       tmp,
	}

	hostConfig := &containertypes.HostConfig{
		Resources: containertypes.Resources{CPUShares: linuxMinCPUShares - 1},
	}
	daemon.adaptContainerSettings(hostConfig, false)
	if hostConfig.CPUShares != linuxMinCPUShares-1 {
		t.Errorf("Expected CPUShares to be %d", linuxMinCPUShares-1)
	}

	hostConfig.CPUShares = linuxMaxCPUShares + 1
	daemon.adaptContainerSettings(hostConfig, false)
	if hostConfig.CPUShares != linuxMaxCPUShares+1 {
		t.Errorf("Expected CPUShares to be %d", linuxMaxCPUShares+1)
	}

	hostConfig.CPUShares = 0
	daemon.adaptContainerSettings(hostConfig, false)
	if hostConfig.CPUShares != 0 {
		t.Error("Expected CPUShares to be unchanged")
	}

	hostConfig.CPUShares = 1024
	daemon.adaptContainerSettings(hostConfig, false)
	if hostConfig.CPUShares != 1024 {
		t.Error("Expected CPUShares to be unchanged")
	}
}

// Unix test as uses settings which are not available on Windows
func TestParseSecurityOptWithDeprecatedColon(t *testing.T) {
	container := &container.Container{}
	config := &containertypes.HostConfig{}

	// test apparmor
	config.SecurityOpt = []string{"apparmor=test_profile"}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}
	if container.AppArmorProfile != "test_profile" {
		t.Fatalf("Unexpected AppArmorProfile, expected: \"test_profile\", got %q", container.AppArmorProfile)
	}

	// test seccomp
	sp := "/path/to/seccomp_test.json"
	config.SecurityOpt = []string{"seccomp=" + sp}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}
	if container.SeccompProfile != sp {
		t.Fatalf("Unexpected AppArmorProfile, expected: %q, got %q", sp, container.SeccompProfile)
	}

	// test valid label
	config.SecurityOpt = []string{"label=user:USER"}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}

	// test invalid label
	config.SecurityOpt = []string{"label"}
	if err := parseSecurityOpt(container, config); err == nil {
		t.Fatal("Expected parseSecurityOpt error, got nil")
	}

	// test invalid opt
	config.SecurityOpt = []string{"test"}
	if err := parseSecurityOpt(container, config); err == nil {
		t.Fatal("Expected parseSecurityOpt error, got nil")
	}
}

func TestParseSecurityOpt(t *testing.T) {
	container := &container.Container{}
	config := &containertypes.HostConfig{}

	// test apparmor
	config.SecurityOpt = []string{"apparmor=test_profile"}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}
	if container.AppArmorProfile != "test_profile" {
		t.Fatalf("Unexpected AppArmorProfile, expected: \"test_profile\", got %q", container.AppArmorProfile)
	}

	// test seccomp
	sp := "/path/to/seccomp_test.json"
	config.SecurityOpt = []string{"seccomp=" + sp}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}
	if container.SeccompProfile != sp {
		t.Fatalf("Unexpected SeccompProfile, expected: %q, got %q", sp, container.SeccompProfile)
	}

	// test valid label
	config.SecurityOpt = []string{"label=user:USER"}
	if err := parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected parseSecurityOpt error: %v", err)
	}

	// test invalid label
	config.SecurityOpt = []string{"label"}
	if err := parseSecurityOpt(container, config); err == nil {
		t.Fatal("Expected parseSecurityOpt error, got nil")
	}

	// test invalid opt
	config.SecurityOpt = []string{"test"}
	if err := parseSecurityOpt(container, config); err == nil {
		t.Fatal("Expected parseSecurityOpt error, got nil")
	}
}

func TestParseNNPSecurityOptions(t *testing.T) {
	daemon := &Daemon{
		configStore: &config.Config{NoNewPrivileges: true},
	}
	container := &container.Container{}
	config := &containertypes.HostConfig{}

	// test NNP when "daemon:true" and "no-new-privileges=false""
	config.SecurityOpt = []string{"no-new-privileges=false"}

	if err := daemon.parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected daemon.parseSecurityOpt error: %v", err)
	}
	if container.NoNewPrivileges {
		t.Fatalf("container.NoNewPrivileges should be FALSE: %v", container.NoNewPrivileges)
	}

	// test NNP when "daemon:false" and "no-new-privileges=true""
	daemon.configStore.NoNewPrivileges = false
	config.SecurityOpt = []string{"no-new-privileges=true"}

	if err := daemon.parseSecurityOpt(container, config); err != nil {
		t.Fatalf("Unexpected daemon.parseSecurityOpt error: %v", err)
	}
	if !container.NoNewPrivileges {
		t.Fatalf("container.NoNewPrivileges should be TRUE: %v", container.NoNewPrivileges)
	}
}

func TestNetworkOptions(t *testing.T) {
	daemon := &Daemon{}
	dconfigCorrect := &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore:     "consul://localhost:8500",
			ClusterAdvertise: "192.168.0.1:8000",
		},
	}

	if _, err := daemon.networkOptions(dconfigCorrect, nil, nil); err != nil {
		t.Fatalf("Expect networkOptions success, got error: %v", err)
	}

	dconfigWrong := &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore: "consul://localhost:8500://test://bbb",
		},
	}

	if _, err := daemon.networkOptions(dconfigWrong, nil, nil); err == nil {
		t.Fatal("Expected networkOptions error, got nil")
	}
}

func TestMigratePre17Volumes(t *testing.T) {
	rootDir, err := ioutil.TempDir("", "test-daemon-volumes")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(rootDir)

	volumeRoot := filepath.Join(rootDir, "volumes")
	err = os.MkdirAll(volumeRoot, 0755)
	if err != nil {
		t.Fatal(err)
	}

	containerRoot := filepath.Join(rootDir, "containers")
	cid := "1234"
	err = os.MkdirAll(filepath.Join(containerRoot, cid), 0755)

	vid := "5678"
	vfsPath := filepath.Join(rootDir, "vfs", "dir", vid)
	err = os.MkdirAll(vfsPath, 0755)
	if err != nil {
		t.Fatal(err)
	}

	config := []byte(`
		{
			"ID": "` + cid + `",
			"Volumes": {
				"/foo": "` + vfsPath + `",
				"/bar": "/foo",
				"/quux": "/quux"
			},
			"VolumesRW": {
				"/foo": true,
				"/bar": true,
				"/quux": false
			}
		}
	`)

	volStore, err := store.New(volumeRoot)
	if err != nil {
		t.Fatal(err)
	}
	drv, err := local.New(volumeRoot, idtools.IDPair{UID: 0, GID: 0})
	if err != nil {
		t.Fatal(err)
	}
	volumedrivers.Register(drv, volume.DefaultDriverName)

	daemon := &Daemon{
		root:       rootDir,
		repository: containerRoot,
		volumes:    volStore,
	}
	err = ioutil.WriteFile(filepath.Join(containerRoot, cid, "config.v2.json"), config, 600)
	if err != nil {
		t.Fatal(err)
	}
	c, err := daemon.load(cid)
	if err != nil {
		t.Fatal(err)
	}
	if err := daemon.verifyVolumesInfo(c); err != nil {
		t.Fatal(err)
	}

	expected := map[string]volume.MountPoint{
		"/foo":  {Destination: "/foo", RW: true, Name: vid},
		"/bar":  {Source: "/foo", Destination: "/bar", RW: true},
		"/quux": {Source: "/quux", Destination: "/quux", RW: false},
	}
	for id, mp := range c.MountPoints {
		x, exists := expected[id]
		if !exists {
			t.Fatal("volume not migrated")
		}
		if mp.Source != x.Source || mp.Destination != x.Destination || mp.RW != x.RW || mp.Name != x.Name {
			t.Fatalf("got unexpected mountpoint, expected: %+v, got: %+v", x, mp)
		}
	}
}
