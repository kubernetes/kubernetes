package configs

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Checks whether the expected capability is specified in the capabilities.
func contains(expected string, values []string) bool {
	for _, v := range values {
		if v == expected {
			return true
		}
	}
	return false
}

func containsDevice(expected *Device, values []*Device) bool {
	for _, d := range values {
		if d.Path == expected.Path &&
			d.Permissions == expected.Permissions &&
			d.FileMode == expected.FileMode &&
			d.Major == expected.Major &&
			d.Minor == expected.Minor &&
			d.Type == expected.Type {
			return true
		}
	}
	return false
}

func loadConfig(name string) (*Config, error) {
	f, err := os.Open(filepath.Join("../sample_configs", name))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var container *Config
	if err := json.NewDecoder(f).Decode(&container); err != nil {
		return nil, err
	}

	// Check that a config doesn't contain extra fields
	var configMap, abstractMap map[string]interface{}

	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}

	if err := json.NewDecoder(f).Decode(&abstractMap); err != nil {
		return nil, err
	}

	configData, err := json.Marshal(&container)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(configData, &configMap); err != nil {
		return nil, err
	}

	for k := range configMap {
		delete(abstractMap, k)
	}

	if len(abstractMap) != 0 {
		return nil, fmt.Errorf("unknown fields: %s", abstractMap)
	}

	return container, nil
}

func TestConfigJsonFormat(t *testing.T) {
	container, err := loadConfig("attach_to_bridge.json")
	if err != nil {
		t.Fatal(err)
	}

	if container.Hostname != "koye" {
		t.Log("hostname is not set")
		t.Fail()
	}

	if !container.Namespaces.Contains(NEWNET) {
		t.Log("namespaces should contain NEWNET")
		t.Fail()
	}

	if container.Namespaces.Contains(NEWUSER) {
		t.Log("namespaces should not contain NEWUSER")
		t.Fail()
	}

	if contains("SYS_ADMIN", container.Capabilities) {
		t.Log("SYS_ADMIN should not be enabled in capabilities mask")
		t.Fail()
	}

	if !contains("MKNOD", container.Capabilities) {
		t.Log("MKNOD should be enabled in capabilities mask")
		t.Fail()
	}

	if !contains("SYS_CHROOT", container.Capabilities) {
		t.Log("capabilities mask should contain SYS_CHROOT")
		t.Fail()
	}

	for _, n := range container.Networks {
		if n.Type == "veth" {
			if n.Bridge != "docker0" {
				t.Logf("veth bridge should be docker0 but received %q", n.Bridge)
				t.Fail()
			}

			if n.Address != "172.17.0.101/16" {
				t.Logf("veth address should be 172.17.0.101/61 but received %q", n.Address)
				t.Fail()
			}

			if n.Gateway != "172.17.42.1" {
				t.Logf("veth gateway should be 172.17.42.1 but received %q", n.Gateway)
				t.Fail()
			}

			if n.Mtu != 1500 {
				t.Logf("veth mtu should be 1500 but received %d", n.Mtu)
				t.Fail()
			}

			break
		}
	}
	for _, d := range DefaultSimpleDevices {
		if !containsDevice(d, container.Devices) {
			t.Logf("expected device configuration for %s", d.Path)
			t.Fail()
		}
	}
}

func TestApparmorProfile(t *testing.T) {
	container, err := loadConfig("apparmor.json")
	if err != nil {
		t.Fatal(err)
	}

	if container.AppArmorProfile != "docker-default" {
		t.Fatalf("expected apparmor profile to be docker-default but received %q", container.AppArmorProfile)
	}
}

func TestSelinuxLabels(t *testing.T) {
	container, err := loadConfig("selinux.json")
	if err != nil {
		t.Fatal(err)
	}
	label := "system_u:system_r:svirt_lxc_net_t:s0:c164,c475"

	if container.ProcessLabel != label {
		t.Fatalf("expected process label %q but received %q", label, container.ProcessLabel)
	}
	if container.MountLabel != label {
		t.Fatalf("expected mount label %q but received %q", label, container.MountLabel)
	}
}

func TestRemoveNamespace(t *testing.T) {
	ns := Namespaces{
		{Type: NEWNET},
	}
	if !ns.Remove(NEWNET) {
		t.Fatal("NEWNET was not removed")
	}
	if len(ns) != 0 {
		t.Fatalf("namespaces should have 0 items but reports %d", len(ns))
	}
}

func TestHostUIDNoUSERNS(t *testing.T) {
	config := &Config{
		Namespaces: Namespaces{},
	}
	uid, err := config.HostUID()
	if err != nil {
		t.Fatal(err)
	}
	if uid != 0 {
		t.Fatalf("expected uid 0 with no USERNS but received %d", uid)
	}
}

func TestHostUIDWithUSERNS(t *testing.T) {
	config := &Config{
		Namespaces: Namespaces{{Type: NEWUSER}},
		UidMappings: []IDMap{
			{
				ContainerID: 0,
				HostID:      1000,
				Size:        1,
			},
		},
	}
	uid, err := config.HostUID()
	if err != nil {
		t.Fatal(err)
	}
	if uid != 1000 {
		t.Fatalf("expected uid 1000 with no USERNS but received %d", uid)
	}
}

func TestHostGIDNoUSERNS(t *testing.T) {
	config := &Config{
		Namespaces: Namespaces{},
	}
	uid, err := config.HostGID()
	if err != nil {
		t.Fatal(err)
	}
	if uid != 0 {
		t.Fatalf("expected gid 0 with no USERNS but received %d", uid)
	}
}

func TestHostGIDWithUSERNS(t *testing.T) {
	config := &Config{
		Namespaces: Namespaces{{Type: NEWUSER}},
		GidMappings: []IDMap{
			{
				ContainerID: 0,
				HostID:      1000,
				Size:        1,
			},
		},
	}
	uid, err := config.HostGID()
	if err != nil {
		t.Fatal(err)
	}
	if uid != 1000 {
		t.Fatalf("expected gid 1000 with no USERNS but received %d", uid)
	}
}
