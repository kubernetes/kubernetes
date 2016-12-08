/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package openstack

import (
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/rand"
)

const volumeAvailableStatus = "available"
const volumeInUseStatus = "in-use"
const volumeCreateTimeoutSeconds = 30
const testClusterName = "testCluster"

func WaitForVolumeStatus(t *testing.T, os *OpenStack, volumeName string, status string, timeoutSeconds int) {
	timeout := timeoutSeconds
	start := time.Now().Second()
	for {
		time.Sleep(1 * time.Second)

		if timeout >= 0 && time.Now().Second()-start >= timeout {
			t.Logf("Volume (%s) status did not change to %s after %v seconds\n",
				volumeName,
				status,
				timeout)
			return
		}

		getVol, err := os.getVolume(volumeName)
		if err != nil {
			t.Fatalf("Cannot get existing Cinder volume (%s): %v", volumeName, err)
		}
		if getVol.Status == status {
			t.Logf("Volume (%s) status changed to %s after %v seconds\n",
				volumeName,
				status,
				timeout)
			return
		}
	}
}

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
 [Global]
 auth-url = http://auth.url
 username = user
 [LoadBalancer]
 create-monitor = yes
 monitor-delay = 1m
 monitor-timeout = 30s
 monitor-max-retries = 3
 [BlockStorage]
 trust-device-path = yes
 `))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.AuthUrl != "http://auth.url" {
		t.Errorf("incorrect authurl: %s", cfg.Global.AuthUrl)
	}

	if !cfg.LoadBalancer.CreateMonitor {
		t.Errorf("incorrect lb.createmonitor: %t", cfg.LoadBalancer.CreateMonitor)
	}
	if cfg.LoadBalancer.MonitorDelay.Duration != 1*time.Minute {
		t.Errorf("incorrect lb.monitordelay: %s", cfg.LoadBalancer.MonitorDelay)
	}
	if cfg.LoadBalancer.MonitorTimeout.Duration != 30*time.Second {
		t.Errorf("incorrect lb.monitortimeout: %s", cfg.LoadBalancer.MonitorTimeout)
	}
	if cfg.LoadBalancer.MonitorMaxRetries != 3 {
		t.Errorf("incorrect lb.monitormaxretries: %d", cfg.LoadBalancer.MonitorMaxRetries)
	}
	if cfg.BlockStorage.TrustDevicePath != true {
		t.Errorf("incorrect bs.trustdevicepath: %v", cfg.BlockStorage.TrustDevicePath)
	}
}

func TestToAuthOptions(t *testing.T) {
	cfg := Config{}
	cfg.Global.Username = "user"
	// etc.

	ao := cfg.toAuthOptions()

	if !ao.AllowReauth {
		t.Errorf("Will need to be able to reauthenticate")
	}
	if ao.Username != cfg.Global.Username {
		t.Errorf("Username %s != %s", ao.Username, cfg.Global.Username)
	}
}

func TestCaller(t *testing.T) {
	called := false
	myFunc := func() { called = true }

	c := NewCaller()
	c.Call(myFunc)

	if !called {
		t.Errorf("Caller failed to call function in default case")
	}

	c.Disarm()
	called = false
	c.Call(myFunc)

	if called {
		t.Error("Caller still called function when disarmed")
	}

	// Confirm the "usual" deferred Caller pattern works as expected

	called = false
	success_case := func() {
		c := NewCaller()
		defer c.Call(func() { called = true })
		c.Disarm()
	}
	if success_case(); called {
		t.Error("Deferred success case still invoked unwind")
	}

	called = false
	failure_case := func() {
		c := NewCaller()
		defer c.Call(func() { called = true })
	}
	if failure_case(); !called {
		t.Error("Deferred failure case failed to invoke unwind")
	}
}

// An arbitrary sort.Interface, just for easier comparison
type AddressSlice []v1.NodeAddress

func (a AddressSlice) Len() int           { return len(a) }
func (a AddressSlice) Less(i, j int) bool { return a[i].Address < a[j].Address }
func (a AddressSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func TestNodeAddresses(t *testing.T) {
	srv := servers.Server{
		Status:     "ACTIVE",
		HostID:     "29d3c8c896a45aa4c34e52247875d7fefc3d94bbcc9f622b5d204362",
		AccessIPv4: "50.56.176.99",
		AccessIPv6: "2001:4800:790e:510:be76:4eff:fe04:82a8",
		Addresses: map[string]interface{}{
			"private": []interface{}{
				map[string]interface{}{
					"OS-EXT-IPS-MAC:mac_addr": "fa:16:3e:7c:1b:2b",
					"version":                 float64(4),
					"addr":                    "10.0.0.32",
					"OS-EXT-IPS:type":         "fixed",
				},
				map[string]interface{}{
					"version":         float64(4),
					"addr":            "50.56.176.36",
					"OS-EXT-IPS:type": "floating",
				},
				map[string]interface{}{
					"version": float64(4),
					"addr":    "10.0.0.31",
					// No OS-EXT-IPS:type
				},
			},
			"public": []interface{}{
				map[string]interface{}{
					"version": float64(4),
					"addr":    "50.56.176.35",
				},
				map[string]interface{}{
					"version": float64(6),
					"addr":    "2001:4800:780e:510:be76:4eff:fe04:84a8",
				},
			},
		},
	}

	addrs, err := nodeAddresses(&srv)
	if err != nil {
		t.Fatalf("nodeAddresses returned error: %v", err)
	}

	sort.Sort(AddressSlice(addrs))
	t.Logf("addresses is %v", addrs)

	want := []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: "10.0.0.31"},
		{Type: v1.NodeInternalIP, Address: "10.0.0.32"},
		{Type: v1.NodeExternalIP, Address: "2001:4800:780e:510:be76:4eff:fe04:84a8"},
		{Type: v1.NodeExternalIP, Address: "2001:4800:790e:510:be76:4eff:fe04:82a8"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.35"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.36"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.99"},
	}

	if !reflect.DeepEqual(want, addrs) {
		t.Errorf("nodeAddresses returned incorrect value %v", addrs)
	}
}

// This allows acceptance testing against an existing OpenStack
// install, using the standard OS_* OpenStack client environment
// variables.
// FIXME: it would be better to hermetically test against canned JSON
// requests/responses.
func configFromEnv() (cfg Config, ok bool) {
	cfg.Global.AuthUrl = os.Getenv("OS_AUTH_URL")

	cfg.Global.TenantId = os.Getenv("OS_TENANT_ID")
	// Rax/nova _insists_ that we don't specify both tenant ID and name
	if cfg.Global.TenantId == "" {
		cfg.Global.TenantName = os.Getenv("OS_TENANT_NAME")
	}

	cfg.Global.Username = os.Getenv("OS_USERNAME")
	cfg.Global.Password = os.Getenv("OS_PASSWORD")
	cfg.Global.ApiKey = os.Getenv("OS_API_KEY")
	cfg.Global.Region = os.Getenv("OS_REGION_NAME")
	cfg.Global.DomainId = os.Getenv("OS_DOMAIN_ID")
	cfg.Global.DomainName = os.Getenv("OS_DOMAIN_NAME")

	ok = (cfg.Global.AuthUrl != "" &&
		cfg.Global.Username != "" &&
		(cfg.Global.Password != "" || cfg.Global.ApiKey != "") &&
		(cfg.Global.TenantId != "" || cfg.Global.TenantName != "" ||
			cfg.Global.DomainId != "" || cfg.Global.DomainName != ""))

	return
}

func TestNewOpenStack(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}
}

func TestInstances(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	os, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}

	i, ok := os.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	srvs, err := i.List(".")
	if err != nil {
		t.Fatalf("Instances.List() failed: %s", err)
	}
	if len(srvs) == 0 {
		t.Fatalf("Instances.List() returned zero servers")
	}
	t.Logf("Found servers (%d): %s\n", len(srvs), srvs)

	srvExternalId, err := i.ExternalID(srvs[0])
	if err != nil {
		t.Fatalf("Instances.ExternalId(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found server (%s), with external id: %s\n", srvs[0], srvExternalId)

	srvInstanceId, err := i.InstanceID(srvs[0])
	if err != nil {
		t.Fatalf("Instance.InstanceId(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found server (%s), with instance id: %s\n", srvs[0], srvInstanceId)

	addrs, err := i.NodeAddresses(srvs[0])
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", srvs[0], addrs)
}

func TestLoadBalancer(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	versions := []string{"v1", "v2", ""}

	for _, v := range versions {
		t.Logf("Trying LBVersion = '%s'\n", v)
		cfg.LoadBalancer.LBVersion = v

		os, err := newOpenStack(cfg)
		if err != nil {
			t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
		}

		lb, ok := os.LoadBalancer()
		if !ok {
			t.Fatalf("LoadBalancer() returned false - perhaps your stack doesn't support Neutron?")
		}

		_, exists, err := lb.GetLoadBalancer(testClusterName, &v1.Service{ObjectMeta: v1.ObjectMeta{Name: "noexist"}})
		if err != nil {
			t.Fatalf("GetLoadBalancer(\"noexist\") returned error: %s", err)
		}
		if exists {
			t.Fatalf("GetLoadBalancer(\"noexist\") returned exists")
		}
	}
}

func TestZones(t *testing.T) {
	SetMetadataFixture(&FakeMetadata)
	defer ClearMetadata()

	os := OpenStack{
		provider: &gophercloud.ProviderClient{
			IdentityBase: "http://auth.url/",
		},
		region: "myRegion",
	}

	z, ok := os.Zones()
	if !ok {
		t.Fatalf("Zones() returned false")
	}

	zone, err := z.GetZone()
	if err != nil {
		t.Fatalf("GetZone() returned error: %s", err)
	}

	if zone.Region != "myRegion" {
		t.Fatalf("GetZone() returned wrong region (%s)", zone.Region)
	}

	if zone.FailureDomain != "nova" {
		t.Fatalf("GetZone() returned wrong failure domain (%s)", zone.FailureDomain)
	}
}

func TestVolumes(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	os, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}

	tags := map[string]string{
		"test": "value",
	}
	vol, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, "", "", &tags)
	if err != nil {
		t.Fatalf("Cannot create a new Cinder volume: %v", err)
	}
	t.Logf("Volume (%s) created\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus, volumeCreateTimeoutSeconds)

	diskId, err := os.AttachDisk(os.localInstanceID, vol)
	if err != nil {
		t.Fatalf("Cannot AttachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) attached, disk ID: %s\n", vol, diskId)

	WaitForVolumeStatus(t, os, vol, volumeInUseStatus, volumeCreateTimeoutSeconds)

	devicePath := os.GetDevicePath(diskId)
	if !strings.HasPrefix(devicePath, "/dev/disk/by-id/") {
		t.Fatalf("GetDevicePath returned and unexpected path for Cinder volume %s, returned %s", vol, devicePath)
	}
	t.Logf("Volume (%s) found at path: %s\n", vol, devicePath)

	err = os.DetachDisk(os.localInstanceID, vol)
	if err != nil {
		t.Fatalf("Cannot DetachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) detached\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus, volumeCreateTimeoutSeconds)

	err = os.DeleteVolume(vol)
	if err != nil {
		t.Fatalf("Cannot delete Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) deleted\n", vol)

}
