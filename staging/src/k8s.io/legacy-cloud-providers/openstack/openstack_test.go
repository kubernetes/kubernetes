// +build !providerless

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
	"context"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	v1 "k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	testClusterName = "testCluster"

	volumeStatusTimeoutSeconds = 30
	// volumeStatus* is configuration of exponential backoff for
	// waiting for specified volume status. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 13 steps at maximum
	// it will time out after 32s, which roughly corresponds to 30s
	volumeStatusInitDelay = 1 * time.Second
	volumeStatusFactor    = 1.2
	volumeStatusSteps     = 13
)

func WaitForVolumeStatus(t *testing.T, os *OpenStack, volumeName string, status string) {
	backoff := wait.Backoff{
		Duration: volumeStatusInitDelay,
		Factor:   volumeStatusFactor,
		Steps:    volumeStatusSteps,
	}
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		getVol, err := os.getVolume(volumeName)
		if err != nil {
			return false, err
		}
		if getVol.Status == status {
			t.Logf("Volume (%s) status changed to %s after %v seconds\n",
				volumeName,
				status,
				volumeStatusTimeoutSeconds)
			return true, nil
		}
		return false, nil
	})
	if err == wait.ErrWaitTimeout {
		t.Logf("Volume (%s) status did not change to %s after %v seconds\n",
			volumeName,
			status,
			volumeStatusTimeoutSeconds)
		return
	}
	if err != nil {
		t.Fatalf("Cannot get existing Cinder volume (%s): %v", volumeName, err)
	}
}

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	// Since we are setting env vars, we need to reset old
	// values for other tests to succeed.
	env := clearEnviron(t)
	defer resetEnviron(t, env)

	os.Setenv("OS_PASSWORD", "mypass")
	defer os.Unsetenv("OS_PASSWORD")

	os.Setenv("OS_TENANT_NAME", "admin")
	defer os.Unsetenv("OS_TENANT_NAME")

	cfg, err := readConfig(strings.NewReader(`
 [Global]
 auth-url = http://auth.url
 user-id = user
 tenant-name = demo
 region = RegionOne
 [LoadBalancer]
 create-monitor = yes
 monitor-delay = 1m
 monitor-timeout = 30s
 monitor-max-retries = 3
 [BlockStorage]
 bs-version = auto
 trust-device-path = yes
 ignore-volume-az = yes
 [Metadata]
 search-order = configDrive, metadataService
 `))
	cfg.Global.Password = os.Getenv("OS_PASSWORD")

	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.AuthURL != "http://auth.url" {
		t.Errorf("incorrect authurl: %s", cfg.Global.AuthURL)
	}

	if cfg.Global.UserID != "user" {
		t.Errorf("incorrect userid: %s", cfg.Global.UserID)
	}

	if cfg.Global.Password != "mypass" {
		t.Errorf("incorrect password: %s", cfg.Global.Password)
	}

	// config file wins over environment variable
	if cfg.Global.TenantName != "demo" {
		t.Errorf("incorrect tenant name: %s", cfg.Global.TenantName)
	}

	if cfg.Global.Region != "RegionOne" {
		t.Errorf("incorrect region: %s", cfg.Global.Region)
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
	if cfg.BlockStorage.BSVersion != "auto" {
		t.Errorf("incorrect bs.bs-version: %v", cfg.BlockStorage.BSVersion)
	}
	if cfg.BlockStorage.IgnoreVolumeAZ != true {
		t.Errorf("incorrect bs.IgnoreVolumeAZ: %v", cfg.BlockStorage.IgnoreVolumeAZ)
	}
	if cfg.Metadata.SearchOrder != "configDrive, metadataService" {
		t.Errorf("incorrect md.search-order: %v", cfg.Metadata.SearchOrder)
	}
}

func TestToAuthOptions(t *testing.T) {
	cfg := Config{}
	cfg.Global.Username = "user"
	cfg.Global.Password = "pass"
	cfg.Global.DomainID = "2a73b8f597c04551a0fdc8e95544be8a"
	cfg.Global.DomainName = "local"
	cfg.Global.AuthURL = "http://auth.url"
	cfg.Global.UserID = "user"

	ao := cfg.toAuthOptions()

	if !ao.AllowReauth {
		t.Errorf("Will need to be able to reauthenticate")
	}
	if ao.Username != cfg.Global.Username {
		t.Errorf("Username %s != %s", ao.Username, cfg.Global.Username)
	}
	if ao.Password != cfg.Global.Password {
		t.Errorf("Password %s != %s", ao.Password, cfg.Global.Password)
	}
	if ao.DomainID != cfg.Global.DomainID {
		t.Errorf("DomainID %s != %s", ao.DomainID, cfg.Global.DomainID)
	}
	if ao.IdentityEndpoint != cfg.Global.AuthURL {
		t.Errorf("IdentityEndpoint %s != %s", ao.IdentityEndpoint, cfg.Global.AuthURL)
	}
	if ao.UserID != cfg.Global.UserID {
		t.Errorf("UserID %s != %s", ao.UserID, cfg.Global.UserID)
	}
	if ao.DomainName != cfg.Global.DomainName {
		t.Errorf("DomainName %s != %s", ao.DomainName, cfg.Global.DomainName)
	}
	if ao.TenantID != cfg.Global.TenantID {
		t.Errorf("TenantID %s != %s", ao.TenantID, cfg.Global.TenantID)
	}
}

func TestCheckOpenStackOpts(t *testing.T) {
	delay := MyDuration{60 * time.Second}
	timeout := MyDuration{30 * time.Second}
	tests := []struct {
		name          string
		openstackOpts *OpenStack
		expectedError error
	}{
		{
			name: "test1",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetID:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkID:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					LBProvider:           "haproxy",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
				},
				metadataOpts: MetadataOpts{
					SearchOrder: configDriveID,
				},
			},
			expectedError: nil,
		},
		{
			name: "test2",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					FloatingNetworkID:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
				},
				metadataOpts: MetadataOpts{
					SearchOrder: configDriveID,
				},
			},
			expectedError: nil,
		},
		{
			name: "test3",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetID:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkID:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
				},
				metadataOpts: MetadataOpts{
					SearchOrder: configDriveID,
				},
			},
			expectedError: fmt.Errorf("monitor-delay not set in cloud provider config"),
		},
		{
			name: "test4",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "",
				},
			},
			expectedError: fmt.Errorf("invalid value in section [Metadata] with key `search-order`. Value cannot be empty"),
		},
		{
			name: "test5",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "value1,value2,value3",
				},
			},
			expectedError: fmt.Errorf("invalid value in section [Metadata] with key `search-order`. Value cannot contain more than 2 elements"),
		},
		{
			name: "test6",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "value1",
				},
			},
			expectedError: fmt.Errorf("invalid element %q found in section [Metadata] with key `search-order`."+
				"Supported elements include %q and %q", "value1", configDriveID, metadataID),
		},
		{
			name: "test7",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetID:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkID:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					ManageSecurityGroups: true,
				},
				metadataOpts: MetadataOpts{
					SearchOrder: configDriveID,
				},
			},
			expectedError: fmt.Errorf("monitor-max-retries not set in cloud provider config"),
		},
		{
			name: "test8",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetID:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkID:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
				},
				metadataOpts: MetadataOpts{
					SearchOrder: configDriveID,
				},
			},
			expectedError: fmt.Errorf("monitor-timeout not set in cloud provider config"),
		},
	}

	for _, testcase := range tests {
		err := checkOpenStackOpts(testcase.openstackOpts)

		if err == nil && testcase.expectedError == nil {
			continue
		}
		if (err != nil && testcase.expectedError == nil) || (err == nil && testcase.expectedError != nil) || err.Error() != testcase.expectedError.Error() {
			t.Errorf("%s failed: expected err=%q, got %q",
				testcase.name, testcase.expectedError, err)
		}
	}
}

func TestCaller(t *testing.T) {
	called := false
	myFunc := func() { called = true }

	c := newCaller()
	c.call(myFunc)

	if !called {
		t.Errorf("caller failed to call function in default case")
	}

	c.disarm()
	called = false
	c.call(myFunc)

	if called {
		t.Error("caller still called function when disarmed")
	}

	// Confirm the "usual" deferred caller pattern works as expected

	called = false
	successCase := func() {
		c := newCaller()
		defer c.call(func() { called = true })
		c.disarm()
	}
	if successCase(); called {
		t.Error("Deferred success case still invoked unwind")
	}

	called = false
	failureCase := func() {
		c := newCaller()
		defer c.call(func() { called = true })
	}
	if failureCase(); !called {
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
		Metadata: map[string]string{
			"name":       "a1-yinvcez57-0-bvynoyawrhcg-kube-minion-fg5i4jwcc2yy",
			TypeHostName: "a1-yinvcez57-0-bvynoyawrhcg-kube-minion-fg5i4jwcc2yy.novalocal",
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
		{Type: v1.NodeHostName, Address: "a1-yinvcez57-0-bvynoyawrhcg-kube-minion-fg5i4jwcc2yy.novalocal"},
	}

	if !reflect.DeepEqual(want, addrs) {
		t.Errorf("nodeAddresses returned incorrect value %v", addrs)
	}
}

func configFromEnvWithPasswd() (cfg Config, ok bool) {
	cfg, ok = configFromEnv()
	if !ok {
		return cfg, ok
	}
	cfg.Global.Password = os.Getenv("OS_PASSWORD")
	return cfg, ok
}

func TestNewOpenStack(t *testing.T) {
	cfg, ok := configFromEnvWithPasswd()
	if !ok {
		t.Skip("No config found in environment")
	}

	_, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}
}

func TestLoadBalancer(t *testing.T) {
	cfg, ok := configFromEnvWithPasswd()
	if !ok {
		t.Skip("No config found in environment")
	}

	versions := []string{"v2", ""}

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

		_, exists, err := lb.GetLoadBalancer(context.TODO(), testClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "noexist"}})
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

	zone, err := z.GetZone(context.TODO())
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

var diskPathRegexp = regexp.MustCompile("/dev/disk/(?:by-id|by-path)/")

func TestVolumes(t *testing.T) {
	cfg, ok := configFromEnvWithPasswd()
	if !ok {
		t.Skip("No config found in environment")
	}

	os, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}

	tags := map[string]string{
		"test": "value",
	}
	vol, _, _, _, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, "", "", &tags)
	if err != nil {
		t.Fatalf("Cannot create a new Cinder volume: %v", err)
	}
	t.Logf("Volume (%s) created\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	id, err := os.InstanceID()
	if err != nil {
		t.Logf("Cannot find instance id: %v - perhaps you are running this test outside a VM launched by OpenStack", err)
	} else {
		diskID, err := os.AttachDisk(id, vol)
		if err != nil {
			t.Fatalf("Cannot AttachDisk Cinder volume %s: %v", vol, err)
		}
		t.Logf("Volume (%s) attached, disk ID: %s\n", vol, diskID)

		WaitForVolumeStatus(t, os, vol, volumeInUseStatus)

		devicePath := os.GetDevicePath(diskID)
		if diskPathRegexp.FindString(devicePath) == "" {
			t.Fatalf("GetDevicePath returned and unexpected path for Cinder volume %s, returned %s", vol, devicePath)
		}
		t.Logf("Volume (%s) found at path: %s\n", vol, devicePath)

		err = os.DetachDisk(id, vol)
		if err != nil {
			t.Fatalf("Cannot DetachDisk Cinder volume %s: %v", vol, err)
		}
		t.Logf("Volume (%s) detached\n", vol)

		WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)
	}

	expectedVolSize := resource.MustParse("2Gi")
	newVolSize, err := os.ExpandVolume(vol, resource.MustParse("1Gi"), expectedVolSize)
	if err != nil {
		t.Fatalf("Cannot expand a Cinder volume: %v", err)
	}
	if newVolSize != expectedVolSize {
		t.Logf("Expected: %v but got: %v ", expectedVolSize, newVolSize)
	}
	t.Logf("Volume expanded to (%v) \n", newVolSize)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	err = os.DeleteVolume(vol)
	if err != nil {
		t.Fatalf("Cannot delete Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) deleted\n", vol)

}

func TestInstanceIDFromProviderID(t *testing.T) {
	testCases := []struct {
		providerID string
		instanceID string
		fail       bool
	}{
		{
			providerID: ProviderName + "://" + "/" + "7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "7b9cf879-7146-417c-abfd-cb4272f0c935",
			fail:       false,
		},
		{
			providerID: "openstack://7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
		{
			providerID: "7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
		{
			providerID: "other-provider:///7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
	}

	for _, test := range testCases {
		instanceID, err := instanceIDFromProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("%s yielded `err != nil` as %t. expected %t", test.providerID, (err != nil), test.fail)
		}

		if test.fail {
			continue
		}

		if instanceID != test.instanceID {
			t.Errorf("%s yielded %s. expected %s", test.providerID, instanceID, test.instanceID)
		}
	}
}

func TestToAuth3Options(t *testing.T) {
	cfg := Config{}
	cfg.Global.Username = "user"
	cfg.Global.Password = "pass"
	cfg.Global.DomainID = "2a73b8f597c04551a0fdc8e95544be8a"
	cfg.Global.DomainName = "local"
	cfg.Global.AuthURL = "http://auth.url"
	cfg.Global.UserID = "user"

	ao := cfg.toAuth3Options()

	if !ao.AllowReauth {
		t.Errorf("Will need to be able to reauthenticate")
	}
	if ao.Username != cfg.Global.Username {
		t.Errorf("Username %s != %s", ao.Username, cfg.Global.Username)
	}
	if ao.Password != cfg.Global.Password {
		t.Errorf("Password %s != %s", ao.Password, cfg.Global.Password)
	}
	if ao.DomainID != cfg.Global.DomainID {
		t.Errorf("DomainID %s != %s", ao.DomainID, cfg.Global.DomainID)
	}
	if ao.IdentityEndpoint != cfg.Global.AuthURL {
		t.Errorf("IdentityEndpoint %s != %s", ao.IdentityEndpoint, cfg.Global.AuthURL)
	}
	if ao.UserID != cfg.Global.UserID {
		t.Errorf("UserID %s != %s", ao.UserID, cfg.Global.UserID)
	}
	if ao.DomainName != cfg.Global.DomainName {
		t.Errorf("DomainName %s != %s", ao.DomainName, cfg.Global.DomainName)
	}
}

func clearEnviron(t *testing.T) []string {
	env := os.Environ()
	for _, pair := range env {
		if strings.HasPrefix(pair, "OS_") {
			i := strings.Index(pair, "=") + 1
			os.Unsetenv(pair[:i-1])
		}
	}
	return env
}
func resetEnviron(t *testing.T, items []string) {
	for _, pair := range items {
		if strings.HasPrefix(pair, "OS_") {
			i := strings.Index(pair, "=") + 1
			if err := os.Setenv(pair[:i-1], pair[i:]); err != nil {
				t.Errorf("Setenv(%q, %q) failed during reset: %v", pair[:i-1], pair[i:], err)
			}
		}
	}
}
