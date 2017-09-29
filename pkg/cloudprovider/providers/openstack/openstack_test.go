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
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/apiversions"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	volumeAvailableStatus = "available"
	volumeInUseStatus     = "in-use"
	testClusterName       = "testCluster"

	volumeStatusTimeoutSeconds = 30
	// volumeStatus* is configuration of exponential backoff for
	// waiting for specified volume status. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 13 steps at maximum
	// it will time out after 32s, which roughly corresponds to 30s
	volumeStatusInitDealy = 1 * time.Second
	volumeStatusFactor    = 1.2
	volumeStatusSteps     = 13
)

func WaitForVolumeStatus(t *testing.T, os *OpenStack, volumeName string, status string) {
	backoff := wait.Backoff{
		Duration: volumeStatusInitDealy,
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
		} else {
			return false, nil
		}
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
 bs-version = auto
 trust-device-path = yes
 [Metadata]
 search-order = configDrive, metadataService
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
	if cfg.BlockStorage.BSVersion != "auto" {
		t.Errorf("incorrect bs.bs-version: %v", cfg.BlockStorage.BSVersion)
	}
	if cfg.Metadata.SearchOrder != "configDrive, metadataService" {
		t.Errorf("incorrect md.search-order: %v", cfg.Metadata.SearchOrder)
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
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
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
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
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
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
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
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
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
			expectedError: fmt.Errorf("node-security-group not set in cloud provider config"),
		},
		{
			name: "test5",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "",
				},
			},
			expectedError: fmt.Errorf("Invalid value in section [Metadata] with key `search-order`. Value cannot be empty"),
		},
		{
			name: "test6",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "value1,value2,value3",
				},
			},
			expectedError: fmt.Errorf("Invalid value in section [Metadata] with key `search-order`. Value cannot contain more than 2 elements"),
		},
		{
			name: "test7",
			openstackOpts: &OpenStack{
				provider: nil,
				metadataOpts: MetadataOpts{
					SearchOrder: "value1",
				},
			},
			expectedError: fmt.Errorf("Invalid element '%s' found in section [Metadata] with key `search-order`."+
				"Supported elements include '%s' and '%s'", "value1", configDriveID, metadataID),
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
	cfg.Global.Region = os.Getenv("OS_REGION_NAME")
	cfg.Global.DomainId = os.Getenv("OS_DOMAIN_ID")
	cfg.Global.DomainName = os.Getenv("OS_DOMAIN_NAME")

	ok = (cfg.Global.AuthUrl != "" &&
		cfg.Global.Username != "" &&
		cfg.Global.Password != "" &&
		(cfg.Global.TenantId != "" || cfg.Global.TenantName != "" ||
			cfg.Global.DomainId != "" || cfg.Global.DomainName != ""))

	cfg.Metadata.SearchOrder = fmt.Sprintf("%s,%s", configDriveID, metadataID)

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

		_, exists, err := lb.GetLoadBalancer(testClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "noexist"}})
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
	vol, _, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, "", "", &tags)
	if err != nil {
		t.Fatalf("Cannot create a new Cinder volume: %v", err)
	}
	t.Logf("Volume (%s) created\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	id, err := os.InstanceID()
	if err != nil {
		t.Fatalf("Cannot find instance id: %v", err)
	}

	diskId, err := os.AttachDisk(id, vol)
	if err != nil {
		t.Fatalf("Cannot AttachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) attached, disk ID: %s\n", vol, diskId)

	WaitForVolumeStatus(t, os, vol, volumeInUseStatus)

	devicePath := os.GetDevicePath(diskId)
	if !strings.HasPrefix(devicePath, "/dev/disk/by-id/") {
		t.Fatalf("GetDevicePath returned and unexpected path for Cinder volume %s, returned %s", vol, devicePath)
	}
	t.Logf("Volume (%s) found at path: %s\n", vol, devicePath)

	err = os.DetachDisk(id, vol)
	if err != nil {
		t.Fatalf("Cannot DetachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) detached\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	err = os.DeleteVolume(vol)
	if err != nil {
		t.Fatalf("Cannot delete Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) deleted\n", vol)

}

func TestCinderAutoDetectApiVersion(t *testing.T) {
	updated := "" // not relevant to this test, can be set to any value
	status_current := "CURRENT"
	status_supported := "SUPpORTED" // lowercase to test regression resitance if api returns different case
	status_deprecated := "DEPRECATED"

	var result_version, api_version [4]string

	for ver := 0; ver <= 3; ver++ {
		api_version[ver] = fmt.Sprintf("v%d.0", ver)
		result_version[ver] = fmt.Sprintf("v%d", ver)
	}
	result_version[0] = ""
	api_current_v1 := apiversions.APIVersion{ID: api_version[1], Status: status_current, Updated: updated}
	api_current_v2 := apiversions.APIVersion{ID: api_version[2], Status: status_current, Updated: updated}
	api_current_v3 := apiversions.APIVersion{ID: api_version[3], Status: status_current, Updated: updated}

	api_supported_v1 := apiversions.APIVersion{ID: api_version[1], Status: status_supported, Updated: updated}
	api_supported_v2 := apiversions.APIVersion{ID: api_version[2], Status: status_supported, Updated: updated}

	api_deprecated_v1 := apiversions.APIVersion{ID: api_version[1], Status: status_deprecated, Updated: updated}
	api_deprecated_v2 := apiversions.APIVersion{ID: api_version[2], Status: status_deprecated, Updated: updated}

	var testCases = []struct {
		test_case     []apiversions.APIVersion
		wanted_result string
	}{
		{[]apiversions.APIVersion{api_current_v1}, result_version[1]},
		{[]apiversions.APIVersion{api_current_v2}, result_version[2]},
		{[]apiversions.APIVersion{api_supported_v1, api_current_v2}, result_version[2]},                     // current always selected
		{[]apiversions.APIVersion{api_current_v1, api_supported_v2}, result_version[1]},                     // current always selected
		{[]apiversions.APIVersion{api_current_v3, api_supported_v2, api_deprecated_v1}, result_version[2]},  // with current v3, but should fall back to v2
		{[]apiversions.APIVersion{api_current_v3, api_deprecated_v2, api_deprecated_v1}, result_version[0]}, // v3 is not supported
	}

	for _, suite := range testCases {
		if autodetectedVersion := doBsApiVersionAutodetect(suite.test_case); autodetectedVersion != suite.wanted_result {
			t.Fatalf("Autodetect for suite: %s, failed with result: '%s', wanted '%s'", suite.test_case, autodetectedVersion, suite.wanted_result)
		}
	}
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
