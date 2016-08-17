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
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/rand"

	"github.com/rackspace/gophercloud"
	"k8s.io/kubernetes/pkg/api"
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

		_, exists, err := lb.GetLoadBalancer(testClusterName, &api.Service{ObjectMeta: api.ObjectMeta{Name: "noexist"}})
		if err != nil {
			t.Fatalf("GetLoadBalancer(\"noexist\") returned error: %s", err)
		}
		if exists {
			t.Fatalf("GetLoadBalancer(\"noexist\") returned exists")
		}
	}
}

func TestZones(t *testing.T) {
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
	vol, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, &tags)
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
