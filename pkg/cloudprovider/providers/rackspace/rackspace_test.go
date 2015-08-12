/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package rackspace

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rackspace/gophercloud"
)

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

// This allows acceptance testing against an existing Rackspace
// install, using the standard OS_* Rackspace client environment
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

func TestNewRackspace(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newRackspace(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate Rackspace: %s", err)
	}
}

func TestInstances(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	os, err := newRackspace(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate Rackspace: %s", err)
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

	addrs, err := i.NodeAddresses(srvs[0])
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", srvs[0], addrs)

	rsrcs, err := i.GetNodeResources(srvs[0])
	if err != nil {
		t.Fatalf("Instances.GetNodeResources(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found GetNodeResources(%s) = %s\n", srvs[0], rsrcs)
}

func TestZones(t *testing.T) {
	os := Rackspace{
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
