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
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/apiversions"
	"k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

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
