/*
Copyright 2016 The Kubernetes Authors.

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

package cloudstack

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testClusterName = "testCluster"

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %v", err)
	}

	cfg, err := readConfig(strings.NewReader(`
 [Global]
 api-url				= https://cloudstack.url
 api-key				= a-valid-api-key
 secret-key			= a-valid-secret-key
 ssl-no-verify	= true
 project-id			= a-valid-project-id
 zone						= a-valid-zone
 `))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %v", err)
	}

	if cfg.Global.APIURL != "https://cloudstack.url" {
		t.Errorf("incorrect api-url: %s", cfg.Global.APIURL)
	}
	if cfg.Global.APIKey != "a-valid-api-key" {
		t.Errorf("incorrect api-key: %s", cfg.Global.APIKey)
	}
	if cfg.Global.SecretKey != "a-valid-secret-key" {
		t.Errorf("incorrect secret-key: %s", cfg.Global.SecretKey)
	}
	if !cfg.Global.SSLNoVerify {
		t.Errorf("incorrect ssl-no-verify: %t", cfg.Global.SSLNoVerify)
	}
	if cfg.Global.Zone != "a-valid-zone" {
		t.Errorf("incorrect zone: %s", cfg.Global.Zone)
	}
}

// This allows acceptance testing against an existing CloudStack environment.
func configFromEnv() (*CSConfig, bool) {
	cfg := &CSConfig{}

	cfg.Global.APIURL = os.Getenv("CS_API_URL")
	cfg.Global.APIKey = os.Getenv("CS_API_KEY")
	cfg.Global.SecretKey = os.Getenv("CS_SECRET_KEY")
	cfg.Global.ProjectID = os.Getenv("CS_PROJECT_ID")
	cfg.Global.Zone = os.Getenv("CS_ZONE")

	// It is save to ignore the error here. If the input cannot be parsed SSLNoVerify
	// will still be a bool with its zero value (false) which is the expected default.
	cfg.Global.SSLNoVerify, _ = strconv.ParseBool(os.Getenv("CS_SSL_NO_VERIFY"))

	// Check if we have the minimum required info to be able to connect to CloudStack.
	ok := cfg.Global.APIURL != "" && cfg.Global.APIKey != "" && cfg.Global.SecretKey != ""

	return cfg, ok
}

func TestNewCSCloud(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newCSCloud(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate CloudStack: %v", err)
	}
}

func TestLoadBalancer(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	cs, err := newCSCloud(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate CloudStack: %v", err)
	}

	lb, ok := cs.LoadBalancer()
	if !ok {
		t.Fatalf("LoadBalancer() returned false")
	}

	_, exists, err := lb.GetLoadBalancer(testClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "noexist"}})
	if err != nil {
		t.Fatalf("GetLoadBalancer(\"noexist\") returned error: %s", err)
	}
	if exists {
		t.Fatalf("GetLoadBalancer(\"noexist\") returned exists")
	}
}

func TestZones(t *testing.T) {
	cs := &CSCloud{
		zone: "myRegion",
	}

	z, ok := cs.Zones()
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
