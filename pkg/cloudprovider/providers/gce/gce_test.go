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

package gce

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/oauth2/google"

	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	computev1 "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestReadConfigFile(t *testing.T) {
	const s = `[Global]
token-url = my-token-url
token-body = my-token-body
project-id = my-project
network-project-id = my-network-project
network-name = my-network
subnetwork-name = my-subnetwork
secondary-range-name = my-secondary-range
node-tags = my-node-tag1
node-instance-prefix = my-prefix
multizone = true
   `
	reader := strings.NewReader(s)
	config, err := readConfig(reader)
	if err != nil {
		t.Fatalf("Unexpected config parsing error %v", err)
	}

	expected := &ConfigFile{Global: ConfigGlobal{
		TokenURL:           "my-token-url",
		TokenBody:          "my-token-body",
		ProjectID:          "my-project",
		NetworkProjectID:   "my-network-project",
		NetworkName:        "my-network",
		SubnetworkName:     "my-subnetwork",
		SecondaryRangeName: "my-secondary-range",
		NodeTags:           []string{"my-node-tag1"},
		NodeInstancePrefix: "my-prefix",
		Multizone:          true,
	}}

	if !reflect.DeepEqual(expected, config) {
		t.Fatalf("Expected config file values to be read into ConfigFile struct.  \nExpected:\n%+v\nActual:\n%+v", expected, config)
	}
}

func TestExtraKeyInConfig(t *testing.T) {
	const s = `[Global]
project-id = my-project
unknown-key = abc
network-name = my-network
   `
	reader := strings.NewReader(s)
	config, err := readConfig(reader)
	if err != nil {
		t.Fatalf("Unexpected config parsing error %v", err)
	}
	if config.Global.ProjectID != "my-project" || config.Global.NetworkName != "my-network" {
		t.Fatalf("Expected config values to continue to be read despite extra key-value pair.")
	}
}

func TestGetRegion(t *testing.T) {
	zoneName := "us-central1-b"
	regionName, err := GetGCERegion(zoneName)
	if err != nil {
		t.Fatalf("unexpected error from GetGCERegion: %v", err)
	}
	if regionName != "us-central1" {
		t.Errorf("Unexpected region from GetGCERegion: %s", regionName)
	}
	gce := &GCECloud{
		localZone: zoneName,
		region:    regionName,
	}
	zones, ok := gce.Zones()
	if !ok {
		t.Fatalf("Unexpected missing zones impl")
	}
	zone, err := zones.GetZone()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if zone.Region != "us-central1" {
		t.Errorf("Unexpected region: %s", zone.Region)
	}
}

func TestComparingHostURLs(t *testing.T) {
	tests := []struct {
		host1       string
		zone        string
		name        string
		expectEqual bool
	}{
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/cool-project/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v23/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v24/projects/1234567/regions/us-central1/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-c",
			name:        "kubernetes-node-fhx1",
			expectEqual: false,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: false,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx",
			expectEqual: false,
		},
	}

	for _, test := range tests {
		link1 := hostURLToComparablePath(test.host1)
		testInstance := &gceInstance{
			Name: canonicalizeInstanceName(test.name),
			Zone: test.zone,
		}
		link2 := testInstance.makeComparableHostPath()
		if test.expectEqual && link1 != link2 {
			t.Errorf("expected link1 and link2 to be equal, got %s and %s", link1, link2)
		} else if !test.expectEqual && link1 == link2 {
			t.Errorf("expected link1 and link2 not to be equal, got %s and %s", link1, link2)
		}
	}
}

func TestScrubDNS(t *testing.T) {
	tcs := []struct {
		nameserversIn  []string
		searchesIn     []string
		nameserversOut []string
		searchesOut    []string
	}{
		{
			nameserversIn:  []string{"1.2.3.4", "5.6.7.8"},
			nameserversOut: []string{"1.2.3.4", "5.6.7.8"},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "google.internal."},
			searchesOut: []string{"c.prj.internal.", "google.internal."},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "zone.c.prj.internal.", "google.internal."},
			searchesOut: []string{"c.prj.internal.", "zone.c.prj.internal.", "google.internal."},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "zone.c.prj.internal.", "google.internal.", "unexpected"},
			searchesOut: []string{"c.prj.internal.", "zone.c.prj.internal.", "google.internal.", "unexpected"},
		},
	}
	gce := &GCECloud{}
	for i := range tcs {
		n, s := gce.ScrubDNS(tcs[i].nameserversIn, tcs[i].searchesIn)
		if !reflect.DeepEqual(n, tcs[i].nameserversOut) {
			t.Errorf("Expected %v, got %v", tcs[i].nameserversOut, n)
		}
		if !reflect.DeepEqual(s, tcs[i].searchesOut) {
			t.Errorf("Expected %v, got %v", tcs[i].searchesOut, s)
		}
	}
}

func TestSplitProviderID(t *testing.T) {
	providers := []struct {
		providerID string

		project  string
		zone     string
		instance string

		fail bool
	}{
		{
			providerID: ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "project-example-164317",
			zone:       "us-central1-f",
			instance:   "kubernetes-node-fhx1",
			fail:       false,
		},
		{
			providerID: ProviderName + "://project-example.164317/us-central1-f/kubernetes-node-fhx1",
			project:    "project-example.164317",
			zone:       "us-central1-f",
			instance:   "kubernetes-node-fhx1",
			fail:       false,
		},
		{
			providerID: ProviderName + "://project-example-164317/us-central1-fkubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + ":/project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: "aws://project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1/",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example.164317//kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example.164317/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
	}

	for _, test := range providers {
		project, zone, instance, err := splitProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("Expected to fail=%t, with pattern %v", test.fail, test)
		}

		if test.fail {
			continue
		}

		if project != test.project {
			t.Errorf("Expected %v, but got %v", test.project, project)
		}
		if zone != test.zone {
			t.Errorf("Expected %v, but got %v", test.zone, zone)
		}
		if instance != test.instance {
			t.Errorf("Expected %v, but got %v", test.instance, instance)
		}
	}
}

func TestGetZoneByProviderID(t *testing.T) {
	tests := []struct {
		providerID string

		expectedZone cloudprovider.Zone

		fail        bool
		description string
	}{
		{
			providerID:   ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1",
			expectedZone: cloudprovider.Zone{FailureDomain: "us-central1-f", Region: "us-central1"},
			fail:         false,
			description:  "standard gce providerID",
		},
		{
			providerID:   ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1/",
			expectedZone: cloudprovider.Zone{},
			fail:         true,
			description:  "too many slashes('/') trailing",
		},
		{
			providerID:   ProviderName + "://project-example.164317//kubernetes-node-fhx1",
			expectedZone: cloudprovider.Zone{},
			fail:         true,
			description:  "too many slashes('/') embedded",
		},
		{
			providerID:   ProviderName + "://project-example-164317/uscentral1f/kubernetes-node-fhx1",
			expectedZone: cloudprovider.Zone{},
			fail:         true,
			description:  "invalid name of the GCE zone",
		},
	}

	gce := &GCECloud{
		localZone: "us-central1-f",
		region:    "us-central1",
	}
	for _, test := range tests {
		zone, err := gce.GetZoneByProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("Expected to fail=%t, provider ID %v, tests %s", test.fail, test, test.description)
		}

		if test.fail {
			continue
		}

		if zone != test.expectedZone {
			t.Errorf("Expected %v, but got %v", test.expectedZone, zone)
		}
	}
}

func TestGenerateCloudConfigs(t *testing.T) {
	configBoilerplate := ConfigGlobal{
		TokenURL:           "",
		TokenBody:          "",
		ProjectID:          "project-id",
		NetworkName:        "network-name",
		SubnetworkName:     "",
		SecondaryRangeName: "",
		NodeTags:           []string{"node-tag"},
		NodeInstancePrefix: "node-prefix",
		Multizone:          false,
		ApiEndpoint:        "",
		LocalZone:          "us-central1-a",
		AlphaFeatures:      []string{},
	}

	cloudBoilerplate := CloudConfig{
		ApiEndpoint:        "",
		ProjectID:          "project-id",
		NetworkProjectID:   "",
		Region:             "us-central1",
		Zone:               "us-central1-a",
		ManagedZones:       []string{"us-central1-a"},
		NetworkName:        "network-name",
		SubnetworkName:     "",
		NetworkURL:         "",
		SubnetworkURL:      "",
		SecondaryRangeName: "",
		NodeTags:           []string{"node-tag"},
		TokenSource:        google.ComputeTokenSource(""),
		NodeInstancePrefix: "node-prefix",
		UseMetadataServer:  true,
		AlphaFeatureGate:   &AlphaFeatureGate{map[string]bool{}},
	}

	testCases := []struct {
		name   string
		config func() ConfigGlobal
		cloud  func() CloudConfig
	}{
		{
			name:   "Empty Config",
			config: func() ConfigGlobal { return configBoilerplate },
			cloud:  func() CloudConfig { return cloudBoilerplate },
		},
		{
			name: "Nil token URL",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.TokenURL = "nil"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.TokenSource = nil
				return v
			},
		},
		{
			name: "Network Project ID",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.NetworkProjectID = "my-awesome-project"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.NetworkProjectID = "my-awesome-project"
				return v
			},
		},
		{
			name: "Specified API Endpint",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.ApiEndpoint = "https://www.googleapis.com/compute/staging_v1/"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.ApiEndpoint = "https://www.googleapis.com/compute/staging_v1/"
				return v
			},
		},
		{
			name: "Network & Subnetwork names",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.NetworkName = "my-network"
				v.SubnetworkName = "my-subnetwork"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.NetworkName = "my-network"
				v.SubnetworkName = "my-subnetwork"
				return v
			},
		},
		{
			name: "Network & Subnetwork URLs",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.NetworkName = "https://www.googleapis.com/compute/v1/projects/project-id/global/networks/my-network"
				v.SubnetworkName = "https://www.googleapis.com/compute/v1/projects/project-id/regions/us-central1/subnetworks/my-subnetwork"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.NetworkName = ""
				v.SubnetworkName = ""
				v.NetworkURL = "https://www.googleapis.com/compute/v1/projects/project-id/global/networks/my-network"
				v.SubnetworkURL = "https://www.googleapis.com/compute/v1/projects/project-id/regions/us-central1/subnetworks/my-subnetwork"
				return v
			},
		},
		{
			name: "Multizone",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.Multizone = true
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.ManagedZones = nil
				return v
			},
		},
		{
			name: "Secondary Range Name",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.SecondaryRangeName = "my-secondary"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.SecondaryRangeName = "my-secondary"
				return v
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resultCloud, err := generateCloudConfig(&ConfigFile{Global: tc.config()})
			if err != nil {
				t.Fatalf("Unexpect error: %v", err)
			}

			v := tc.cloud()
			if !reflect.DeepEqual(*resultCloud, v) {
				t.Errorf("Got: \n%v\nWant\n%v\n", v, *resultCloud)
			}
		})
	}
}

func TestConvertToV1Operation(t *testing.T) {
	v1Op := getTestOperation()
	enc, _ := v1Op.MarshalJSON()
	var op interface{}
	var alphaOp computealpha.Operation
	var betaOp computebeta.Operation

	if err := json.Unmarshal(enc, &alphaOp); err != nil {
		t.Errorf("Failed to unmarshal operation: %v", err)
	}

	if err := json.Unmarshal(enc, &betaOp); err != nil {
		t.Errorf("Failed to unmarshal operation: %v", err)
	}

	op = convertToV1Operation(&alphaOp)
	if _, ok := op.(*computev1.Operation); ok {
		if !reflect.DeepEqual(op, v1Op) {
			t.Errorf("Failed to maintain consistency across conversion")
		}
	} else {
		t.Errorf("Expect output to be type v1 operation, but got %v", op)
	}

	op = convertToV1Operation(&betaOp)
	if _, ok := op.(*computev1.Operation); ok {
		if !reflect.DeepEqual(op, v1Op) {
			t.Errorf("Failed to maintain consistency across conversion")
		}
	} else {
		t.Errorf("Expect output to be type v1 operation, but got %v", op)
	}
}

func getTestOperation() *computev1.Operation {
	return &computev1.Operation{
		Name:        "test",
		Description: "test",
		Id:          uint64(12345),
		Error: &computev1.OperationError{
			Errors: []*computev1.OperationErrorErrors{
				{
					Code:    "555",
					Message: "error",
				},
			},
		},
	}
}

func TestNewAlphaFeatureGate(t *testing.T) {
	knownAlphaFeatures["foo"] = true
	knownAlphaFeatures["bar"] = true

	testCases := []struct {
		alphaFeatures  []string
		expectEnabled  []string
		expectDisabled []string
		expectError    bool
	}{
		// enable foo bar
		{
			alphaFeatures:  []string{"foo", "bar"},
			expectEnabled:  []string{"foo", "bar"},
			expectDisabled: []string{"aaa"},
			expectError:    false,
		},
		// no alpha feature
		{
			alphaFeatures:  []string{},
			expectEnabled:  []string{},
			expectDisabled: []string{"foo", "bar"},
			expectError:    false,
		},
		// unsupported alpha feature
		{
			alphaFeatures:  []string{"aaa", "foo"},
			expectError:    true,
			expectEnabled:  []string{"foo"},
			expectDisabled: []string{"aaa"},
		},
		// enable foo
		{
			alphaFeatures:  []string{"foo"},
			expectEnabled:  []string{"foo"},
			expectDisabled: []string{"bar"},
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		featureGate, err := NewAlphaFeatureGate(tc.alphaFeatures)

		if (tc.expectError && err == nil) || (!tc.expectError && err != nil) {
			t.Errorf("Expect error to be %v, but got error %v", tc.expectError, err)
		}

		for _, key := range tc.expectEnabled {
			if !featureGate.Enabled(key) {
				t.Errorf("Expect %q to be enabled.", key)
			}
		}
		for _, key := range tc.expectDisabled {
			if featureGate.Enabled(key) {
				t.Errorf("Expect %q to be disabled.", key)
			}
		}
	}
	delete(knownAlphaFeatures, "foo")
	delete(knownAlphaFeatures, "bar")
}
