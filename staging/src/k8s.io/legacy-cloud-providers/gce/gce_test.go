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

package gce

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/oauth2/google"

	cloudprovider "k8s.io/cloud-provider"
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
regional = true
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
		Regional:           true,
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
	gce := &Cloud{
		localZone: zoneName,
		region:    regionName,
	}
	zones, ok := gce.Zones()
	if !ok {
		t.Fatalf("Unexpected missing zones impl")
	}
	zone, err := zones.GetZone(context.TODO())
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

	gce := &Cloud{
		localZone: "us-central1-f",
		region:    "us-central1",
	}
	for _, test := range tests {
		zone, err := gce.GetZoneByProviderID(context.TODO(), test.providerID)
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
		Regional:           false,
		APIEndpoint:        "",
		LocalZone:          "us-central1-a",
		AlphaFeatures:      []string{},
	}

	cloudBoilerplate := CloudConfig{
		APIEndpoint:        "",
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
				v.APIEndpoint = "https://www.googleapis.com/compute/staging_v1/"
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.APIEndpoint = "https://www.googleapis.com/compute/staging_v1/"
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
			name: "Regional",
			config: func() ConfigGlobal {
				v := configBoilerplate
				v.Regional = true
				return v
			},
			cloud: func() CloudConfig {
				v := cloudBoilerplate
				v.Regional = true
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

func TestNewAlphaFeatureGate(t *testing.T) {
	testCases := []struct {
		alphaFeatures  []string
		expectEnabled  []string
		expectDisabled []string
	}{
		// enable foo bar
		{
			alphaFeatures:  []string{"foo", "bar"},
			expectEnabled:  []string{"foo", "bar"},
			expectDisabled: []string{"aaa"},
		},
		// no alpha feature
		{
			alphaFeatures:  []string{},
			expectEnabled:  []string{},
			expectDisabled: []string{"foo", "bar"},
		},
		// unsupported alpha feature
		{
			alphaFeatures:  []string{"aaa", "foo"},
			expectEnabled:  []string{"foo"},
			expectDisabled: []string{},
		},
		// enable foo
		{
			alphaFeatures:  []string{"foo"},
			expectEnabled:  []string{"foo"},
			expectDisabled: []string{"bar"},
		},
	}

	for _, tc := range testCases {
		featureGate := NewAlphaFeatureGate(tc.alphaFeatures)

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
}

func TestGetRegionInURL(t *testing.T) {
	cases := map[string]string{
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-central1/subnetworks/a": "us-central1",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-west2/subnetworks/b":    "us-west2",
		"projects/my-project/regions/asia-central1/subnetworks/c":                                     "asia-central1",
		"regions/europe-north2": "europe-north2",
		"my-url":                "",
		"":                      "",
	}
	for input, output := range cases {
		result := getRegionInURL(input)
		if result != output {
			t.Errorf("Actual result %q does not match expected result %q for input: %q", result, output, input)
		}
	}
}

func TestFindSubnetForRegion(t *testing.T) {
	s := []string{
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-central1/subnetworks/default-38b01f54907a15a7",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-west1/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-east1/subnetworks/default-277eec3815f742b6",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-east4/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/asia-northeast1/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/asia-east1/subnetworks/default-8e020b4b8b244809",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/australia-southeast1/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/southamerica-east1/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/europe-west3/subnetworks/default",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/asia-southeast1/subnetworks/default",
		"",
	}
	actual := findSubnetForRegion(s, "asia-east1")
	expectedResult := "https://www.googleapis.com/compute/v1/projects/my-project/regions/asia-east1/subnetworks/default-8e020b4b8b244809"
	if actual != expectedResult {
		t.Errorf("Actual result %q does not match expected result %q", actual, expectedResult)
	}

	var nilSlice []string
	res := findSubnetForRegion(nilSlice, "us-central1")
	if res != "" {
		t.Errorf("expected an empty result, got %v", res)
	}
}

func TestLastComponent(t *testing.T) {
	cases := map[string]string{
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-central1/subnetworks/a": "a",
		"https://www.googleapis.com/compute/v1/projects/my-project/regions/us-central1/subnetworks/b": "b",
		"projects/my-project/regions/us-central1/subnetworks/c":                                       "c",
		"d": "d",
		"":  "",
	}
	for input, output := range cases {
		result := lastComponent(input)
		if result != output {
			t.Errorf("Actual result %q does not match expected result %q for input: %q", result, output, input)
		}
	}
}
