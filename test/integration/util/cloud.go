/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"time"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"golang.org/x/oauth2"
	"k8s.io/legacy-cloud-providers/gce"
)

const (
	// TestProjectID is the project id used for creating NewMockGCECloud
	TestProjectID = "test-project"
	// TestNetworkProjectID is the network project id for creating NewMockGCECloud
	TestNetworkProjectID = "net-test-project"
	// TestRegion is the region for creating NewMockGCECloud
	TestRegion = "test-region"
	// TestZone is the zone for creating NewMockGCECloud
	TestZone = "test-zone"
	// TestNetworkName is the network name for creating NewMockGCECloud
	TestNetworkName = "test-network"
	// TestSubnetworkName is the sub network name for creating NewMockGCECloud
	TestSubnetworkName = "test-sub-network"
	// TestSecondaryRangeName is the secondary range name for creating NewMockGCECloud
	TestSecondaryRangeName = "test-secondary-range"
)

type mockTokenSource struct{}

func (*mockTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{
		AccessToken:  "access",
		TokenType:    "Bearer",
		RefreshToken: "refresh",
		Expiry:       time.Now().Add(1 * time.Hour),
	}, nil
}

// NewMockGCECloud returns a handle to a Cloud instance that is
// served by a mock http server
func NewMockGCECloud(cloud cloud.Cloud) (*gce.Cloud, error) {
	config := &gce.CloudConfig{
		ProjectID:          TestProjectID,
		NetworkProjectID:   TestNetworkProjectID,
		Region:             TestRegion,
		Zone:               TestZone,
		ManagedZones:       []string{TestZone},
		NetworkName:        TestNetworkName,
		SubnetworkName:     TestSubnetworkName,
		SecondaryRangeName: TestSecondaryRangeName,
		NodeTags:           []string{},
		UseMetadataServer:  false,
		TokenSource:        &mockTokenSource{},
	}
	return gce.CreateGCECloudWithCloud(config, cloud)
}
