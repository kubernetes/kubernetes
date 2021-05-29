// +build !providerless

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

package gce

import (
	"context"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	compute "google.golang.org/api/compute/v1"
	option "google.golang.org/api/option"
	"k8s.io/client-go/tools/cache"
)

// TestClusterValues holds the config values for the fake/test gce cloud object.
type TestClusterValues struct {
	ProjectID         string
	Region            string
	ZoneName          string
	SecondaryZoneName string
	ClusterID         string
	ClusterName       string
	OnXPN             bool
}

// DefaultTestClusterValues Creates a reasonable set of default cluster values
// for generating a new test fake GCE cloud instance.
func DefaultTestClusterValues() TestClusterValues {
	return TestClusterValues{
		ProjectID:         "test-project",
		Region:            "us-central1",
		ZoneName:          "us-central1-b",
		SecondaryZoneName: "us-central1-c",
		ClusterID:         "test-cluster-id",
		ClusterName:       "Test Cluster Name",
	}
}

// Stubs ClusterID so that ClusterID.getOrInitialize() does not require calling
// gce.Initialize()
func fakeClusterID(clusterID string) ClusterID {
	return ClusterID{
		clusterID: &clusterID,
		store: cache.NewStore(func(obj interface{}) (string, error) {
			return "", nil
		}),
	}
}

// NewFakeGCECloud constructs a fake GCE Cloud from the cluster values.
func NewFakeGCECloud(vals TestClusterValues) *Cloud {
	service, err := compute.NewService(context.Background(), option.WithoutAuthentication())
	if err != nil {
		panic(err)
	}
	gce := &Cloud{
		region:           vals.Region,
		service:          service,
		managedZones:     []string{vals.ZoneName},
		projectID:        vals.ProjectID,
		networkProjectID: vals.ProjectID,
		ClusterID:        fakeClusterID(vals.ClusterID),
		onXPN:            vals.OnXPN,
		metricsCollector: newLoadBalancerMetrics(),
	}
	c := cloud.NewMockGCE(&gceProjectRouter{gce})
	gce.c = c
	return gce
}

// UpdateFakeGCECloud updates the fake GCE cloud with the specified values. Currently only the onXPN value is updated.
func UpdateFakeGCECloud(g *Cloud, vals TestClusterValues) {
	g.onXPN = vals.OnXPN
}
