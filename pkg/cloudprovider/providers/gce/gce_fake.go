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
	"fmt"
	"net/http"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
)

type TestClusterValues struct {
	ProjectID         string
	Region            string
	ZoneName          string
	SecondaryZoneName string
	ClusterID         string
	ClusterName       string
}

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

func FakeGCECloud(vals TestClusterValues) *GCECloud {
	return simpleFakeGCECloud(vals)
}

type fakeRoundTripper struct{}

func (*fakeRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("err: test used fake http client")
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

func simpleFakeGCECloud(vals TestClusterValues) *GCECloud {
	client := &http.Client{Transport: &fakeRoundTripper{}}
	service, _ := compute.New(client)
	gce := &GCECloud{
		region:           vals.Region,
		service:          service,
		managedZones:     []string{vals.ZoneName},
		projectID:        vals.ProjectID,
		networkProjectID: vals.ProjectID,
		ClusterID:        fakeClusterID(vals.ClusterID),
	}
	c := cloud.NewMockGCE(&gceProjectRouter{gce})
	gce.c = c
	return gce
}
