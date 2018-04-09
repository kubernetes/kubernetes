/*
Copyright 2017 The Kubernetes Authors.

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

// This file contains shared functions and variables to set up for tests for
// ExternalLoadBalancer and InternalLoadBalancers. It currently cannot live in a
// separate package from GCE because then it would cause a circular import.

package gce

import (
	"fmt"
	"net/http"
	"os"
	"sync"
	"testing"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/mock"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

// TODO(yankaiz): Create shared error types for both test/non-test codes.
const (
	eventReasonManualChange = "LoadBalancerManualChange"
	eventMsgFirewallChange  = "Firewall change required by network admin"
	errPrefixGetTargetPool  = "error getting load balancer's target pool:"
	errStrLbNoHosts         = "Cannot EnsureLoadBalancer() with no hosts"
	wrongTier               = "SupremeLuxury"
	errStrUnsupportedTier   = "unsupported network tier: \"" + wrongTier + "\""
)

type TestClusterValues struct {
	ProjectID   string
	Region      string
	ZoneName    string
	ClusterID   string
	ClusterName string
}

func DefaultTestClusterValues() TestClusterValues {
	return TestClusterValues{
		ProjectID:   "test-project",
		Region:      "us-central1",
		ZoneName:    "us-central1-b",
		ClusterID:   "test-cluster-id",
		ClusterName: "Test Cluster Name",
	}
}

var fakeApiService *v1.Service

type fakeRoundTripper struct{}

func (*fakeRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("err: test used fake http client")
}

func fakeGCECloud(vals TestClusterValues) (*GCECloud, error) {
	client := &http.Client{Transport: &fakeRoundTripper{}}

	service, err := compute.New(client)
	if err != nil {
		return nil, err
	}

	// Used in disk unit tests
	fakeManager := newFakeManager(vals.ProjectID, vals.Region)
	zonesWithNodes := createNodeZones([]string{vals.ZoneName})

	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	if err != nil {
		return nil, err
	}

	gce := &GCECloud{
		region:             vals.Region,
		service:            service,
		manager:            fakeManager,
		managedZones:       []string{vals.ZoneName},
		projectID:          vals.ProjectID,
		networkProjectID:   vals.ProjectID,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          zonesWithNodes,
		nodeInformerSynced: func() bool { return true },
	}

	c := cloud.NewMockGCE(&gceProjectRouter{gce})
	c.MockTargetPools.AddInstanceHook = mock.AddInstanceHook
	c.MockTargetPools.RemoveInstanceHook = mock.RemoveInstanceHook
	c.MockForwardingRules.InsertHook = mock.InsertFwdRuleHook
	c.MockAddresses.InsertHook = mock.InsertAddressHook
	c.MockAlphaAddresses.InsertHook = mock.InsertAlphaAddressHook

	c.MockInstanceGroups.X = mock.InstanceGroupAttributes{
		InstanceMap: make(map[meta.Key]map[string]*compute.InstanceWithNamedPorts),
		Lock:        &sync.Mutex{},
	}
	c.MockInstanceGroups.AddInstancesHook = mock.AddInstancesHook
	c.MockInstanceGroups.RemoveInstancesHook = mock.RemoveInstancesHook
	c.MockInstanceGroups.ListInstancesHook = mock.ListInstancesHook

	c.MockRegionBackendServices.UpdateHook = mock.UpdateRegionBackendServiceHook
	c.MockHealthChecks.UpdateHook = mock.UpdateHealthCheckHook
	c.MockFirewalls.UpdateHook = mock.UpdateFirewallHook

	keyGA := meta.GlobalKey("key-ga")
	c.MockZones.Objects[*keyGA] = &cloud.MockZonesObj{
		Obj: &compute.Zone{Name: vals.ZoneName, Region: gce.getRegionLink(vals.Region)},
	}

	gce.c = c

	return gce, nil
}

func createAndInsertNodes(gce *GCECloud, nodeNames []string, zoneName string) ([]*v1.Node, error) {
	nodes := []*v1.Node{}

	for _, name := range nodeNames {
		// Inserting the same node name twice causes an error - here we check if
		// the instance exists already before insertion.
		// TestUpdateExternalLoadBalancer inserts a new node, and relies on an older
		// node to already have been inserted.
		instance, _ := gce.getInstanceByName(name)

		if instance == nil {
			err := gce.InsertInstance(
				gce.ProjectID(),
				zoneName,
				&compute.Instance{
					Name: name,
					Tags: &compute.Tags{
						Items: []string{name},
					},
				},
			)
			if err != nil {
				return nodes, err
			}
		}

		nodes = append(
			nodes,
			&v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						kubeletapis.LabelHostname:          name,
						kubeletapis.LabelZoneFailureDomain: zoneName,
					},
				},
				Status: v1.NodeStatus{
					NodeInfo: v1.NodeSystemInfo{
						KubeProxyVersion: "v1.7.2",
					},
				},
			},
		)

	}

	return nodes, nil
}

func setup() {
	fakeApiService = &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: ""},
		Spec: v1.ServiceSpec{
			SessionAffinity: v1.ServiceAffinityClientIP,
			Type:            v1.ServiceTypeClusterIP,
			Ports:           []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: int32(123)}},
		},
	}
}

func TestMain(m *testing.M) {
	setup()
	os.Exit(m.Run())
}
