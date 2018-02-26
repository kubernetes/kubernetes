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

This file contains shared functions and variables to set up for tests for
ExternalLoadBalancer and InternalLoadBalancers. It currently cannot live in a
separate package from GCE because then it would cause a circular import.
*/

package gce

import (
	"fmt"
	"net/http"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/mock"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

var fakeApiService = &v1.Service{
	Spec: v1.ServiceSpec{
		SessionAffinity: v1.ServiceAffinityClientIP,
		Type:            v1.ServiceTypeClusterIP,
		Ports:           []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: int32(123)}},
	},
}

type fakeRoundTripper struct{}

func (*fakeRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("err: test used fake http client")
}

func fakeGCECloud(projectID, region, zoneName string) (*GCECloud, error) {
	client := &http.Client{Transport: &fakeRoundTripper{}}

	service, err := compute.New(client)
	if err != nil {
		return nil, err
	}

	// Used in disk unit tests
	fakeManager := newFakeManager(projectID, region)
	zonesWithNodes := createNodeZones([]string{zoneName})

	alphaFeatureGate, err := NewAlphaFeatureGate([]string{})
	if err != nil {
		return nil, err
	}

	gce := &GCECloud{
		region:             region,
		service:            service,
		manager:            fakeManager,
		managedZones:       []string{zoneName},
		projectID:          projectID,
		networkProjectID:   projectID,
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

	keyGA := meta.GlobalKey("key-ga")
	c.MockZones.Objects[*keyGA] = &cloud.MockZonesObj{
		Obj: &compute.Zone{Name: zoneName, Region: gce.getRegionLink(region)},
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
