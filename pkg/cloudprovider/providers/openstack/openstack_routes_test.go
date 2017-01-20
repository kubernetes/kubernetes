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

package openstack

import (
	"net"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestRoutes(t *testing.T) {
	const clusterName = "ignored"

	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	os, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}

	r, ok := os.Routes()
	if !ok {
		t.Fatalf("Routes() returned false - perhaps your stack doens't support Neutron?")
	}

	newroute := cloudprovider.Route{
		DestinationCIDR: "10.164.2.0/24",
		TargetNode:      types.NodeName("testinstance"),
	}
	err = r.CreateRoute(clusterName, "myhint", &newroute)
	if err != nil {
		t.Fatalf("CreateRoute error: %v", err)
	}

	routelist, err := r.ListRoutes(clusterName)
	if err != nil {
		t.Fatalf("ListRoutes() error: %v", err)
	}
	for _, route := range routelist {
		_, cidr, err := net.ParseCIDR(route.DestinationCIDR)
		if err != nil {
			t.Logf("Ignoring route %s, unparsable CIDR: %v", route.Name, err)
			continue
		}
		t.Logf("%s via %s", cidr, route.TargetNode)
	}

	err = r.DeleteRoute(clusterName, &newroute)
	if err != nil {
		t.Fatalf("DeleteRoute error: %v", err)
	}
}
