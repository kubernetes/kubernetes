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

	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/routers"
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

	// Pick the first router and server to try a test with
	os.routeOpts.RouterId = getRouters(os)[0].ID
	servername := getServers(os)[0].Name

	r, ok := os.Routes()
	if !ok {
		t.Skip("Routes() returned false - perhaps your stack does not support Neutron extraroute extension?")
	}

	newroute := cloudprovider.Route{
		DestinationCIDR: "10.164.2.0/24",
		TargetNode:      types.NodeName(servername),
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

func getServers(os *OpenStack) []servers.Server {
	c, err := os.NewComputeV2()
	allPages, err := servers.List(c, servers.ListOpts{}).AllPages()
	if err != nil {
		panic(err)
	}
	allServers, err := servers.ExtractServers(allPages)
	if err != nil {
		panic(err)
	}
	if len(allServers) == 0 {
		panic("No servers to test with")
	}
	return allServers
}

func getRouters(os *OpenStack) []routers.Router {
	listOpts := routers.ListOpts{}
	n, err := os.NewNetworkV2()
	if err != nil {
		panic(err)
	}
	allPages, err := routers.List(n, listOpts).AllPages()
	if err != nil {
		panic(err)
	}
	allRouters, err := routers.ExtractRouters(allPages)
	if err != nil {
		panic(err)
	}
	if len(allRouters) == 0 {
		panic("No routers to test with")
	}
	return allRouters
}
