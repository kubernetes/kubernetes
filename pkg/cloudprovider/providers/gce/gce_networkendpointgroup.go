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

package gce

import (
	"fmt"
	"strings"

	computealpha "google.golang.org/api/compute/v0.alpha"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

const (
	NEGLoadBalancerType          = "LOAD_BALANCING"
	NEGIPPortNetworkEndpointType = "GCE_VM_IP_PORT"
)

func newNetworkEndpointGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("networkendpointgroup_", request, unusedMetricLabel, zone, computeAlphaVersion)
}

func (gce *GCECloud) GetNetworkEndpointGroup(name string, zone string) (*computealpha.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("get", zone)
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, mc.Observe(err)
	}
	v, err := gce.c.AlphaNetworkEndpointGroups().Get(ctx, meta.ZonalKey(name, zone))
	return v, mc.Observe(err)
}

func (gce *GCECloud) ListNetworkEndpointGroup(zone string) ([]*computealpha.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("list", zone)
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, mc.Observe(err)
	}
	negs, err := gce.c.AlphaNetworkEndpointGroups().List(ctx, zone, filter.None)
	return negs, mc.Observe(err)
}

// AggregatedListNetworkEndpointGroup returns a map of zone -> endpoint group.
func (gce *GCECloud) AggregatedListNetworkEndpointGroup() (map[string][]*computealpha.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("aggregated_list", "")
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, mc.Observe(err)
	}
	// TODO: filter for the region the cluster is in.
	all, err := gce.c.AlphaNetworkEndpointGroups().AggregatedList(ctx, filter.None)
	if err != nil {
		return nil, mc.Observe(err)
	}
	ret := map[string][]*computealpha.NetworkEndpointGroup{}
	for key, byZone := range all {
		// key is "zones/<zone name>"
		parts := strings.Split(key, "/")
		if len(parts) != 2 {
			return nil, mc.Observe(fmt.Errorf("invalid key for AggregatedListNetworkEndpointGroup: %q", key))
		}
		zone := parts[1]
		ret[zone] = append(ret[zone], byZone...)
	}
	return ret, mc.Observe(nil)
}

func (gce *GCECloud) CreateNetworkEndpointGroup(neg *computealpha.NetworkEndpointGroup, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("create", zone)
	return mc.Observe(gce.c.AlphaNetworkEndpointGroups().Insert(ctx, meta.ZonalKey(neg.Name, zone), neg))
}

func (gce *GCECloud) DeleteNetworkEndpointGroup(name string, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("delete", zone)
	return mc.Observe(gce.c.AlphaNetworkEndpointGroups().Delete(ctx, meta.ZonalKey(name, zone)))
}

func (gce *GCECloud) AttachNetworkEndpoints(name, zone string, endpoints []*computealpha.NetworkEndpoint) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("attach", zone)
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return mc.Observe(err)
	}
	req := &computealpha.NetworkEndpointGroupsAttachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}
	return mc.Observe(gce.c.AlphaNetworkEndpointGroups().AttachNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req))
}

func (gce *GCECloud) DetachNetworkEndpoints(name, zone string, endpoints []*computealpha.NetworkEndpoint) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("detach", zone)
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return mc.Observe(err)
	}
	req := &computealpha.NetworkEndpointGroupsDetachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}
	return mc.Observe(gce.c.AlphaNetworkEndpointGroups().DetachNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req))
}

func (gce *GCECloud) ListNetworkEndpoints(name, zone string, showHealthStatus bool) ([]*computealpha.NetworkEndpointWithHealthStatus, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("list_networkendpoints", zone)
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, mc.Observe(err)
	}
	healthStatus := "SKIP"
	if showHealthStatus {
		healthStatus = "SHOW"
	}
	req := &computealpha.NetworkEndpointGroupsListEndpointsRequest{
		HealthStatus: healthStatus,
	}
	l, err := gce.c.AlphaNetworkEndpointGroups().ListNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req, filter.None)
	return l, mc.Observe(err)
}
