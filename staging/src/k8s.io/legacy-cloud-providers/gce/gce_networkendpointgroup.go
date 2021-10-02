//go:build !providerless
// +build !providerless

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

	computebeta "google.golang.org/api/compute/v0.beta"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

func newNetworkEndpointGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("networkendpointgroup_", request, unusedMetricLabel, zone, computeBetaVersion)
}

// GetNetworkEndpointGroup returns the collection of network endpoints for the name in zone
func (g *Cloud) GetNetworkEndpointGroup(name string, zone string) (*computebeta.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("get", zone)
	v, err := g.c.BetaNetworkEndpointGroups().Get(ctx, meta.ZonalKey(name, zone))
	return v, mc.Observe(err)
}

// ListNetworkEndpointGroup returns the collection of network endpoints for the zone
func (g *Cloud) ListNetworkEndpointGroup(zone string) ([]*computebeta.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("list", zone)
	negs, err := g.c.BetaNetworkEndpointGroups().List(ctx, zone, filter.None)
	return negs, mc.Observe(err)
}

// AggregatedListNetworkEndpointGroup returns a map of zone -> endpoint group.
func (g *Cloud) AggregatedListNetworkEndpointGroup() (map[string][]*computebeta.NetworkEndpointGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("aggregated_list", "")
	// TODO: filter for the region the cluster is in.
	all, err := g.c.BetaNetworkEndpointGroups().AggregatedList(ctx, filter.None)
	if err != nil {
		return nil, mc.Observe(err)
	}
	ret := map[string][]*computebeta.NetworkEndpointGroup{}
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

// CreateNetworkEndpointGroup creates an endpoint group in the zone
func (g *Cloud) CreateNetworkEndpointGroup(neg *computebeta.NetworkEndpointGroup, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("create", zone)
	return mc.Observe(g.c.BetaNetworkEndpointGroups().Insert(ctx, meta.ZonalKey(neg.Name, zone), neg))
}

// DeleteNetworkEndpointGroup deletes the name endpoint group from the zone
func (g *Cloud) DeleteNetworkEndpointGroup(name string, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("delete", zone)
	return mc.Observe(g.c.BetaNetworkEndpointGroups().Delete(ctx, meta.ZonalKey(name, zone)))
}

// AttachNetworkEndpoints associates the referenced endpoints with the named endpoint group in the zone
func (g *Cloud) AttachNetworkEndpoints(name, zone string, endpoints []*computebeta.NetworkEndpoint) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("attach", zone)
	req := &computebeta.NetworkEndpointGroupsAttachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}
	return mc.Observe(g.c.BetaNetworkEndpointGroups().AttachNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req))
}

// DetachNetworkEndpoints breaks the association between the referenced endpoints and the named endpoint group in the zone
func (g *Cloud) DetachNetworkEndpoints(name, zone string, endpoints []*computebeta.NetworkEndpoint) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("detach", zone)
	req := &computebeta.NetworkEndpointGroupsDetachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}
	return mc.Observe(g.c.BetaNetworkEndpointGroups().DetachNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req))
}

// ListNetworkEndpoints returns all the endpoints associated with the endpoint group in zone and optionally their status.
func (g *Cloud) ListNetworkEndpoints(name, zone string, showHealthStatus bool) ([]*computebeta.NetworkEndpointWithHealthStatus, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newNetworkEndpointGroupMetricContext("list_networkendpoints", zone)
	healthStatus := "SKIP"
	if showHealthStatus {
		healthStatus = "SHOW"
	}
	req := &computebeta.NetworkEndpointGroupsListEndpointsRequest{
		HealthStatus: healthStatus,
	}
	l, err := g.c.BetaNetworkEndpointGroups().ListNetworkEndpoints(ctx, meta.ZonalKey(name, zone), req, filter.None)
	return l, mc.Observe(err)
}
