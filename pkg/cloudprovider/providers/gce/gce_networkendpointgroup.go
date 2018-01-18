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
	"context"
	computealpha "google.golang.org/api/compute/v0.alpha"
	"strings"
)

const (
	NEGLoadBalancerType          = "LOAD_BALANCING"
	NEGIPPortNetworkEndpointType = "GCE_VM_IP_PORT"
)

func newNetworkEndpointGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("networkendpointgroup_", request, unusedMetricLabel, zone, computeAlphaVersion)
}

func (gce *GCECloud) GetNetworkEndpointGroup(name string, zone string) (*computealpha.NetworkEndpointGroup, error) {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, err
	}
	mc := newNetworkEndpointGroupMetricContext("get", zone)
	v, err := gce.serviceAlpha.NetworkEndpointGroups.Get(gce.ProjectID(), zone, name).Do()
	return v, mc.Observe(err)
}

func (gce *GCECloud) ListNetworkEndpointGroup(zone string) ([]*computealpha.NetworkEndpointGroup, error) {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, err
	}
	mc := newNetworkEndpointGroupMetricContext("list", zone)
	networkEndpointGroups := []*computealpha.NetworkEndpointGroup{}
	err := gce.serviceAlpha.NetworkEndpointGroups.List(gce.ProjectID(), zone).Pages(context.Background(), func(res *computealpha.NetworkEndpointGroupList) error {
		networkEndpointGroups = append(networkEndpointGroups, res.Items...)
		return nil
	})
	return networkEndpointGroups, mc.Observe(err)
}

func (gce *GCECloud) AggregatedListNetworkEndpointGroup() (map[string][]*computealpha.NetworkEndpointGroup, error) {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, err
	}
	mc := newNetworkEndpointGroupMetricContext("aggregated_list", "")
	zoneNetworkEndpointGroupMap := map[string][]*computealpha.NetworkEndpointGroup{}
	err := gce.serviceAlpha.NetworkEndpointGroups.AggregatedList(gce.ProjectID()).Pages(context.Background(), func(res *computealpha.NetworkEndpointGroupAggregatedList) error {
		for key, negs := range res.Items {
			if len(negs.NetworkEndpointGroups) == 0 {
				continue
			}
			// key has the format of "zones/${zone_name}"
			zone := strings.Split(key, "/")[1]
			if _, ok := zoneNetworkEndpointGroupMap[zone]; !ok {
				zoneNetworkEndpointGroupMap[zone] = []*computealpha.NetworkEndpointGroup{}
			}
			zoneNetworkEndpointGroupMap[zone] = append(zoneNetworkEndpointGroupMap[zone], negs.NetworkEndpointGroups...)
		}
		return nil
	})
	return zoneNetworkEndpointGroupMap, mc.Observe(err)
}

func (gce *GCECloud) CreateNetworkEndpointGroup(neg *computealpha.NetworkEndpointGroup, zone string) error {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("create", zone)
	op, err := gce.serviceAlpha.NetworkEndpointGroups.Insert(gce.ProjectID(), zone, neg).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForZoneOp(op, zone, mc)
}

func (gce *GCECloud) DeleteNetworkEndpointGroup(name string, zone string) error {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("delete", zone)
	op, err := gce.serviceAlpha.NetworkEndpointGroups.Delete(gce.ProjectID(), zone, name).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForZoneOp(op, zone, mc)
}

func (gce *GCECloud) AttachNetworkEndpoints(name, zone string, endpoints []*computealpha.NetworkEndpoint) error {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("attach", zone)
	op, err := gce.serviceAlpha.NetworkEndpointGroups.AttachNetworkEndpoints(gce.ProjectID(), zone, name, &computealpha.NetworkEndpointGroupsAttachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForZoneOp(op, zone, mc)
}

func (gce *GCECloud) DetachNetworkEndpoints(name, zone string, endpoints []*computealpha.NetworkEndpoint) error {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return err
	}
	mc := newNetworkEndpointGroupMetricContext("detach", zone)
	op, err := gce.serviceAlpha.NetworkEndpointGroups.DetachNetworkEndpoints(gce.ProjectID(), zone, name, &computealpha.NetworkEndpointGroupsDetachEndpointsRequest{
		NetworkEndpoints: endpoints,
	}).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForZoneOp(op, zone, mc)
}

func (gce *GCECloud) ListNetworkEndpoints(name, zone string, showHealthStatus bool) ([]*computealpha.NetworkEndpointWithHealthStatus, error) {
	if err := gce.alphaFeatureEnabled(AlphaFeatureNetworkEndpointGroup); err != nil {
		return nil, err
	}
	healthStatus := "SKIP"
	if showHealthStatus {
		healthStatus = "SHOW"
	}
	mc := newNetworkEndpointGroupMetricContext("list_networkendpoints", zone)
	networkEndpoints := []*computealpha.NetworkEndpointWithHealthStatus{}
	err := gce.serviceAlpha.NetworkEndpointGroups.ListNetworkEndpoints(gce.ProjectID(), zone, name, &computealpha.NetworkEndpointGroupsListEndpointsRequest{
		HealthStatus: healthStatus,
	}).Pages(context.Background(), func(res *computealpha.NetworkEndpointGroupsListNetworkEndpoints) error {
		networkEndpoints = append(networkEndpoints, res.Items...)
		return nil
	})
	return networkEndpoints, mc.Observe(err)
}
