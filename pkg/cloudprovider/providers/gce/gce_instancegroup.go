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
	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newInstanceGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("instancegroup", request, unusedMetricLabel, zone, computeV1Version)
}

// CreateInstanceGroup creates an instance group with the given
// instances. It is the callers responsibility to add named ports.
func (gce *GCECloud) CreateInstanceGroup(ig *compute.InstanceGroup, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("create", zone)
	return mc.Observe(gce.c.InstanceGroups().Insert(ctx, meta.ZonalKey(ig.Name, zone), ig))
}

// DeleteInstanceGroup deletes an instance group.
func (gce *GCECloud) DeleteInstanceGroup(name string, zone string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("delete", zone)
	return mc.Observe(gce.c.InstanceGroups().Delete(ctx, meta.ZonalKey(name, zone)))
}

// ListInstanceGroups lists all InstanceGroups in the project and
// zone.
func (gce *GCECloud) ListInstanceGroups(zone string) ([]*compute.InstanceGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("list", zone)
	v, err := gce.c.InstanceGroups().List(ctx, zone, filter.None)
	return v, mc.Observe(err)
}

// ListInstancesInInstanceGroup lists all the instances in a given
// instance group and state.
func (gce *GCECloud) ListInstancesInInstanceGroup(name string, zone string, state string) ([]*compute.InstanceWithNamedPorts, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("list_instances", zone)
	req := &compute.InstanceGroupsListInstancesRequest{InstanceState: state}
	v, err := gce.c.InstanceGroups().ListInstances(ctx, meta.ZonalKey(name, zone), req, filter.None)
	return v, mc.Observe(err)
}

// AddInstancesToInstanceGroup adds the given instances to the given
// instance group.
func (gce *GCECloud) AddInstancesToInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("add_instances", zone)
	// TODO: should cull operation above this layer.
	if len(instanceRefs) == 0 {
		return nil
	}
	req := &compute.InstanceGroupsAddInstancesRequest{
		Instances: instanceRefs,
	}
	return mc.Observe(gce.c.InstanceGroups().AddInstances(ctx, meta.ZonalKey(name, zone), req))
}

// RemoveInstancesFromInstanceGroup removes the given instances from
// the instance group.
func (gce *GCECloud) RemoveInstancesFromInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("remove_instances", zone)
	// TODO: should cull operation above this layer.
	if len(instanceRefs) == 0 {
		return nil
	}
	req := &compute.InstanceGroupsRemoveInstancesRequest{
		Instances: instanceRefs,
	}
	return mc.Observe(gce.c.InstanceGroups().RemoveInstances(ctx, meta.ZonalKey(name, zone), req))
}

// SetNamedPortsOfInstanceGroup sets the list of named ports on a given instance group
func (gce *GCECloud) SetNamedPortsOfInstanceGroup(igName, zone string, namedPorts []*compute.NamedPort) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("set_namedports", zone)
	req := &compute.InstanceGroupsSetNamedPortsRequest{NamedPorts: namedPorts}
	return mc.Observe(gce.c.InstanceGroups().SetNamedPorts(ctx, meta.ZonalKey(igName, zone), req))
}

// GetInstanceGroup returns an instance group by name.
func (gce *GCECloud) GetInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstanceGroupMetricContext("get", zone)
	v, err := gce.c.InstanceGroups().Get(ctx, meta.ZonalKey(name, zone))
	return v, mc.Observe(err)
}
