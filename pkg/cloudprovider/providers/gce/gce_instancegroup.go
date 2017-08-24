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

import compute "google.golang.org/api/compute/v1"

func newInstanceGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("instancegroup", request, unusedMetricLabel, zone, computeV1Version)
}

// CreateInstanceGroup creates an instance group with the given
// instances. It is the callers responsibility to add named ports.
func (gce *GCECloud) CreateInstanceGroup(ig *compute.InstanceGroup, zone string) error {
	mc := newInstanceGroupMetricContext("create", zone)
	op, err := gce.service.InstanceGroups.Insert(gce.projectID, zone, ig).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// DeleteInstanceGroup deletes an instance group.
func (gce *GCECloud) DeleteInstanceGroup(name string, zone string) error {
	mc := newInstanceGroupMetricContext("delete", zone)
	op, err := gce.service.InstanceGroups.Delete(
		gce.projectID, zone, name).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// ListInstanceGroups lists all InstanceGroups in the project and
// zone.
func (gce *GCECloud) ListInstanceGroups(zone string) (*compute.InstanceGroupList, error) {
	mc := newInstanceGroupMetricContext("list", zone)
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.InstanceGroups.List(gce.projectID, zone).Do()
	return v, mc.Observe(err)
}

// ListInstancesInInstanceGroup lists all the instances in a given
// instance group and state.
func (gce *GCECloud) ListInstancesInInstanceGroup(name string, zone string, state string) (*compute.InstanceGroupsListInstances, error) {
	mc := newInstanceGroupMetricContext("list_instances", zone)
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.InstanceGroups.ListInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsListInstancesRequest{InstanceState: state}).Do()
	return v, mc.Observe(err)
}

// AddInstancesToInstanceGroup adds the given instances to the given
// instance group.
func (gce *GCECloud) AddInstancesToInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	mc := newInstanceGroupMetricContext("add_instances", zone)
	if len(instanceRefs) == 0 {
		return nil
	}

	op, err := gce.service.InstanceGroups.AddInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsAddInstancesRequest{
			Instances: instanceRefs,
		}).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// RemoveInstancesFromInstanceGroup removes the given instances from
// the instance group.
func (gce *GCECloud) RemoveInstancesFromInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	mc := newInstanceGroupMetricContext("remove_instances", zone)
	if len(instanceRefs) == 0 {
		return nil
	}

	op, err := gce.service.InstanceGroups.RemoveInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsRemoveInstancesRequest{
			Instances: instanceRefs,
		}).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// SetNamedPortsOfInstanceGroup sets the list of named ports on a given instance group
func (gce *GCECloud) SetNamedPortsOfInstanceGroup(igName, zone string, namedPorts []*compute.NamedPort) error {
	mc := newInstanceGroupMetricContext("set_namedports", zone)
	op, err := gce.service.InstanceGroups.SetNamedPorts(
		gce.projectID, zone, igName,
		&compute.InstanceGroupsSetNamedPortsRequest{NamedPorts: namedPorts}).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// GetInstanceGroup returns an instance group by name.
func (gce *GCECloud) GetInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	mc := newInstanceGroupMetricContext("get", zone)
	v, err := gce.service.InstanceGroups.Get(gce.projectID, zone, name).Do()
	return v, mc.Observe(err)
}
