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

	"github.com/golang/glog"
)

func newInstanceGroupMetricContext(request string, zone string) *metricContext {
	return newGenericMetricContext("instancegroup", request, unusedMetricLabel, zone, computeV1Version)
}

// CreateInstanceGroup creates an instance group with the given
// instances. It is the callers responsibility to add named ports.
func (gce *GCECloud) CreateInstanceGroup(ig *compute.InstanceGroup, zone string) error {
	mc := newInstanceGroupMetricContext("create", zone)
	glog.V(4).Infof("InstanceGroups.Insert(%s, %s, %v): start", gce.projectID, zone, ig)
	op, err := gce.service.InstanceGroups.Insert(gce.projectID, zone, ig).Do()
	glog.V(4).Infof("InstanceGroups.Insert(%s, %s, %v): end", gce.projectID, zone, ig)
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// DeleteInstanceGroup deletes an instance group.
func (gce *GCECloud) DeleteInstanceGroup(name string, zone string) error {
	mc := newInstanceGroupMetricContext("delete", zone)
	glog.V(4).Infof("InstanceGroups.Delete(%s, %s, %s): start", gce.projectID, zone, name)
	op, err := gce.service.InstanceGroups.Delete(
		gce.projectID, zone, name).Do()
	glog.V(4).Infof("InstanceGroups.Delete(%s, %s, %s): end", gce.projectID, zone, name)
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
	glog.V(4).Infof("InstanceGroups.List(%s, %s): start", gce.projectID, zone)
	v, err := gce.service.InstanceGroups.List(gce.projectID, zone).Do()
	glog.V(4).Infof("InstanceGroups.List(%s, %s): end", gce.projectID, zone)
	return v, mc.Observe(err)
}

// ListInstancesInInstanceGroup lists all the instances in a given
// instance group and state.
func (gce *GCECloud) ListInstancesInInstanceGroup(name string, zone string, state string) (*compute.InstanceGroupsListInstances, error) {
	mc := newInstanceGroupMetricContext("list_instances", zone)
	obj := compute.InstanceGroupsListInstancesRequest{InstanceState: state}
	// TODO: use PageToken to list all not just the first 500
	glog.V(4).Infof("InstanceGroups.ListInstances(%s, %s, %s, %v): start", gce.projectID, zone, name, obj)
	v, err := gce.service.InstanceGroups.ListInstances(
		gce.projectID, zone, name, &obj).Do()
	glog.V(4).Infof("InstanceGroups.ListInstances(%s, %s, %s, %v): end", gce.projectID, zone, name, obj)
	return v, mc.Observe(err)
}

// AddInstancesToInstanceGroup adds the given instances to the given
// instance group.
func (gce *GCECloud) AddInstancesToInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	mc := newInstanceGroupMetricContext("add_instances", zone)
	obj := compute.InstanceGroupsAddInstancesRequest{
		Instances: instanceRefs,
	}
	if len(instanceRefs) == 0 {
		return nil
	}

	glog.V(4).Infof("InstanceGroups.AddInstances(%s, %s, %s, %v): start", gce.projectID, zone, name, obj)
	op, err := gce.service.InstanceGroups.AddInstances(
		gce.projectID, zone, name, &obj).Do()
	glog.V(4).Infof("InstanceGroups.AddInstances(%s, %s, %s, %v): end", gce.projectID, zone, name, obj)
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// RemoveInstancesFromInstanceGroup removes the given instances from
// the instance group.
func (gce *GCECloud) RemoveInstancesFromInstanceGroup(name string, zone string, instanceRefs []*compute.InstanceReference) error {
	mc := newInstanceGroupMetricContext("remove_instances", zone)
	obj := compute.InstanceGroupsRemoveInstancesRequest{
		Instances: instanceRefs,
	}
	if len(instanceRefs) == 0 {
		return nil
	}

	glog.V(4).Infof("InstanceGroups.RemoveInstances(%s, %s, %s, %v): start", gce.projectID, zone, name, obj)
	op, err := gce.service.InstanceGroups.RemoveInstances(
		gce.projectID, zone, name, &obj).Do()
	glog.V(4).Infof("InstanceGroups.RemoveInstances(%s, %s, %s, %v): end", gce.projectID, zone, name, obj)
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// SetNamedPortsOfInstanceGroup sets the list of named ports on a given instance group
func (gce *GCECloud) SetNamedPortsOfInstanceGroup(igName, zone string, namedPorts []*compute.NamedPort) error {
	mc := newInstanceGroupMetricContext("set_namedports", zone)
	obj := compute.InstanceGroupsSetNamedPortsRequest{NamedPorts: namedPorts}
	glog.V(4).Infof("InstanceGroups.SetNamedPorts(%s, %s, %s, %v): start", gce.projectID, zone, igName, obj)
	op, err := gce.service.InstanceGroups.SetNamedPorts(
		gce.projectID, zone, igName, &obj).Do()
	glog.V(4).Infof("InstanceGroups.SetNamedPorts(%s, %s, %s, %v): end", gce.projectID, zone, igName, obj)
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// GetInstanceGroup returns an instance group by name.
func (gce *GCECloud) GetInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	mc := newInstanceGroupMetricContext("get", zone)
	glog.V(4).Infof("InstanceGroups.Get(%s, %s, %s): start", gce.projectID, zone, name)
	v, err := gce.service.InstanceGroups.Get(gce.projectID, zone, name).Do()
	glog.V(4).Infof("InstanceGroups.Get(%s, %s, %s): end", gce.projectID, zone, name)
	return v, mc.Observe(err)
}
