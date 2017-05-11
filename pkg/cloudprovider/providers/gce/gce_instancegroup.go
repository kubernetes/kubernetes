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
	"net/http"
	"strings"
	"time"

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
)

func newInstanceGroupMetricContext(request string, zone string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"instancegroup_" + request, unusedMetricLabel, zone},
	}
}

// CreateInstanceGroup creates an instance group with the given
// instances. It is the callers responsibility to add named ports.
func (gce *GCECloud) CreateInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	mc := newInstanceGroupMetricContext("create", zone)

	op, err := gce.service.InstanceGroups.Insert(
		gce.projectID, zone, &compute.InstanceGroup{Name: name}).Do()
	if err != nil {
		mc.Observe(err)
		return nil, err
	}

	if err = gce.waitForZoneOp(op, zone, mc); err != nil {
		return nil, err
	}

	return gce.GetInstanceGroup(name, zone)
}

// DeleteInstanceGroup deletes an instance group.
func (gce *GCECloud) DeleteInstanceGroup(name string, zone string) error {
	mc := newInstanceGroupMetricContext("delete", zone)

	op, err := gce.service.InstanceGroups.Delete(
		gce.projectID, zone, name).Do()
	if err != nil {
		mc.Observe(err)
		return err
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
func (gce *GCECloud) AddInstancesToInstanceGroup(name string, zone string, instanceNames []string) error {
	mc := newInstanceGroupMetricContext("add_instances", zone)
	if len(instanceNames) == 0 {
		return nil
	}
	// Adding the same instance twice will result in a 4xx error
	instances := []*compute.InstanceReference{}
	for _, ins := range instanceNames {
		instances = append(instances, &compute.InstanceReference{Instance: makeHostURL(gce.projectID, zone, ins)})
	}
	op, err := gce.service.InstanceGroups.AddInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsAddInstancesRequest{
			Instances: instances,
		}).Do()

	if err != nil {
		mc.Observe(err)
		return err
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// RemoveInstancesFromInstanceGroup removes the given instances from
// the instance group.
func (gce *GCECloud) RemoveInstancesFromInstanceGroup(name string, zone string, instanceNames []string) error {
	mc := newInstanceGroupMetricContext("remove_instances", zone)
	if len(instanceNames) == 0 {
		return nil
	}

	instances := []*compute.InstanceReference{}
	for _, ins := range instanceNames {
		instanceLink := makeHostURL(gce.projectID, zone, ins)
		instances = append(instances, &compute.InstanceReference{Instance: instanceLink})
	}
	op, err := gce.service.InstanceGroups.RemoveInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsRemoveInstancesRequest{
			Instances: instances,
		}).Do()

	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			mc.Observe(nil)
			return nil
		}

		mc.Observe(err)
		return err
	}

	return gce.waitForZoneOp(op, zone, mc)
}

// AddPortToInstanceGroup adds a port to the given instance group.
func (gce *GCECloud) AddPortToInstanceGroup(ig *compute.InstanceGroup, port int64) (*compute.NamedPort, error) {
	mc := newInstanceGroupMetricContext("add_port", ig.Zone)
	for _, np := range ig.NamedPorts {
		if np.Port == port {
			glog.V(3).Infof("Instance group %v already has named port %+v", ig.Name, np)
			return np, nil
		}
	}

	glog.Infof("Adding port %v to instance group %v with %d ports", port, ig.Name, len(ig.NamedPorts))
	namedPort := compute.NamedPort{Name: fmt.Sprintf("port%v", port), Port: port}
	ig.NamedPorts = append(ig.NamedPorts, &namedPort)

	// setNamedPorts is a zonal endpoint, meaning we invoke it by re-creating a URL like:
	// {project}/zones/{zone}/instanceGroups/{instanceGroup}/setNamedPorts, so the "zone"
	// parameter given to SetNamedPorts must not be the entire zone URL.
	zoneURLParts := strings.Split(ig.Zone, "/")
	zone := zoneURLParts[len(zoneURLParts)-1]

	op, err := gce.service.InstanceGroups.SetNamedPorts(
		gce.projectID, zone, ig.Name,
		&compute.InstanceGroupsSetNamedPortsRequest{
			NamedPorts: ig.NamedPorts}).Do()

	if err != nil {
		mc.Observe(err)
		return nil, err
	}

	if err = gce.waitForZoneOp(op, zone, mc); err != nil {
		return nil, err
	}

	return &namedPort, nil
}

// GetInstanceGroup returns an instance group by name.
func (gce *GCECloud) GetInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	mc := newInstanceGroupMetricContext("get", zone)
	v, err := gce.service.InstanceGroups.Get(gce.projectID, zone, name).Do()
	return v, mc.Observe(err)
}
