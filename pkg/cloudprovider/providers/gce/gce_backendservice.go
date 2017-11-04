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
	"net/http"

	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
)

func newBackendServiceMetricContext(request, region string) *metricContext {
	return newBackendServiceMetricContextWithVersion(request, region, computeV1Version)
}

func newBackendServiceMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("backendservice", request, region, unusedMetricLabel, version)
}

// GetGlobalBackendService retrieves a backend by name.
func (gce *GCECloud) GetGlobalBackendService(name string) (*compute.BackendService, error) {
	mc := newBackendServiceMetricContext("get", "")
	v, err := gce.service.BackendServices.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// GetAlphaGlobalBackendService retrieves alpha backend by name.
func (gce *GCECloud) GetAlphaGlobalBackendService(name string) (*computealpha.BackendService, error) {
	mc := newBackendServiceMetricContextWithVersion("get", "", computeAlphaVersion)
	v, err := gce.serviceAlpha.BackendServices.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// UpdateGlobalBackendService applies the given BackendService as an update to an existing service.
func (gce *GCECloud) UpdateGlobalBackendService(bg *compute.BackendService) error {
	mc := newBackendServiceMetricContext("update", "")
	op, err := gce.service.BackendServices.Update(gce.projectID, bg.Name, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// UpdateAlphaGlobalBackendService applies the given alpha BackendService as an update to an existing service.
func (gce *GCECloud) UpdateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	mc := newBackendServiceMetricContextWithVersion("update", "", computeAlphaVersion)
	op, err := gce.serviceAlpha.BackendServices.Update(gce.projectID, bg.Name, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// DeleteGlobalBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteGlobalBackendService(name string) error {
	mc := newBackendServiceMetricContext("delete", "")
	op, err := gce.service.BackendServices.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// CreateGlobalBackendService creates the given BackendService.
func (gce *GCECloud) CreateGlobalBackendService(bg *compute.BackendService) error {
	mc := newBackendServiceMetricContext("create", "")
	op, err := gce.service.BackendServices.Insert(gce.projectID, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// CreateAlphaGlobalBackendService creates the given alpha BackendService.
func (gce *GCECloud) CreateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	mc := newBackendServiceMetricContextWithVersion("create", "", computeAlphaVersion)
	op, err := gce.serviceAlpha.BackendServices.Insert(gce.projectID, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// ListGlobalBackendServices lists all backend services in the project.
func (gce *GCECloud) ListGlobalBackendServices() (*compute.BackendServiceList, error) {
	mc := newBackendServiceMetricContext("list", "")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.BackendServices.List(gce.projectID).Do()
	return v, mc.Observe(err)
}

// GetGlobalBackendServiceHealth returns the health of the BackendService identified by the given
// name, in the given instanceGroup. The instanceGroupLink is the fully
// qualified self link of an instance group.
func (gce *GCECloud) GetGlobalBackendServiceHealth(name string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	mc := newBackendServiceMetricContext("get_health", "")
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := gce.service.BackendServices.GetHealth(gce.projectID, name, groupRef).Do()
	return v, mc.Observe(err)
}

// GetRegionBackendService retrieves a backend by name.
func (gce *GCECloud) GetRegionBackendService(name, region string) (*compute.BackendService, error) {
	mc := newBackendServiceMetricContext("get", region)
	v, err := gce.service.RegionBackendServices.Get(gce.projectID, region, name).Do()
	return v, mc.Observe(err)
}

// UpdateRegionBackendService applies the given BackendService as an update to an existing service.
func (gce *GCECloud) UpdateRegionBackendService(bg *compute.BackendService, region string) error {
	mc := newBackendServiceMetricContext("update", region)
	op, err := gce.service.RegionBackendServices.Update(gce.projectID, region, bg.Name, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForRegionOp(op, region, mc)
}

// DeleteRegionBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteRegionBackendService(name, region string) error {
	mc := newBackendServiceMetricContext("delete", region)
	op, err := gce.service.RegionBackendServices.Delete(gce.projectID, region, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return mc.Observe(err)
	}

	return gce.waitForRegionOp(op, region, mc)
}

// CreateRegionBackendService creates the given BackendService.
func (gce *GCECloud) CreateRegionBackendService(bg *compute.BackendService, region string) error {
	mc := newBackendServiceMetricContext("create", region)
	op, err := gce.service.RegionBackendServices.Insert(gce.projectID, region, bg).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForRegionOp(op, region, mc)
}

// ListRegionBackendServices lists all backend services in the project.
func (gce *GCECloud) ListRegionBackendServices(region string) (*compute.BackendServiceList, error) {
	mc := newBackendServiceMetricContext("list", region)
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.RegionBackendServices.List(gce.projectID, region).Do()
	return v, mc.Observe(err)
}

// GetRegionalBackendServiceHealth returns the health of the BackendService identified by the given
// name, in the given instanceGroup. The instanceGroupLink is the fully
// qualified self link of an instance group.
func (gce *GCECloud) GetRegionalBackendServiceHealth(name, region string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	mc := newBackendServiceMetricContext("get_health", region)
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := gce.service.RegionBackendServices.GetHealth(gce.projectID, region, name, groupRef).Do()
	return v, mc.Observe(err)
}
