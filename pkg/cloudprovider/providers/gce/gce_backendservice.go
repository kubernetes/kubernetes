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
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newBackendServiceMetricContext(request, region string) *metricContext {
	return newBackendServiceMetricContextWithVersion(request, region, computeV1Version)
}

func newBackendServiceMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("backendservice", request, region, unusedMetricLabel, version)
}

// GetGlobalBackendService retrieves a backend by name.
func (gce *GCECloud) GetGlobalBackendService(name string) (*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get", "")
	v, err := gce.c.BackendServices().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// GetAlphaGlobalBackendService retrieves alpha backend by name.
func (gce *GCECloud) GetAlphaGlobalBackendService(name string) (*computealpha.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("get", "", computeAlphaVersion)
	v, err := gce.c.AlphaBackendServices().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// UpdateGlobalBackendService applies the given BackendService as an update to
// an existing service.
func (gce *GCECloud) UpdateGlobalBackendService(bg *compute.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("update", "")
	return mc.Observe(gce.c.BackendServices().Update(ctx, meta.GlobalKey(bg.Name), bg))
}

// UpdateAlphaGlobalBackendService applies the given alpha BackendService as an
// update to an existing service.
func (gce *GCECloud) UpdateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("update", "")
	return mc.Observe(gce.c.AlphaBackendServices().Update(ctx, meta.GlobalKey(bg.Name), bg))
}

// DeleteGlobalBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteGlobalBackendService(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("delete", "")
	return mc.Observe(gce.c.BackendServices().Delete(ctx, meta.GlobalKey(name)))
}

// CreateGlobalBackendService creates the given BackendService.
func (gce *GCECloud) CreateGlobalBackendService(bg *compute.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("create", "")
	return mc.Observe(gce.c.BackendServices().Insert(ctx, meta.GlobalKey(bg.Name), bg))
}

// CreateAlphaGlobalBackendService creates the given alpha BackendService.
func (gce *GCECloud) CreateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("create", "")
	return mc.Observe(gce.c.AlphaBackendServices().Insert(ctx, meta.GlobalKey(bg.Name), bg))
}

// ListGlobalBackendServices lists all backend services in the project.
func (gce *GCECloud) ListGlobalBackendServices() ([]*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("list", "")
	v, err := gce.c.BackendServices().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// GetGlobalBackendServiceHealth returns the health of the BackendService
// identified by the given name, in the given instanceGroup. The
// instanceGroupLink is the fully qualified self link of an instance group.
func (gce *GCECloud) GetGlobalBackendServiceHealth(name string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get_health", "")
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := gce.c.BackendServices().GetHealth(ctx, meta.GlobalKey(name), groupRef)
	return v, mc.Observe(err)
}

// GetRegionBackendService retrieves a backend by name.
func (gce *GCECloud) GetRegionBackendService(name, region string) (*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get", region)
	v, err := gce.c.RegionBackendServices().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// UpdateRegionBackendService applies the given BackendService as an update to
// an existing service.
func (gce *GCECloud) UpdateRegionBackendService(bg *compute.BackendService, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("update", region)
	return mc.Observe(gce.c.RegionBackendServices().Update(ctx, meta.RegionalKey(bg.Name, region), bg))
}

// DeleteRegionBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteRegionBackendService(name, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("delete", region)
	return mc.Observe(gce.c.RegionBackendServices().Delete(ctx, meta.RegionalKey(name, region)))
}

// CreateRegionBackendService creates the given BackendService.
func (gce *GCECloud) CreateRegionBackendService(bg *compute.BackendService, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("create", region)
	return mc.Observe(gce.c.RegionBackendServices().Insert(ctx, meta.RegionalKey(bg.Name, region), bg))
}

// ListRegionBackendServices lists all backend services in the project.
func (gce *GCECloud) ListRegionBackendServices(region string) ([]*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("list", region)
	v, err := gce.c.RegionBackendServices().List(ctx, region, filter.None)
	return v, mc.Observe(err)
}

// GetRegionalBackendServiceHealth returns the health of the BackendService
// identified by the given name, in the given instanceGroup. The
// instanceGroupLink is the fully qualified self link of an instance group.
func (gce *GCECloud) GetRegionalBackendServiceHealth(name, region string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get_health", region)
	ref := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := gce.c.RegionBackendServices().GetHealth(ctx, meta.RegionalKey(name, region), ref)
	return v, mc.Observe(err)
}
