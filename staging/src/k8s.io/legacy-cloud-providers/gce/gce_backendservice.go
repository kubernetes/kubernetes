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
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

func newBackendServiceMetricContext(request, region string) *metricContext {
	return newBackendServiceMetricContextWithVersion(request, region, computeV1Version)
}

func newBackendServiceMetricContextWithVersion(request, region, version string) *metricContext {
	return newGenericMetricContext("backendservice", request, region, unusedMetricLabel, version)
}

// GetGlobalBackendService retrieves a backend by name.
func (g *Cloud) GetGlobalBackendService(name string) (*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get", "")
	v, err := g.c.BackendServices().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// GetBetaGlobalBackendService retrieves beta backend by name.
func (g *Cloud) GetBetaGlobalBackendService(name string) (*computebeta.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("get", "", computeBetaVersion)
	v, err := g.c.BetaBackendServices().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// GetAlphaGlobalBackendService retrieves alpha backend by name.
func (g *Cloud) GetAlphaGlobalBackendService(name string) (*computealpha.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("get", "", computeAlphaVersion)
	v, err := g.c.AlphaBackendServices().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// UpdateGlobalBackendService applies the given BackendService as an update to
// an existing service.
func (g *Cloud) UpdateGlobalBackendService(bg *compute.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("update", "")
	return mc.Observe(g.c.BackendServices().Update(ctx, meta.GlobalKey(bg.Name), bg))
}

// UpdateBetaGlobalBackendService applies the given beta BackendService as an
// update to an existing service.
func (g *Cloud) UpdateBetaGlobalBackendService(bg *computebeta.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("update", "", computeBetaVersion)
	return mc.Observe(g.c.BetaBackendServices().Update(ctx, meta.GlobalKey(bg.Name), bg))
}

// UpdateAlphaGlobalBackendService applies the given alpha BackendService as an
// update to an existing service.
func (g *Cloud) UpdateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("update", "", computeAlphaVersion)
	return mc.Observe(g.c.AlphaBackendServices().Update(ctx, meta.GlobalKey(bg.Name), bg))
}

// DeleteGlobalBackendService deletes the given BackendService by name.
func (g *Cloud) DeleteGlobalBackendService(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("delete", "")
	return mc.Observe(g.c.BackendServices().Delete(ctx, meta.GlobalKey(name)))
}

// CreateGlobalBackendService creates the given BackendService.
func (g *Cloud) CreateGlobalBackendService(bg *compute.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("create", "")
	return mc.Observe(g.c.BackendServices().Insert(ctx, meta.GlobalKey(bg.Name), bg))
}

// CreateBetaGlobalBackendService creates the given beta BackendService.
func (g *Cloud) CreateBetaGlobalBackendService(bg *computebeta.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("create", "", computeBetaVersion)
	return mc.Observe(g.c.BetaBackendServices().Insert(ctx, meta.GlobalKey(bg.Name), bg))
}

// CreateAlphaGlobalBackendService creates the given alpha BackendService.
func (g *Cloud) CreateAlphaGlobalBackendService(bg *computealpha.BackendService) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("create", "", computeAlphaVersion)
	return mc.Observe(g.c.AlphaBackendServices().Insert(ctx, meta.GlobalKey(bg.Name), bg))
}

// ListGlobalBackendServices lists all backend services in the project.
func (g *Cloud) ListGlobalBackendServices() ([]*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("list", "")
	v, err := g.c.BackendServices().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// GetGlobalBackendServiceHealth returns the health of the BackendService
// identified by the given name, in the given instanceGroup. The
// instanceGroupLink is the fully qualified self link of an instance group.
func (g *Cloud) GetGlobalBackendServiceHealth(name string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get_health", "")
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := g.c.BackendServices().GetHealth(ctx, meta.GlobalKey(name), groupRef)
	return v, mc.Observe(err)
}

// GetRegionBackendService retrieves a backend by name.
func (g *Cloud) GetRegionBackendService(name, region string) (*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get", region)
	v, err := g.c.RegionBackendServices().Get(ctx, meta.RegionalKey(name, region))
	return v, mc.Observe(err)
}

// UpdateRegionBackendService applies the given BackendService as an update to
// an existing service.
func (g *Cloud) UpdateRegionBackendService(bg *compute.BackendService, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("update", region)
	return mc.Observe(g.c.RegionBackendServices().Update(ctx, meta.RegionalKey(bg.Name, region), bg))
}

// DeleteRegionBackendService deletes the given BackendService by name.
func (g *Cloud) DeleteRegionBackendService(name, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("delete", region)
	return mc.Observe(g.c.RegionBackendServices().Delete(ctx, meta.RegionalKey(name, region)))
}

// CreateRegionBackendService creates the given BackendService.
func (g *Cloud) CreateRegionBackendService(bg *compute.BackendService, region string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("create", region)
	return mc.Observe(g.c.RegionBackendServices().Insert(ctx, meta.RegionalKey(bg.Name, region), bg))
}

// ListRegionBackendServices lists all backend services in the project.
func (g *Cloud) ListRegionBackendServices(region string) ([]*compute.BackendService, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("list", region)
	v, err := g.c.RegionBackendServices().List(ctx, region, filter.None)
	return v, mc.Observe(err)
}

// GetRegionalBackendServiceHealth returns the health of the BackendService
// identified by the given name, in the given instanceGroup. The
// instanceGroupLink is the fully qualified self link of an instance group.
func (g *Cloud) GetRegionalBackendServiceHealth(name, region string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContext("get_health", region)
	ref := &compute.ResourceGroupReference{Group: instanceGroupLink}
	v, err := g.c.RegionBackendServices().GetHealth(ctx, meta.RegionalKey(name, region), ref)
	return v, mc.Observe(err)
}

// SetSecurityPolicyForBetaGlobalBackendService sets the given
// SecurityPolicyReference for the BackendService identified by the given name.
func (g *Cloud) SetSecurityPolicyForBetaGlobalBackendService(backendServiceName string, securityPolicyReference *computebeta.SecurityPolicyReference) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("set_security_policy", "", computeBetaVersion)
	return mc.Observe(g.c.BetaBackendServices().SetSecurityPolicy(ctx, meta.GlobalKey(backendServiceName), securityPolicyReference))
}

// SetSecurityPolicyForAlphaGlobalBackendService sets the given
// SecurityPolicyReference for the BackendService identified by the given name.
func (g *Cloud) SetSecurityPolicyForAlphaGlobalBackendService(backendServiceName string, securityPolicyReference *computealpha.SecurityPolicyReference) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newBackendServiceMetricContextWithVersion("set_security_policy", "", computeAlphaVersion)
	return mc.Observe(g.c.AlphaBackendServices().SetSecurityPolicy(ctx, meta.GlobalKey(backendServiceName), securityPolicyReference))
}
