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
	"k8s.io/klog/v2"

	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
	v1 "k8s.io/api/core/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
)

const (
	nodesHealthCheckPath = "/healthz"
	// NOTE: Please keep the following port in sync with ProxyHealthzPort in pkg/master/ports/ports.go
	// ports.ProxyHealthzPort was not used here to avoid dependencies to k8s.io/kubernetes in the
	// GCE cloud provider which is required as part of the out-of-tree cloud provider efforts.
	// TODO: use a shared constant once ports in pkg/master/ports are in a common external repo.
	lbNodesHealthCheckPort = 10256
)

var (
	minNodesHealthCheckVersion *utilversion.Version
)

func init() {
	if v, err := utilversion.ParseGeneric("1.7.2"); err != nil {
		klog.Fatalf("Failed to parse version for minNodesHealthCheckVersion: %v", err)
	} else {
		minNodesHealthCheckVersion = v
	}
}

func newHealthcheckMetricContext(request string) *metricContext {
	return newHealthcheckMetricContextWithVersion(request, computeV1Version)
}

func newHealthcheckMetricContextWithVersion(request, version string) *metricContext {
	return newGenericMetricContext("healthcheck", request, unusedMetricLabel, unusedMetricLabel, version)
}

// GetHTTPHealthCheck returns the given HttpHealthCheck by name.
func (g *Cloud) GetHTTPHealthCheck(name string) (*compute.HttpHealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("get_legacy")
	v, err := g.c.HttpHealthChecks().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// UpdateHTTPHealthCheck applies the given HttpHealthCheck as an update.
func (g *Cloud) UpdateHTTPHealthCheck(hc *compute.HttpHealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("update_legacy")
	return mc.Observe(g.c.HttpHealthChecks().Update(ctx, meta.GlobalKey(hc.Name), hc))
}

// DeleteHTTPHealthCheck deletes the given HttpHealthCheck by name.
func (g *Cloud) DeleteHTTPHealthCheck(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("delete_legacy")
	return mc.Observe(g.c.HttpHealthChecks().Delete(ctx, meta.GlobalKey(name)))
}

// CreateHTTPHealthCheck creates the given HttpHealthCheck.
func (g *Cloud) CreateHTTPHealthCheck(hc *compute.HttpHealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("create_legacy")
	return mc.Observe(g.c.HttpHealthChecks().Insert(ctx, meta.GlobalKey(hc.Name), hc))
}

// ListHTTPHealthChecks lists all HttpHealthChecks in the project.
func (g *Cloud) ListHTTPHealthChecks() ([]*compute.HttpHealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("list_legacy")
	v, err := g.c.HttpHealthChecks().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// Legacy HTTPS Health Checks

// GetHTTPSHealthCheck returns the given HttpsHealthCheck by name.
func (g *Cloud) GetHTTPSHealthCheck(name string) (*compute.HttpsHealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("get_legacy")
	v, err := g.c.HttpsHealthChecks().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// UpdateHTTPSHealthCheck applies the given HttpsHealthCheck as an update.
func (g *Cloud) UpdateHTTPSHealthCheck(hc *compute.HttpsHealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("update_legacy")
	return mc.Observe(g.c.HttpsHealthChecks().Update(ctx, meta.GlobalKey(hc.Name), hc))
}

// DeleteHTTPSHealthCheck deletes the given HttpsHealthCheck by name.
func (g *Cloud) DeleteHTTPSHealthCheck(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("delete_legacy")
	return mc.Observe(g.c.HttpsHealthChecks().Delete(ctx, meta.GlobalKey(name)))
}

// CreateHTTPSHealthCheck creates the given HttpsHealthCheck.
func (g *Cloud) CreateHTTPSHealthCheck(hc *compute.HttpsHealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("create_legacy")
	return mc.Observe(g.c.HttpsHealthChecks().Insert(ctx, meta.GlobalKey(hc.Name), hc))
}

// ListHTTPSHealthChecks lists all HttpsHealthChecks in the project.
func (g *Cloud) ListHTTPSHealthChecks() ([]*compute.HttpsHealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("list_legacy")
	v, err := g.c.HttpsHealthChecks().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// Generic HealthCheck

// GetHealthCheck returns the given HealthCheck by name.
func (g *Cloud) GetHealthCheck(name string) (*compute.HealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("get")
	v, err := g.c.HealthChecks().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// GetAlphaHealthCheck returns the given alpha HealthCheck by name.
func (g *Cloud) GetAlphaHealthCheck(name string) (*computealpha.HealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("get", computeAlphaVersion)
	v, err := g.c.AlphaHealthChecks().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// GetBetaHealthCheck returns the given beta HealthCheck by name.
func (g *Cloud) GetBetaHealthCheck(name string) (*computebeta.HealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("get", computeBetaVersion)
	v, err := g.c.BetaHealthChecks().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// UpdateHealthCheck applies the given HealthCheck as an update.
func (g *Cloud) UpdateHealthCheck(hc *compute.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("update")
	return mc.Observe(g.c.HealthChecks().Update(ctx, meta.GlobalKey(hc.Name), hc))
}

// UpdateAlphaHealthCheck applies the given alpha HealthCheck as an update.
func (g *Cloud) UpdateAlphaHealthCheck(hc *computealpha.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("update", computeAlphaVersion)
	return mc.Observe(g.c.AlphaHealthChecks().Update(ctx, meta.GlobalKey(hc.Name), hc))
}

// UpdateBetaHealthCheck applies the given beta HealthCheck as an update.
func (g *Cloud) UpdateBetaHealthCheck(hc *computebeta.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("update", computeBetaVersion)
	return mc.Observe(g.c.BetaHealthChecks().Update(ctx, meta.GlobalKey(hc.Name), hc))
}

// DeleteHealthCheck deletes the given HealthCheck by name.
func (g *Cloud) DeleteHealthCheck(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("delete")
	return mc.Observe(g.c.HealthChecks().Delete(ctx, meta.GlobalKey(name)))
}

// CreateHealthCheck creates the given HealthCheck.
func (g *Cloud) CreateHealthCheck(hc *compute.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("create")
	return mc.Observe(g.c.HealthChecks().Insert(ctx, meta.GlobalKey(hc.Name), hc))
}

// CreateAlphaHealthCheck creates the given alpha HealthCheck.
func (g *Cloud) CreateAlphaHealthCheck(hc *computealpha.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("create", computeAlphaVersion)
	return mc.Observe(g.c.AlphaHealthChecks().Insert(ctx, meta.GlobalKey(hc.Name), hc))
}

// CreateBetaHealthCheck creates the given beta HealthCheck.
func (g *Cloud) CreateBetaHealthCheck(hc *computebeta.HealthCheck) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContextWithVersion("create", computeBetaVersion)
	return mc.Observe(g.c.BetaHealthChecks().Insert(ctx, meta.GlobalKey(hc.Name), hc))
}

// ListHealthChecks lists all HealthCheck in the project.
func (g *Cloud) ListHealthChecks() ([]*compute.HealthCheck, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newHealthcheckMetricContext("list")
	v, err := g.c.HealthChecks().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// GetNodesHealthCheckPort returns the health check port used by the GCE load
// balancers (l4) for performing health checks on nodes.
func GetNodesHealthCheckPort() int32 {
	return lbNodesHealthCheckPort
}

// GetNodesHealthCheckPath returns the health check path used by the GCE load
// balancers (l4) for performing health checks on nodes.
func GetNodesHealthCheckPath() string {
	return nodesHealthCheckPath
}

// isAtLeastMinNodesHealthCheckVersion checks if a version is higher than
// `minNodesHealthCheckVersion`.
func isAtLeastMinNodesHealthCheckVersion(vstring string) bool {
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		klog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return false
	}
	return version.AtLeast(minNodesHealthCheckVersion)
}

// supportsNodesHealthCheck returns false if anyone of the nodes has version
// lower than `minNodesHealthCheckVersion`.
func supportsNodesHealthCheck(nodes []*v1.Node) bool {
	for _, node := range nodes {
		if !isAtLeastMinNodesHealthCheckVersion(node.Status.NodeInfo.KubeProxyVersion) {
			return false
		}
	}
	return true
}
