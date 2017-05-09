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
	"flag"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/master/ports"
	utilversion "k8s.io/kubernetes/pkg/util/version"

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
)

const (
	minNodesHealthCheckVersion = "1.7.0"
)

var (
	lbNodesHealthCheckPortFlag int
)

func init() {
	flag.IntVar(&lbNodesHealthCheckPortFlag, "cloud-provider-gce-lb-nodes-healthcheck-port", ports.ProxyHealthzPort, "Port opened on nodes for LB health checks")
}

func newHealthcheckMetricContext(request string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"healthcheck_" + request, unusedMetricLabel, unusedMetricLabel},
	}
}

// GetHttpHealthCheck returns the given HttpHealthCheck by name.
func (gce *GCECloud) GetHttpHealthCheck(name string) (*compute.HttpHealthCheck, error) {
	mc := newHealthcheckMetricContext("get_legacy")
	v, err := gce.service.HttpHealthChecks.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// UpdateHttpHealthCheck applies the given HttpHealthCheck as an update.
func (gce *GCECloud) UpdateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	mc := newHealthcheckMetricContext("update_legacy")
	op, err := gce.service.HttpHealthChecks.Update(gce.projectID, hc.Name, hc).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// DeleteHttpHealthCheck deletes the given HttpHealthCheck by name.
func (gce *GCECloud) DeleteHttpHealthCheck(name string) error {
	mc := newHealthcheckMetricContext("delete_legacy")
	op, err := gce.service.HttpHealthChecks.Delete(gce.projectID, name).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// CreateHttpHealthCheck creates the given HttpHealthCheck.
func (gce *GCECloud) CreateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	mc := newHealthcheckMetricContext("create_legacy")
	op, err := gce.service.HttpHealthChecks.Insert(gce.projectID, hc).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// ListHttpHealthChecks lists all HttpHealthChecks in the project.
func (gce *GCECloud) ListHttpHealthChecks() (*compute.HttpHealthCheckList, error) {
	mc := newHealthcheckMetricContext("list_legacy")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.HttpHealthChecks.List(gce.projectID).Do()
	return v, mc.Observe(err)
}

// Legacy HTTPS Health Checks

// GetHttpsHealthCheck returns the given HttpsHealthCheck by name.
func (gce *GCECloud) GetHttpsHealthCheck(name string) (*compute.HttpsHealthCheck, error) {
	mc := newHealthcheckMetricContext("get_legacy")
	v, err := gce.service.HttpsHealthChecks.Get(gce.projectID, name).Do()
	mc.Observe(err)
	return v, err
}

// UpdateHttpsHealthCheck applies the given HttpsHealthCheck as an update.
func (gce *GCECloud) UpdateHttpsHealthCheck(hc *compute.HttpsHealthCheck) error {
	mc := newHealthcheckMetricContext("update_legacy")
	op, err := gce.service.HttpsHealthChecks.Update(gce.projectID, hc.Name, hc).Do()
	if err != nil {
		mc.Observe(err)
		return err
	}

	return gce.waitForGlobalOp(op, mc)
}

// DeleteHttpsHealthCheck deletes the given HttpsHealthCheck by name.
func (gce *GCECloud) DeleteHttpsHealthCheck(name string) error {
	mc := newHealthcheckMetricContext("delete_legacy")
	op, err := gce.service.HttpsHealthChecks.Delete(gce.projectID, name).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// CreateHttpsHealthCheck creates the given HttpsHealthCheck.
func (gce *GCECloud) CreateHttpsHealthCheck(hc *compute.HttpsHealthCheck) error {
	mc := newHealthcheckMetricContext("create_legacy")
	op, err := gce.service.HttpsHealthChecks.Insert(gce.projectID, hc).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// ListHttpsHealthChecks lists all HttpsHealthChecks in the project.
func (gce *GCECloud) ListHttpsHealthChecks() (*compute.HttpsHealthCheckList, error) {
	mc := newHealthcheckMetricContext("list_legacy")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.HttpsHealthChecks.List(gce.projectID).Do()
	return v, mc.Observe(err)
}

// Generic HealthCheck

// GetHealthCheck returns the given HealthCheck by name.
func (gce *GCECloud) GetHealthCheck(name string) (*compute.HealthCheck, error) {
	mc := newHealthcheckMetricContext("get")
	v, err := gce.service.HealthChecks.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// UpdateHealthCheck applies the given HealthCheck as an update.
func (gce *GCECloud) UpdateHealthCheck(hc *compute.HealthCheck) error {
	mc := newHealthcheckMetricContext("update")
	op, err := gce.service.HealthChecks.Update(gce.projectID, hc.Name, hc).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// DeleteHealthCheck deletes the given HealthCheck by name.
func (gce *GCECloud) DeleteHealthCheck(name string) error {
	mc := newHealthcheckMetricContext("delete")
	op, err := gce.service.HealthChecks.Delete(gce.projectID, name).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// CreateHealthCheck creates the given HealthCheck.
func (gce *GCECloud) CreateHealthCheck(hc *compute.HealthCheck) error {
	mc := newHealthcheckMetricContext("create")
	op, err := gce.service.HealthChecks.Insert(gce.projectID, hc).Do()
	if err != nil {
		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// ListHealthChecks lists all HealthCheck in the project.
func (gce *GCECloud) ListHealthChecks() (*compute.HealthCheckList, error) {
	mc := newHealthcheckMetricContext("list")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.HealthChecks.List(gce.projectID).Do()
	return v, mc.Observe(err)
}

// getNodesHealthCheckPort returns the health check port used by the GCE load
// balancers (l4) for performing health checks on nodes.
func getNodesHealthCheckPort() int32 {
	return int32(lbNodesHealthCheckPortFlag)
}

// getNodesHealthCheckPath returns the health check path used by the GCE load
// balancers (l4) for performing health checks on nodes.
func getNodesHealthCheckPath() string {
	return "/healthz"
}

// makeNodesHealthCheckName returns name of the health check resource used by
// the GCE load balancers (l4) for performing health checks on nodes.
func makeNodesHealthCheckName(uid string) string {
	return fmt.Sprintf("k8s-l4-xlb-nodes-healthcheck-%s", uid)
}

// isAtLeastMinNodesHealthCheckVersion checks if a version is higher than
// `minNodesHealthCheckVersion`.
func isAtLeastMinNodesHealthCheckVersion(vstring string) bool {
	minVersion, err := utilversion.ParseGeneric(minNodesHealthCheckVersion)
	if err != nil {
		glog.Errorf("MinNodesHealthCheckVersion (%s) is not a valid version string: %v", minNodesHealthCheckVersion, err)
		return false
	}
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return false
	}
	return version.AtLeast(minVersion)
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

// makeNodesHealthCheckFirewallNameSuffix returns suffix of the firewall name
// used by the GCE load balancers (l4) for performing health checks on nodes.
func makeNodesHealthCheckFirewallNameSuffix(uid string) string {
	return fmt.Sprintf("nodes-healthcheck-%s", uid)
}
