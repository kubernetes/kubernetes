/*
Copyright 2014 The Kubernetes Authors.

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

package skipper

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

func skipInternalf(caller int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ginkgo.Skip(msg, caller+1)
	panic("unreachable")
}

// Skipf skips with information about why the test is being skipped.
// The direct caller is recorded in the callstack.
func Skipf(format string, args ...interface{}) {
	skipInternalf(1, format, args...)
	panic("unreachable")
}

// SkipUnlessAtLeast skips if the value is less than the minValue.
func SkipUnlessAtLeast(value int, minValue int, message string) {
	if value < minValue {
		skipInternalf(1, message)
	}
}

var featureGate featuregate.FeatureGate

// InitFeatureGates must be called in test suites that have a --feature-gates parameter.
// If not called, SkipUnlessFeatureGateEnabled and SkipIfFeatureGateEnabled will
// record a test failure.
func InitFeatureGates(defaults featuregate.FeatureGate, overrides map[string]bool) error {
	clone := defaults.DeepCopy()
	if err := clone.SetFromMap(overrides); err != nil {
		return err
	}
	featureGate = clone
	return nil
}

// SkipUnlessFeatureGateEnabled skips if the feature is disabled.
//
// Beware that this only works in test suites that have a --feature-gate
// parameter and call InitFeatureGates. In test/e2e, the `Feature: XYZ` tag
// has to be used instead and invocations have to make sure that they
// only run tests that work with the given test cluster.
func SkipUnlessFeatureGateEnabled(gate featuregate.Feature) {
	if featureGate == nil {
		framework.Failf("Feature gate checking is not enabled, don't use SkipUnlessFeatureGateEnabled(%v). Instead use the Feature tag.", gate)
	}
	if !featureGate.Enabled(gate) {
		skipInternalf(1, "Only supported when %v feature is enabled", gate)
	}
}

// SkipUnlessNodeCountIsAtLeast skips if the number of nodes is less than the minNodeCount.
func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if framework.TestContext.CloudConfig.NumNodes < minNodeCount {
		skipInternalf(1, "Requires at least %d nodes (not %d)", minNodeCount, framework.TestContext.CloudConfig.NumNodes)
	}
}

// SkipUnlessNodeCountIsAtMost skips if the number of nodes is greater than the maxNodeCount.
func SkipUnlessNodeCountIsAtMost(maxNodeCount int) {
	if framework.TestContext.CloudConfig.NumNodes > maxNodeCount {
		skipInternalf(1, "Requires at most %d nodes (not %d)", maxNodeCount, framework.TestContext.CloudConfig.NumNodes)
	}
}

// SkipIfProviderIs skips if the provider is included in the unsupportedProviders.
func SkipIfProviderIs(unsupportedProviders ...string) {
	if framework.ProviderIs(unsupportedProviders...) {
		skipInternalf(1, "Not supported for providers %v (found %s)", unsupportedProviders, framework.TestContext.Provider)
	}
}

// SkipUnlessProviderIs skips if the provider is not included in the supportedProviders.
func SkipUnlessProviderIs(supportedProviders ...string) {
	if !framework.ProviderIs(supportedProviders...) {
		skipInternalf(1, "Only supported for providers %v (not %s)", supportedProviders, framework.TestContext.Provider)
	}
}

// SkipUnlessMultizone skips if the cluster does not have multizone.
func SkipUnlessMultizone(ctx context.Context, c clientset.Interface) {
	zones, err := e2enode.GetClusterZones(ctx, c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() <= 1 {
		skipInternalf(1, "Requires more than one zone")
	}
}

// SkipUnlessAtLeastNZones skips if the cluster does not have n multizones.
func SkipUnlessAtLeastNZones(ctx context.Context, c clientset.Interface, n int) {
	zones, err := e2enode.GetClusterZones(ctx, c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() < n {
		skipInternalf(1, "Requires >= %d zones", n)
	}
}

// SkipIfMultizone skips if the cluster has multizone.
func SkipIfMultizone(ctx context.Context, c clientset.Interface) {
	zones, err := e2enode.GetClusterZones(ctx, c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() > 1 {
		skipInternalf(1, "Requires at most one zone")
	}
}

// SkipUnlessMasterOSDistroIs skips if the master OS distro is not included in the supportedMasterOsDistros.
func SkipUnlessMasterOSDistroIs(supportedMasterOsDistros ...string) {
	if !framework.MasterOSDistroIs(supportedMasterOsDistros...) {
		skipInternalf(1, "Only supported for master OS distro %v (not %s)", supportedMasterOsDistros, framework.TestContext.MasterOSDistro)
	}
}

// SkipUnlessNodeOSDistroIs skips if the node OS distro is not included in the supportedNodeOsDistros.
func SkipUnlessNodeOSDistroIs(supportedNodeOsDistros ...string) {
	if !framework.NodeOSDistroIs(supportedNodeOsDistros...) {
		skipInternalf(1, "Only supported for node OS distro %v (not %s)", supportedNodeOsDistros, framework.TestContext.NodeOSDistro)
	}
}

// SkipUnlessNodeOSArchIs skips if the node OS distro is not included in the supportedNodeOsArchs.
func SkipUnlessNodeOSArchIs(supportedNodeOsArchs ...string) {
	if !framework.NodeOSArchIs(supportedNodeOsArchs...) {
		skipInternalf(1, "Only supported for node OS arch %v (not %s)", supportedNodeOsArchs, framework.TestContext.NodeOSArch)
	}
}

// SkipIfNodeOSDistroIs skips if the node OS distro is included in the unsupportedNodeOsDistros.
func SkipIfNodeOSDistroIs(unsupportedNodeOsDistros ...string) {
	if framework.NodeOSDistroIs(unsupportedNodeOsDistros...) {
		skipInternalf(1, "Not supported for node OS distro %v (is %s)", unsupportedNodeOsDistros, framework.TestContext.NodeOSDistro)
	}
}

// SkipUnlessServerVersionGTE skips if the server version is less than v.
func SkipUnlessServerVersionGTE(v *utilversion.Version, c discovery.ServerVersionInterface) {
	gte, err := serverVersionGTE(v, c)
	if err != nil {
		framework.Failf("Failed to get server version: %v", err)
	}
	if !gte {
		skipInternalf(1, "Not supported for server versions before %q", v)
	}
}

// SkipUnlessSSHKeyPresent skips if no SSH key is found.
func SkipUnlessSSHKeyPresent() {
	if _, err := e2essh.GetSigner(framework.TestContext.Provider); err != nil {
		skipInternalf(1, "No SSH Key for provider %s: '%v'", framework.TestContext.Provider, err)
	}
}

// serverVersionGTE returns true if v is greater than or equal to the server version.
func serverVersionGTE(v *utilversion.Version, c discovery.ServerVersionInterface) (bool, error) {
	serverVersion, err := c.ServerVersion()
	if err != nil {
		return false, fmt.Errorf("Unable to get server version: %w", err)
	}
	sv, err := utilversion.ParseSemantic(serverVersion.GitVersion)
	if err != nil {
		return false, fmt.Errorf("Unable to parse server version %q: %w", serverVersion.GitVersion, err)
	}
	return sv.AtLeast(v), nil
}

// AppArmorDistros are distros with AppArmor support
var AppArmorDistros = []string{"gci", "ubuntu"}

// SkipIfAppArmorNotSupported skips if the AppArmor is not supported by the node OS distro.
func SkipIfAppArmorNotSupported() {
	SkipUnlessNodeOSDistroIs(AppArmorDistros...)
}

// SkipUnlessComponentRunsAsPodsAndClientCanDeleteThem run if the component run as pods and client can delete them
func SkipUnlessComponentRunsAsPodsAndClientCanDeleteThem(ctx context.Context, componentName string, c clientset.Interface, ns string, labelSet labels.Set) {
	// verify if component run as pod
	label := labels.SelectorFromSet(labelSet)
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(ns).List(ctx, listOpts)
	framework.Logf("SkipUnlessComponentRunsAsPodsAndClientCanDeleteThem: %v, %v", pods, err)
	if err != nil {
		skipInternalf(1, "Skipped because client failed to get component:%s pod err:%v", componentName, err)
	}

	if len(pods.Items) == 0 {
		skipInternalf(1, "Skipped because component:%s is not running as pod.", componentName)
	}

	// verify if client can delete pod
	pod := pods.Items[0]
	if err := c.CoreV1().Pods(ns).Delete(ctx, pod.Name, metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}}); err != nil {
		skipInternalf(1, "Skipped because client failed to delete component:%s pod, err:%v", componentName, err)
	}
}

// SkipIfIPv6 skips if the cluster IP family is IPv6 and the provider is included in the unsupportedProviders.
func SkipIfIPv6(unsupportedProviders ...string) {
	if framework.TestContext.ClusterIsIPv6() && framework.ProviderIs(unsupportedProviders...) {
		skipInternalf(1, "Not supported for IPv6 clusters and providers %v (found %s)", unsupportedProviders, framework.TestContext.Provider)
	}
}
