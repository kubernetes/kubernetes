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

package framework

import (
	"fmt"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

func skipInternalf(caller int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	ginkgowrapper.Skip(msg, caller+1)
}

// Skipf skips with information about why the test is being skipped.
func Skipf(format string, args ...interface{}) {
	skipInternalf(1, format, args...)
}

// SkipUnlessNodeCountIsAtLeast skips if the number of nodes is less than the minNodeCount.
func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if TestContext.CloudConfig.NumNodes < minNodeCount {
		skipInternalf(1, "Requires at least %d nodes (not %d)", minNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

// SkipUnlessNodeCountIsAtMost skips if the number of nodes is greater than the maxNodeCount.
func SkipUnlessNodeCountIsAtMost(maxNodeCount int) {
	if TestContext.CloudConfig.NumNodes > maxNodeCount {
		skipInternalf(1, "Requires at most %d nodes (not %d)", maxNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

// SkipUnlessAtLeast skips if the value is less than the minValue.
func SkipUnlessAtLeast(value int, minValue int, message string) {
	if value < minValue {
		skipInternalf(1, message)
	}
}

// SkipIfProviderIs skips if the provider is included in the unsupportedProviders.
func SkipIfProviderIs(unsupportedProviders ...string) {
	if ProviderIs(unsupportedProviders...) {
		skipInternalf(1, "Not supported for providers %v (found %s)", unsupportedProviders, TestContext.Provider)
	}
}

// SkipUnlessLocalEphemeralStorageEnabled skips if the LocalStorageCapacityIsolation is not enabled.
func SkipUnlessLocalEphemeralStorageEnabled() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		skipInternalf(1, "Only supported when %v feature is enabled", features.LocalStorageCapacityIsolation)
	}
}

// SkipUnlessSSHKeyPresent skips if no SSH key is found.
func SkipUnlessSSHKeyPresent() {
	if _, err := e2essh.GetSigner(TestContext.Provider); err != nil {
		skipInternalf(1, "No SSH Key for provider %s: '%v'", TestContext.Provider, err)
	}
}

// SkipUnlessProviderIs skips if the provider is not included in the supportedProviders.
func SkipUnlessProviderIs(supportedProviders ...string) {
	if !ProviderIs(supportedProviders...) {
		skipInternalf(1, "Only supported for providers %v (not %s)", supportedProviders, TestContext.Provider)
	}
}

// SkipUnlessMultizone skips if the cluster does not have multizone.
func SkipUnlessMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() <= 1 {
		skipInternalf(1, "Requires more than one zone")
	}
}

// SkipIfMultizone skips if the cluster has multizone.
func SkipIfMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() > 1 {
		skipInternalf(1, "Requires at most one zone")
	}
}

// SkipUnlessPrometheusMonitoringIsEnabled skips if the prometheus monitoring is not enabled.
func SkipUnlessPrometheusMonitoringIsEnabled(supportedMonitoring ...string) {
	if !TestContext.EnablePrometheusMonitoring {
		skipInternalf(1, "Skipped because prometheus monitoring is not enabled")
	}
}

// SkipUnlessMasterOSDistroIs skips if the master OS distro is not included in the supportedMasterOsDistros.
func SkipUnlessMasterOSDistroIs(supportedMasterOsDistros ...string) {
	if !MasterOSDistroIs(supportedMasterOsDistros...) {
		skipInternalf(1, "Only supported for master OS distro %v (not %s)", supportedMasterOsDistros, TestContext.MasterOSDistro)
	}
}

// SkipUnlessNodeOSDistroIs skips if the node OS distro is not included in the supportedNodeOsDistros.
func SkipUnlessNodeOSDistroIs(supportedNodeOsDistros ...string) {
	if !NodeOSDistroIs(supportedNodeOsDistros...) {
		skipInternalf(1, "Only supported for node OS distro %v (not %s)", supportedNodeOsDistros, TestContext.NodeOSDistro)
	}
}

// SkipIfNodeOSDistroIs skips if the node OS distro is included in the unsupportedNodeOsDistros.
func SkipIfNodeOSDistroIs(unsupportedNodeOsDistros ...string) {
	if NodeOSDistroIs(unsupportedNodeOsDistros...) {
		skipInternalf(1, "Not supported for node OS distro %v (is %s)", unsupportedNodeOsDistros, TestContext.NodeOSDistro)
	}
}

// SkipUnlessTaintBasedEvictionsEnabled skips if the TaintBasedEvictions is not enabled.
func SkipUnlessTaintBasedEvictionsEnabled() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.TaintBasedEvictions) {
		skipInternalf(1, "Only supported when %v feature is enabled", features.TaintBasedEvictions)
	}
}

// SkipIfContainerRuntimeIs skips if the container runtime is included in the runtimes.
func SkipIfContainerRuntimeIs(runtimes ...string) {
	for _, containerRuntime := range runtimes {
		if containerRuntime == TestContext.ContainerRuntime {
			skipInternalf(1, "Not supported under container runtime %s", containerRuntime)
		}
	}
}
