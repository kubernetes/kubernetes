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

package validation

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

var (
	defaultCFSQuota = metav1.Duration{Duration: 100 * time.Millisecond}
)

// ValidateKubeletConfiguration validates `kc` and returns an error if it is invalid
func ValidateKubeletConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	allErrors := []error{}

	// Make a local copy of the global feature gates and combine it with the gates set by this configuration.
	// This allows us to validate the config against the set of gates it will actually run against.
	localFeatureGate := utilfeature.DefaultFeatureGate.DeepCopy()
	if err := localFeatureGate.SetFromMap(kc.FeatureGates); err != nil {
		return err
	}

	if kc.NodeLeaseDurationSeconds <= 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: nodeLeaseDurationSeconds must be greater than 0"))
	}
	if !kc.CgroupsPerQOS && len(kc.EnforceNodeAllocatable) > 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: enforceNodeAllocatable (--enforce-node-allocatable) is not supported unless cgroupsPerQOS (--cgroups-per-qos) is set to true"))
	}
	if kc.SystemCgroups != "" && kc.CgroupRoot == "" {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: systemCgroups (--system-cgroups) was specified and cgroupRoot (--cgroup-root) was not specified"))
	}
	if kc.EventBurst < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: eventBurst (--event-burst) %v must not be a negative number", kc.EventBurst))
	}
	if kc.EventRecordQPS < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: eventRecordQPS (--event-qps) %v must not be a negative number", kc.EventRecordQPS))
	}
	if kc.HealthzPort != 0 && utilvalidation.IsValidPortNum(int(kc.HealthzPort)) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: healthzPort (--healthz-port) %v must be between 1 and 65535, inclusive", kc.HealthzPort))
	}
	if !localFeatureGate.Enabled(features.CPUCFSQuotaPeriod) && kc.CPUCFSQuotaPeriod != defaultCFSQuota {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: cpuCFSQuotaPeriod %v requires feature gate CustomCPUCFSQuotaPeriod", kc.CPUCFSQuotaPeriod))
	}
	if localFeatureGate.Enabled(features.CPUCFSQuotaPeriod) && utilvalidation.IsInRange(int(kc.CPUCFSQuotaPeriod.Duration), int(1*time.Microsecond), int(time.Second)) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: cpuCFSQuotaPeriod (--cpu-cfs-quota-period) %v must be between 1usec and 1sec, inclusive", kc.CPUCFSQuotaPeriod))
	}
	if utilvalidation.IsInRange(int(kc.ImageGCHighThresholdPercent), 0, 100) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: imageGCHighThresholdPercent (--image-gc-high-threshold) %v must be between 0 and 100, inclusive", kc.ImageGCHighThresholdPercent))
	}
	if utilvalidation.IsInRange(int(kc.ImageGCLowThresholdPercent), 0, 100) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: imageGCLowThresholdPercent (--image-gc-low-threshold) %v must be between 0 and 100, inclusive", kc.ImageGCLowThresholdPercent))
	}
	if kc.ImageGCLowThresholdPercent >= kc.ImageGCHighThresholdPercent {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: imageGCLowThresholdPercent (--image-gc-low-threshold) %v must be less than imageGCHighThresholdPercent (--image-gc-high-threshold) %v", kc.ImageGCLowThresholdPercent, kc.ImageGCHighThresholdPercent))
	}
	if utilvalidation.IsInRange(int(kc.IPTablesDropBit), 0, 31) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: iptablesDropBit (--iptables-drop-bit) %v must be between 0 and 31, inclusive", kc.IPTablesDropBit))
	}
	if utilvalidation.IsInRange(int(kc.IPTablesMasqueradeBit), 0, 31) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: iptablesMasqueradeBit (--iptables-masquerade-bit) %v must be between 0 and 31, inclusive", kc.IPTablesMasqueradeBit))
	}
	if kc.KubeAPIBurst < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: kubeAPIBurst (--kube-api-burst) %v must not be a negative number", kc.KubeAPIBurst))
	}
	if kc.KubeAPIQPS < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: kubeAPIQPS (--kube-api-qps) %v must not be a negative number", kc.KubeAPIQPS))
	}
	if kc.NodeStatusMaxImages < -1 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: nodeStatusMaxImages (--node-status-max-images) must be -1 or greater"))
	}
	if kc.MaxOpenFiles < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: maxOpenFiles (--max-open-files) %v must not be a negative number", kc.MaxOpenFiles))
	}
	if kc.MaxPods < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: maxPods (--max-pods) %v must not be a negative number", kc.MaxPods))
	}
	if utilvalidation.IsInRange(int(kc.OOMScoreAdj), -1000, 1000) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: oomScoreAdj (--oom-score-adj) %v must be between -1000 and 1000, inclusive", kc.OOMScoreAdj))
	}
	if kc.PodsPerCore < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: podsPerCore (--pods-per-core) %v must not be a negative number", kc.PodsPerCore))
	}
	if utilvalidation.IsValidPortNum(int(kc.Port)) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: port (--port) %v must be between 1 and 65535, inclusive", kc.Port))
	}
	if kc.ReadOnlyPort != 0 && utilvalidation.IsValidPortNum(int(kc.ReadOnlyPort)) != nil {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: readOnlyPort (--read-only-port) %v must be between 0 and 65535, inclusive", kc.ReadOnlyPort))
	}
	if kc.RegistryBurst < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: registryBurst (--registry-burst) %v must not be a negative number", kc.RegistryBurst))
	}
	if kc.RegistryPullQPS < 0 {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: registryPullQPS (--registry-qps) %v must not be a negative number", kc.RegistryPullQPS))
	}
	if kc.ServerTLSBootstrap && !localFeatureGate.Enabled(features.RotateKubeletServerCertificate) {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: serverTLSBootstrap %v requires feature gate RotateKubeletServerCertificate", kc.ServerTLSBootstrap))
	}
	if kc.TopologyManagerPolicy != kubeletconfig.NoneTopologyManagerPolicy && !localFeatureGate.Enabled(features.TopologyManager) {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: topologyManagerPolicy %v requires feature gate TopologyManager", kc.TopologyManagerPolicy))
	}
	switch kc.TopologyManagerPolicy {
	case kubeletconfig.NoneTopologyManagerPolicy:
	case kubeletconfig.BestEffortTopologyManagerPolicy:
	case kubeletconfig.RestrictedTopologyManagerPolicy:
	case kubeletconfig.SingleNumaNodeTopologyManagerPolicy:
	default:
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: topologyManagerPolicy non-allowable value: %v", kc.TopologyManagerPolicy))
	}
	if kc.TopologyManagerScope != kubeletconfig.ContainerTopologyManagerScope && !localFeatureGate.Enabled(features.TopologyManager) {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: topologyManagerScope %v requires feature gate TopologyManager", kc.TopologyManagerScope))
	}
	if kc.TopologyManagerScope != kubeletconfig.ContainerTopologyManagerScope && kc.TopologyManagerScope != kubeletconfig.PodTopologyManagerScope {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: topologyManagerScope non-allowable value: %v", kc.TopologyManagerScope))
	}

	if localFeatureGate.Enabled(features.GracefulNodeShutdown) {
		if kc.ShutdownGracePeriod.Duration < 0 || kc.ShutdownGracePeriodCriticalPods.Duration < 0 || kc.ShutdownGracePeriodCriticalPods.Duration > kc.ShutdownGracePeriod.Duration {
			allErrors = append(allErrors, fmt.Errorf("invalid configuration: ShutdownGracePeriod %v must be >= 0, ShutdownGracePeriodCriticalPods %v must be >= 0, and ShutdownGracePeriodCriticalPods %v must be <= ShutdownGracePeriod %v", kc.ShutdownGracePeriod, kc.ShutdownGracePeriodCriticalPods, kc.ShutdownGracePeriodCriticalPods, kc.ShutdownGracePeriod))
		}
		if kc.ShutdownGracePeriod.Duration > 0 && kc.ShutdownGracePeriod.Duration < time.Duration(time.Second) {
			allErrors = append(allErrors, fmt.Errorf("invalid configuration: ShutdownGracePeriod %v must be either zero or otherwise >= 1 sec", kc.ShutdownGracePeriod))
		}
		if kc.ShutdownGracePeriodCriticalPods.Duration > 0 && kc.ShutdownGracePeriodCriticalPods.Duration < time.Duration(time.Second) {
			allErrors = append(allErrors, fmt.Errorf("invalid configuration: ShutdownGracePeriodCriticalPods %v must be either zero or otherwise >= 1 sec", kc.ShutdownGracePeriodCriticalPods))
		}
	}
	if (kc.ShutdownGracePeriod.Duration > 0 || kc.ShutdownGracePeriodCriticalPods.Duration > 0) && !localFeatureGate.Enabled(features.GracefulNodeShutdown) {
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: Specifying ShutdownGracePeriod or ShutdownGracePeriodCriticalPods requires feature gate GracefulNodeShutdown"))
	}

	for _, val := range kc.EnforceNodeAllocatable {
		switch val {
		case kubetypes.NodeAllocatableEnforcementKey:
		case kubetypes.SystemReservedEnforcementKey:
			if kc.SystemReservedCgroup == "" {
				allErrors = append(allErrors, fmt.Errorf("invalid configuration: systemReservedCgroup (--system-reserved-cgroup) must be specified when 'system-reserved' contained in enforceNodeAllocatable (--enforce-node-allocatable)"))
			}
		case kubetypes.KubeReservedEnforcementKey:
			if kc.KubeReservedCgroup == "" {
				allErrors = append(allErrors, fmt.Errorf("invalid configuration: kubeReservedCgroup (--kube-reserved-cgroup) must be specified when 'kube-reserved' contained in enforceNodeAllocatable (--enforce-node-allocatable)"))
			}
		case kubetypes.NodeAllocatableNoneKey:
			if len(kc.EnforceNodeAllocatable) > 1 {
				allErrors = append(allErrors, fmt.Errorf("invalid configuration: enforceNodeAllocatable (--enforce-node-allocatable) may not contain additional enforcements when '%s' is specified", kubetypes.NodeAllocatableNoneKey))
			}
		default:
			allErrors = append(allErrors, fmt.Errorf("invalid configuration: option %q specified for enforceNodeAllocatable (--enforce-node-allocatable). Valid options are %q, %q, %q, or %q",
				val, kubetypes.NodeAllocatableEnforcementKey, kubetypes.SystemReservedEnforcementKey, kubetypes.KubeReservedEnforcementKey, kubetypes.NodeAllocatableNoneKey))
		}
	}
	switch kc.HairpinMode {
	case kubeletconfig.HairpinNone:
	case kubeletconfig.HairpinVeth:
	case kubeletconfig.PromiscuousBridge:
	default:
		allErrors = append(allErrors, fmt.Errorf("invalid configuration: option %q specified for hairpinMode (--hairpin-mode). Valid options are %q, %q or %q",
			kc.HairpinMode, kubeletconfig.HairpinNone, kubeletconfig.HairpinVeth, kubeletconfig.PromiscuousBridge))
	}
	if kc.ReservedSystemCPUs != "" {
		// --reserved-cpus does not support --system-reserved-cgroup or --kube-reserved-cgroup
		if kc.SystemReservedCgroup != "" || kc.KubeReservedCgroup != "" {
			allErrors = append(allErrors, fmt.Errorf("can't use reservedSystemCPUs (--reserved-cpus) with systemReservedCgroup (--system-reserved-cgroup) or kubeReservedCgroup (--kube-reserved-cgroup)"))
		}
		if _, err := cpuset.Parse(kc.ReservedSystemCPUs); err != nil {
			allErrors = append(allErrors, fmt.Errorf("unable to parse reservedSystemCPUs (--reserved-cpus), error: %v", err))
		}
	}

	if err := validateKubeletOSConfiguration(kc); err != nil {
		allErrors = append(allErrors, err)
	}
	allErrors = append(allErrors, metrics.ValidateShowHiddenMetricsVersion(kc.ShowHiddenMetricsForVersion)...)

	logOption := logs.NewOptions()
	if kc.Logging.Format != "" {
		logOption.LogFormat = kc.Logging.Format
	}
	allErrors = append(allErrors, logOption.Validate()...)

	return utilerrors.NewAggregate(allErrors)
}
