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
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	componentbaseconfig "k8s.io/component-base/config"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateKubeletConfiguration(t *testing.T) {
	successCase1 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:                   true,
		EnforceNodeAllocatable:          []string{"pods", "system-reserved", "kube-reserved"},
		SystemReservedCgroup:            "/system.slice",
		KubeReservedCgroup:              "/kubelet.service",
		SystemCgroups:                   "",
		CgroupRoot:                      "",
		EventBurst:                      10,
		EventRecordQPS:                  5,
		HealthzPort:                     10248,
		ImageGCHighThresholdPercent:     85,
		ImageGCLowThresholdPercent:      80,
		IPTablesDropBit:                 15,
		IPTablesMasqueradeBit:           14,
		KubeAPIBurst:                    10,
		KubeAPIQPS:                      5,
		MaxOpenFiles:                    1000000,
		MaxPods:                         110,
		OOMScoreAdj:                     -999,
		PodsPerCore:                     100,
		Port:                            65535,
		ReadOnlyPort:                    0,
		RegistryBurst:                   10,
		RegistryPullQPS:                 5,
		HairpinMode:                     kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:        1,
		CPUCFSQuotaPeriod:               metav1.Duration{Duration: 25 * time.Millisecond},
		TopologyManagerScope:            kubeletconfig.PodTopologyManagerScope,
		TopologyManagerPolicy:           kubeletconfig.SingleNumaNodeTopologyManagerPolicy,
		ShutdownGracePeriod:             metav1.Duration{Duration: 30 * time.Second},
		ShutdownGracePeriodCriticalPods: metav1.Duration{Duration: 10 * time.Second},
		MemoryThrottlingFactor:          utilpointer.Float64Ptr(0.8),
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod": true,
			"GracefulNodeShutdown":    true,
			"MemoryQoS":               true,
		},
		Logging: componentbaseconfig.LoggingConfiguration{
			Format: "text",
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase1); allErrors != nil {
		t.Errorf("expect no errors, got %v", allErrors)
	}

	successCase2 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:                   true,
		EnforceNodeAllocatable:          []string{"pods"},
		SystemReservedCgroup:            "",
		KubeReservedCgroup:              "",
		SystemCgroups:                   "",
		CgroupRoot:                      "",
		EventBurst:                      10,
		EventRecordQPS:                  5,
		HealthzPort:                     10248,
		ImageGCHighThresholdPercent:     85,
		ImageGCLowThresholdPercent:      80,
		IPTablesDropBit:                 15,
		IPTablesMasqueradeBit:           14,
		KubeAPIBurst:                    10,
		KubeAPIQPS:                      5,
		MaxOpenFiles:                    1000000,
		MaxPods:                         110,
		OOMScoreAdj:                     -999,
		PodsPerCore:                     100,
		Port:                            65535,
		ReadOnlyPort:                    0,
		RegistryBurst:                   10,
		RegistryPullQPS:                 5,
		HairpinMode:                     kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:        1,
		CPUCFSQuotaPeriod:               metav1.Duration{Duration: 50 * time.Millisecond},
		ReservedSystemCPUs:              "0-3",
		TopologyManagerScope:            kubeletconfig.ContainerTopologyManagerScope,
		TopologyManagerPolicy:           kubeletconfig.NoneTopologyManagerPolicy,
		ShutdownGracePeriod:             metav1.Duration{Duration: 10 * time.Minute},
		ShutdownGracePeriodCriticalPods: metav1.Duration{Duration: 0},
		MemoryThrottlingFactor:          utilpointer.Float64Ptr(0.9),
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod":                true,
			"MemoryQoS":                              true,
			"GracefulNodeShutdownBasedOnPodPriority": true,
		},
		Logging: componentbaseconfig.LoggingConfiguration{
			Format: "text",
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase2); allErrors != nil {
		t.Errorf("expect no errors, got %v", allErrors)
	}

	successCase3 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:                   true,
		EnforceNodeAllocatable:          []string{"pods"},
		SystemReservedCgroup:            "",
		KubeReservedCgroup:              "",
		SystemCgroups:                   "",
		CgroupRoot:                      "",
		EventBurst:                      10,
		EventRecordQPS:                  5,
		HealthzPort:                     10248,
		ImageGCHighThresholdPercent:     85,
		ImageGCLowThresholdPercent:      80,
		IPTablesDropBit:                 15,
		IPTablesMasqueradeBit:           14,
		KubeAPIBurst:                    10,
		KubeAPIQPS:                      5,
		MaxOpenFiles:                    1000000,
		MaxPods:                         110,
		OOMScoreAdj:                     -999,
		PodsPerCore:                     100,
		Port:                            65535,
		ReadOnlyPort:                    0,
		RegistryBurst:                   10,
		RegistryPullQPS:                 5,
		HairpinMode:                     kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:        1,
		CPUCFSQuotaPeriod:               metav1.Duration{Duration: 50 * time.Millisecond},
		ReservedSystemCPUs:              "0-3",
		TopologyManagerScope:            kubeletconfig.ContainerTopologyManagerScope,
		TopologyManagerPolicy:           kubeletconfig.NoneTopologyManagerPolicy,
		ShutdownGracePeriod:             metav1.Duration{Duration: 0},
		ShutdownGracePeriodCriticalPods: metav1.Duration{Duration: 0},
		ShutdownGracePeriodByPodPriority: []kubeletconfig.ShutdownGracePeriodByPodPriority{
			{
				Priority:                   0,
				ShutdownGracePeriodSeconds: 10,
			},
		},
		MemorySwap:             kubeletconfig.MemorySwapConfiguration{SwapBehavior: kubetypes.UnlimitedSwap},
		MemoryThrottlingFactor: utilpointer.Float64Ptr(0.5),
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod":                true,
			"GracefulNodeShutdown":                   true,
			"GracefulNodeShutdownBasedOnPodPriority": true,
			"NodeSwap":                               true,
			"MemoryQoS":                              true,
		},
		Logging: componentbaseconfig.LoggingConfiguration{
			Format: "text",
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase3); allErrors != nil {
		t.Errorf("expect no errors, got %v", allErrors)
	}

	errorCase1 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:                   false,
		EnforceNodeAllocatable:          []string{"pods", "system-reserved", "kube-reserved", "illegal-key"},
		SystemCgroups:                   "/",
		CgroupRoot:                      "",
		EventBurst:                      -10,
		EventRecordQPS:                  -10,
		HealthzPort:                     -10,
		ImageGCHighThresholdPercent:     101,
		ImageGCLowThresholdPercent:      101,
		IPTablesDropBit:                 -10,
		IPTablesMasqueradeBit:           -10,
		KubeAPIBurst:                    -10,
		KubeAPIQPS:                      -10,
		MaxOpenFiles:                    -10,
		MaxPods:                         -10,
		OOMScoreAdj:                     -1001,
		PodsPerCore:                     -10,
		Port:                            0,
		ReadOnlyPort:                    -10,
		RegistryBurst:                   -10,
		RegistryPullQPS:                 -10,
		HairpinMode:                     "foo",
		NodeLeaseDurationSeconds:        -1,
		CPUCFSQuotaPeriod:               metav1.Duration{Duration: 100 * time.Millisecond},
		ShutdownGracePeriod:             metav1.Duration{Duration: 30 * time.Second},
		ShutdownGracePeriodCriticalPods: metav1.Duration{Duration: 60 * time.Second},
		ShutdownGracePeriodByPodPriority: []kubeletconfig.ShutdownGracePeriodByPodPriority{
			{
				Priority:                   0,
				ShutdownGracePeriodSeconds: 10,
			},
		},
		Logging: componentbaseconfig.LoggingConfiguration{
			Format: "",
		},
		MemorySwap: kubeletconfig.MemorySwapConfiguration{SwapBehavior: kubetypes.UnlimitedSwap},
	}
	const numErrsErrorCase1 = 31
	if allErrors := ValidateKubeletConfiguration(errorCase1); len(allErrors.(utilerrors.Aggregate).Errors()) != numErrsErrorCase1 {
		t.Errorf("expect %d errors, got %v", numErrsErrorCase1, len(allErrors.(utilerrors.Aggregate).Errors()))
	}

	errorCase2 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:                   true,
		EnforceNodeAllocatable:          []string{"pods", "system-reserved", "kube-reserved"},
		SystemReservedCgroup:            "/system.slice",
		KubeReservedCgroup:              "/kubelet.service",
		SystemCgroups:                   "",
		CgroupRoot:                      "",
		EventBurst:                      10,
		EventRecordQPS:                  5,
		HealthzPort:                     10248,
		ImageGCHighThresholdPercent:     85,
		ImageGCLowThresholdPercent:      80,
		IPTablesDropBit:                 15,
		IPTablesMasqueradeBit:           14,
		KubeAPIBurst:                    10,
		KubeAPIQPS:                      5,
		MaxOpenFiles:                    1000000,
		MaxPods:                         110,
		OOMScoreAdj:                     -999,
		PodsPerCore:                     100,
		Port:                            65535,
		ReadOnlyPort:                    0,
		RegistryBurst:                   10,
		RegistryPullQPS:                 5,
		HairpinMode:                     kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:        1,
		CPUCFSQuotaPeriod:               metav1.Duration{Duration: 50 * time.Millisecond},
		ReservedSystemCPUs:              "0-3",
		TopologyManagerScope:            "invalid",
		TopologyManagerPolicy:           "invalid",
		ShutdownGracePeriod:             metav1.Duration{Duration: 40 * time.Second},
		ShutdownGracePeriodCriticalPods: metav1.Duration{Duration: 10 * time.Second},
		MemorySwap:                      kubeletconfig.MemorySwapConfiguration{SwapBehavior: "invalid"},
		MemoryThrottlingFactor:          utilpointer.Float64Ptr(1.1),
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod": true,
			"GracefulNodeShutdown":    true,
			"NodeSwap":                true,
			"MemoryQoS":               true,
		},
		Logging: componentbaseconfig.LoggingConfiguration{
			Format: "text",
		},
	}
	const numErrsErrorCase2 = 5
	if allErrors := ValidateKubeletConfiguration(errorCase2); len(allErrors.(utilerrors.Aggregate).Errors()) != numErrsErrorCase2 {
		t.Errorf("expect %d errors, got %v", numErrsErrorCase2, len(allErrors.(utilerrors.Aggregate).Errors()))
	}
}
