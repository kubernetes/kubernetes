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
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestValidateKubeletConfiguration(t *testing.T) {
	successCase1 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:               true,
		EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved"},
		SystemReservedCgroup:        "/system.slice",
		KubeReservedCgroup:          "/kubelet.service",
		SystemCgroups:               "",
		CgroupRoot:                  "",
		EventBurst:                  10,
		EventRecordQPS:              5,
		HealthzPort:                 10248,
		ImageGCHighThresholdPercent: 85,
		ImageGCLowThresholdPercent:  80,
		IPTablesDropBit:             15,
		IPTablesMasqueradeBit:       14,
		KubeAPIBurst:                10,
		KubeAPIQPS:                  5,
		MaxOpenFiles:                1000000,
		MaxPods:                     110,
		OOMScoreAdj:                 -999,
		PodsPerCore:                 100,
		Port:                        65535,
		ReadOnlyPort:                0,
		RegistryBurst:               10,
		RegistryPullQPS:             5,
		HairpinMode:                 kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:    1,
		CPUCFSQuotaPeriod:           metav1.Duration{Duration: 25 * time.Millisecond},
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod": true,
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase1); allErrors != nil {
		t.Errorf("expect no errors, got %v", allErrors)
	}

	successCase2 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:               true,
		EnforceNodeAllocatable:      []string{"pods"},
		SystemReservedCgroup:        "",
		KubeReservedCgroup:          "",
		SystemCgroups:               "",
		CgroupRoot:                  "",
		EventBurst:                  10,
		EventRecordQPS:              5,
		HealthzPort:                 10248,
		ImageGCHighThresholdPercent: 85,
		ImageGCLowThresholdPercent:  80,
		IPTablesDropBit:             15,
		IPTablesMasqueradeBit:       14,
		KubeAPIBurst:                10,
		KubeAPIQPS:                  5,
		MaxOpenFiles:                1000000,
		MaxPods:                     110,
		OOMScoreAdj:                 -999,
		PodsPerCore:                 100,
		Port:                        65535,
		ReadOnlyPort:                0,
		RegistryBurst:               10,
		RegistryPullQPS:             5,
		HairpinMode:                 kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:    1,
		CPUCFSQuotaPeriod:           metav1.Duration{Duration: 50 * time.Millisecond},
		ReservedSystemCPUs:          "0-3",
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod": true,
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase2); allErrors != nil {
		t.Errorf("expect no errors, got %v", allErrors)
	}

	errorCase1 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:               false,
		EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved", "illegal-key"},
		SystemCgroups:               "/",
		CgroupRoot:                  "",
		EventBurst:                  -10,
		EventRecordQPS:              -10,
		HealthzPort:                 -10,
		ImageGCHighThresholdPercent: 101,
		ImageGCLowThresholdPercent:  101,
		IPTablesDropBit:             -10,
		IPTablesMasqueradeBit:       -10,
		KubeAPIBurst:                -10,
		KubeAPIQPS:                  -10,
		MaxOpenFiles:                -10,
		MaxPods:                     -10,
		OOMScoreAdj:                 -1001,
		PodsPerCore:                 -10,
		Port:                        0,
		ReadOnlyPort:                -10,
		RegistryBurst:               -10,
		RegistryPullQPS:             -10,
		HairpinMode:                 "foo",
		NodeLeaseDurationSeconds:    -1,
		CPUCFSQuotaPeriod:           metav1.Duration{Duration: 100 * time.Millisecond},
	}
	const numErrsErrorCase1 = 25
	if allErrors := ValidateKubeletConfiguration(errorCase1); len(allErrors.(utilerrors.Aggregate).Errors()) != numErrsErrorCase1 {
		t.Errorf("expect %d errors, got %v", numErrsErrorCase1, len(allErrors.(utilerrors.Aggregate).Errors()))
	}

	errorCase2 := &kubeletconfig.KubeletConfiguration{
		CgroupsPerQOS:               true,
		EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved"},
		SystemReservedCgroup:        "/system.slice",
		KubeReservedCgroup:          "/kubelet.service",
		SystemCgroups:               "",
		CgroupRoot:                  "",
		EventBurst:                  10,
		EventRecordQPS:              5,
		HealthzPort:                 10248,
		ImageGCHighThresholdPercent: 85,
		ImageGCLowThresholdPercent:  80,
		IPTablesDropBit:             15,
		IPTablesMasqueradeBit:       14,
		KubeAPIBurst:                10,
		KubeAPIQPS:                  5,
		MaxOpenFiles:                1000000,
		MaxPods:                     110,
		OOMScoreAdj:                 -999,
		PodsPerCore:                 100,
		Port:                        65535,
		ReadOnlyPort:                0,
		RegistryBurst:               10,
		RegistryPullQPS:             5,
		HairpinMode:                 kubeletconfig.PromiscuousBridge,
		NodeLeaseDurationSeconds:    1,
		CPUCFSQuotaPeriod:           metav1.Duration{Duration: 50 * time.Millisecond},
		ReservedSystemCPUs:          "0-3",
		FeatureGates: map[string]bool{
			"CustomCPUCFSQuotaPeriod": true,
		},
	}
	const numErrsErrorCase2 = 1
	if allErrors := ValidateKubeletConfiguration(errorCase2); len(allErrors.(utilerrors.Aggregate).Errors()) != numErrsErrorCase2 {
		t.Errorf("expect %d errors, got %v", numErrsErrorCase2, len(allErrors.(utilerrors.Aggregate).Errors()))
	}
}
