/*
Copyright 2024 The Kubernetes Authors.

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

package cpumanager

import (
	"testing"

	"github.com/blang/semver/v4"

	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/cpuset"
)

func TestMetricsAllocate(t *testing.T) {
	counterMetricValue, _ := testutil.GetCounterMetricValue(metrics.ContainerAlignedComputeResources.WithLabelValues(metrics.AlignScopeContainer, metrics.AlignedPhysicalCPU))

	testCases := []struct {
		staticPolicyTest
		expCounterMetricValue float64
	}{
		{
			staticPolicyTest: staticPolicyTest{
				description: "SocketNoSMT",
				topo:        topoUncoreDualSocketNoSMT,
				pod: makeMultiContainerPod(
					[]struct{ request, limit string }{
						{"4000m", "4000m"}},
					[]struct{ request, limit string }{
						{"2000m", "2000m"}}),
				containerName:   "initContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expCounterMetricValue: counterMetricValue + 2,
		},
		{
			staticPolicyTest: staticPolicyTest{
				description: "SingleSocketHT",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPod(
					[]struct{ request, limit string }{
						{"4000m", "4000m"}},
					[]struct{ request, limit string }{
						{"2000m", "2000m"}}),
				containerName:   "initContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expCounterMetricValue: counterMetricValue + 2,
		},
	}

	// either clear the metric if it's not created or restore it after this UT.
	if !metrics.ContainerAlignedComputeResources.IsCreated() {
		defer metrics.ContainerAlignedComputeResources.ClearState()
		metrics.ContainerAlignedComputeResources.Create(&semver.SpecVersion)
	} else {
		defer metrics.ContainerAlignedComputeResources.WithLabelValues(metrics.AlignScopeContainer, metrics.AlignedPhysicalCPU).Add(-2)
	}

	for _, testCase := range testCases {
		policy, _ := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(), nil)

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		pod := testCase.pod // shortcut

		// allocate
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
			if err := policy.Allocate(st, pod, &container); err != nil {
				t.Errorf("StaticPolicy Allocate() error (%v).", err)
			}
		}

		mValue, err := testutil.GetCounterMetricValue(metrics.ContainerAlignedComputeResources.WithLabelValues(metrics.AlignScopeContainer, metrics.AlignedPhysicalCPU))
		if err != nil {
			t.Errorf("StaticPolicy GetCounterMetricValue() error (%v)", err)
		}

		if mValue != testCase.expCounterMetricValue {
			t.Errorf("expected counter metric value  %v but got %v", testCase.expCounterMetricValue, mValue)
		}
	}
}
