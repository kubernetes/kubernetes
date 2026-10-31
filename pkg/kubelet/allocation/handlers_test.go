/*
Copyright The Kubernetes Authors.

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

package allocation

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// swapBehaviorRuntime lets a test return different swap behaviors for the
// desired and allocated pods, which the shared FakeRuntime (keyed by container
// name only) cannot express.
type swapBehaviorRuntime struct {
	*containertest.FakeRuntime
	behaviorFn func(pod *v1.Pod, c *v1.Container) kubetypes.SwapBehavior
}

func (r *swapBehaviorRuntime) GetContainerSwapBehavior(pod *v1.Pod, c *v1.Container) kubetypes.SwapBehavior {
	return r.behaviorFn(pod, c)
}

func TestDisallowResizeForSwappableContainers(t *testing.T) {
	const desiredMarker = "desired"
	mkPod := func(marker string, memReq, memLim, swap string, restartOnMemory bool) *v1.Pod {
		c := v1.Container{Name: "c", Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse(memReq)},
			Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse(memLim)},
		}}
		if swap != "" {
			c.Resources.Limits[v1.ResourceSwap] = resource.MustParse(swap)
		}
		if restartOnMemory {
			c.ResizePolicy = []v1.ContainerResizePolicy{{ResourceName: v1.ResourceMemory, RestartPolicy: v1.RestartContainer}}
		}
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "p", Annotations: map[string]string{"marker": marker}},
			Spec:       v1.PodSpec{Containers: []v1.Container{c}},
		}
	}
	// Mirrors GetContainerSwapBehavior on a LimitedSwap node: NoSwap once
	// requests.memory == limits.memory.
	limitedSwapNode := func(pod *v1.Pod, c *v1.Container) kubetypes.SwapBehavior {
		if c == nil {
			return kubetypes.LimitedSwap
		}
		if c.Resources.Requests.Memory().Cmp(*c.Resources.Limits.Memory()) == 0 {
			return kubetypes.NoSwap
		}
		return kubetypes.LimitedSwap
	}
	workloadControlledNode := func(*v1.Pod, *v1.Container) kubetypes.SwapBehavior {
		return kubetypes.WorkloadControlledSwap
	}
	noSwapNode := func(*v1.Pod, *v1.Container) kubetypes.SwapBehavior {
		return kubetypes.NoSwap
	}
	withPodSwap := func(pod *v1.Pod, swap string) *v1.Pod {
		pod.Spec.Resources = &v1.ResourceRequirements{
			Limits: v1.ResourceList{v1.ResourceSwap: resource.MustParse(swap)},
		}
		return pod
	}

	for _, tc := range []struct {
		name       string
		gate       bool
		plrGate    bool
		behavior   func(*v1.Pod, *v1.Container) kubetypes.SwapBehavior
		allocated  *v1.Pod
		desired    *v1.Pod
		disallowed bool
	}{
		{
			name: "gate off: LimitedSwap memory request change is infeasible (legacy)",
			gate: false, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "2Gi", "4Gi", "", false),
			disallowed: true,
		},
		{
			name: "gate on: LimitedSwap memory request change flows (validated swap delta)",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "2Gi", "4Gi", "", false),
			disallowed: false,
		},
		{
			name: "gate on: LimitedSwap memory request decrease is admitted (usage checked at actuation)",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "4Gi", "8Gi", "", false), desired: mkPod(desiredMarker, "1Gi", "8Gi", "", false),
			disallowed: false,
		},
		{
			name: "gate on: LimitedSwap recalculating to 0 (request == limit) stays infeasible",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "4Gi", "4Gi", "", false),
			disallowed: true,
		},
		{
			name: "gate on: LimitedSwap limit-only resize recalculating to 0 (request == limit) is infeasible",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "1Gi", "1Gi", "", false),
			disallowed: true,
		},
		{
			name: "gate on: explicit limits.swap opts out of the recalculation rule",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "256Mi", false), desired: mkPod(desiredMarker, "4Gi", "4Gi", "256Mi", false),
			disallowed: false,
		},
		{
			name: "gate on: pod-level limits.swap does not opt out when PodLevelResources is disabled",
			gate: true, plrGate: false, behavior: limitedSwapNode,
			allocated:  withPodSwap(mkPod("", "1Gi", "4Gi", "", false), "256Mi"),
			desired:    withPodSwap(mkPod(desiredMarker, "4Gi", "4Gi", "", false), "256Mi"),
			disallowed: true,
		},
		{
			name: "gate on: RestartContainer memory policy bypasses the rule",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", true), desired: mkPod(desiredMarker, "4Gi", "4Gi", "", true),
			disallowed: false,
		},
		{
			name: "gate on: WorkloadControlledSwap node never blocks",
			gate: true, behavior: workloadControlledNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "4Gi", "4Gi", "", false),
			disallowed: false,
		},
		{
			name: "gate on: unchanged memory request is never evaluated",
			gate: true, behavior: limitedSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "", false), desired: mkPod(desiredMarker, "1Gi", "8Gi", "", false),
			disallowed: false,
		},
		{
			name: "gate on: container swap limit resize on NoSwap node is infeasible",
			gate: true, behavior: noSwapNode,
			allocated: mkPod("", "1Gi", "4Gi", "256Mi", false), desired: mkPod(desiredMarker, "1Gi", "4Gi", "512Mi", false),
			disallowed: true,
		},
		{
			name: "gate on: container swap limit resize on WorkloadControlledSwap node is allowed",
			gate: true, behavior: workloadControlledNode,
			allocated: mkPod("", "1Gi", "4Gi", "256Mi", false), desired: mkPod(desiredMarker, "1Gi", "4Gi", "512Mi", false),
			disallowed: false,
		},
		{
			name: "gate on: pod-level swap limit resize on NoSwap node is infeasible",
			gate: true, plrGate: true, behavior: noSwapNode,
			allocated:  withPodSwap(mkPod("", "1Gi", "4Gi", "", false), "1Gi"),
			desired:    withPodSwap(mkPod(desiredMarker, "1Gi", "4Gi", "", false), "2Gi"),
			disallowed: true,
		},
		{
			name: "gate on: pod-level swap limit resize on WorkloadControlledSwap node is allowed",
			gate: true, plrGate: true, behavior: workloadControlledNode,
			allocated:  withPodSwap(mkPod("", "1Gi", "4Gi", "", false), "1Gi"),
			desired:    withPodSwap(mkPod(desiredMarker, "1Gi", "4Gi", "", false), "2Gi"),
			disallowed: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, tc.gate)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, tc.plrGate)
			rt := &swapBehaviorRuntime{FakeRuntime: &containertest.FakeRuntime{}, behaviorFn: tc.behavior}
			disallowed, msg := disallowResizeForSwappableContainers(rt, tc.desired, tc.allocated)
			assert.Equal(t, tc.disallowed, disallowed, "disallowed (msg=%q)", msg)
			if disallowed {
				assert.NotEmpty(t, msg)
			}
		})
	}
}
