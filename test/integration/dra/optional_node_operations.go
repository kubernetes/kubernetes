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

package dra

import (
	"fmt"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// testOptionalNodeOperations verifies DRA optional node operations.
// Note: This test relies on the shared test setup (in dra.go) having configured
// worker-0 to declare support for DRAOptionalNodeOperations when this feature is enabled.
// All other nodes (worker-1 through worker-7) lack this declared feature.
func testOptionalNodeOperations(tCtx ktesting.TContext, enabled bool) {
	if !enabled {
		return
	}
	tCtx.Parallel()

	tests := map[string]struct {
		usePodGroup     bool
		nodeSelector    map[string]string
		expectScheduled bool
		expectedNode    string
	}{
		"pod-scheduled-on-supported-node": {
			usePodGroup:     false,
			expectScheduled: true,
			expectedNode:    "worker-0",
		},
		"pod-unschedulable-on-unsupported-node": {
			usePodGroup:     false,
			nodeSelector:    map[string]string{"kubernetes.io/hostname": "worker-1"},
			expectScheduled: false,
		},
		"podgroup-scheduled-on-supported-node": {
			usePodGroup:     true,
			expectScheduled: true,
			expectedNode:    "worker-0",
		},
		"podgroup-unschedulable-on-unsupported-node": {
			usePodGroup:     true,
			nodeSelector:    map[string]string{"kubernetes.io/hostname": "worker-1"},
			expectScheduled: false,
		},
	}

	for name, tt := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			tCtx.Parallel()
			namespace := createTestNamespace(tCtx, nil)
			startScheduler(tCtx)
			class, driverName := createTestClass(tCtx, namespace)

			slice := st.MakeResourceSliceWithAllNodes("shared-slice", driverName).
				Devices(device1).
				SkipNodeOperations(resourceapi.SkipNodeOperationAll).
				Obj()
			createSlice(tCtx, slice)

			claimObj := createClaim(tCtx, namespace, "", class, claim)

			var podGroup *schedulingapi.PodGroup
			if tt.usePodGroup {
				pg := &schedulingapi.PodGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "podgroup",
						Namespace: namespace,
					},
					Spec: schedulingapi.PodGroupSpec{
						SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
							Basic: &schedulingapi.BasicSchedulingPolicy{},
						},
						ResourceClaims: []schedulingapi.PodGroupResourceClaim{
							{
								Name:              claimObj.Name,
								ResourceClaimName: &claimObj.Name,
							},
						},
					},
				}
				podGroup = must(tCtx, tCtx.Client().SchedulingV1beta1().PodGroups(namespace).Create, pg, metav1.CreateOptions{})
			}

			numPods := 1
			if tt.usePodGroup {
				numPods = 2
			}

			var pods []*v1.Pod
			for i := range numPods {
				p := podWithClaimName.DeepCopy()
				if tt.usePodGroup {
					p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
						PodGroupName: &podGroup.Name,
					}
				}
				// Apply node selector to the last pod if specified to test rejection.
				if i == numPods-1 && tt.nodeSelector != nil {
					p.Spec.NodeSelector = tt.nodeSelector
				}
				pod := createPod(tCtx, namespace, fmt.Sprintf("-%d", i), p, claimObj)
				pods = append(pods, pod)
			}

			if tt.expectScheduled {
				for _, pod := range pods {
					scheduledPod := waitForPodScheduled(tCtx, namespace, pod.Name)
					tCtx.Expect(scheduledPod.Spec.NodeName).To(gomega.Equal(tt.expectedNode), "pod should be scheduled to %s", tt.expectedNode)
				}
			} else {
				// Verify that the pod with node selector (the last one) is unschedulable.
				unschedulablePod := pods[numPods-1]
				expectPodUnschedulable(tCtx, unschedulablePod, "")
			}
		})
	}
}
