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

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func testPodGroup(tCtx ktesting.TContext) {
	tCtx.Parallel()

	podGroupName := "podgroup"
	tests := map[string]struct {
		numPods  int
		podGroup *schedulingapi.PodGroup
	}{
		"gang-2-pods-mincount-2": {
			numPods: 2,
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: podGroupName,
				},
				Spec: schedulingapi.PodGroupSpec{
					SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
						Gang: &schedulingapi.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				},
			},
		},
		"gang-20-pods-mincount-20": {
			numPods: 20,
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: podGroupName,
				},
				Spec: schedulingapi.PodGroupSpec{
					SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
						Gang: &schedulingapi.GangSchedulingPolicy{
							MinCount: 20,
						},
					},
				},
			},
		},
		"gang-5-pods-mincount-2": {
			numPods: 5,
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: podGroupName,
				},
				Spec: schedulingapi.PodGroupSpec{
					SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
						Gang: &schedulingapi.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				},
			},
		},
		"basic-2-pods": {
			numPods: 2,
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: podGroupName,
				},
				Spec: schedulingapi.PodGroupSpec{
					SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
						Basic: &schedulingapi.BasicSchedulingPolicy{},
					},
				},
			},
		},
		"basic-20-pods": {
			numPods: 20,
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: podGroupName,
				},
				Spec: schedulingapi.PodGroupSpec{
					SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
						Basic: &schedulingapi.BasicSchedulingPolicy{},
					},
				},
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			tCtx.Parallel()

			namespace := createTestNamespace(tCtx, nil)
			startScheduler(tCtx)

			class, driverName := createTestClass(tCtx, namespace)
			slice := st.MakeResourceSlice("worker-0", driverName).Devices("device-0")
			createSlice(tCtx, slice.Obj())

			claim := createClaim(tCtx, namespace, "", class, claim)

			podGroup, err := tCtx.Client().SchedulingV1alpha2().PodGroups(namespace).Create(tCtx, test.podGroup, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create PodGroup")
			schedGroup := &v1.PodSchedulingGroup{
				PodGroupName: &podGroup.Name,
			}

			pods := make([]*v1.Pod, test.numPods)
			for i := range pods {
				pod := podWithClaimName.DeepCopy()
				pod.Spec.SchedulingGroup = schedGroup
				pods[i] = createPod(tCtx, namespace, fmt.Sprintf("-%d", i), pod, claim)
			}

			waitForClaimAllocatedToDevice(tCtx, namespace, claim.Name, schedulingTimeout)
			for _, pod := range pods {
				waitForPodScheduled(tCtx, namespace, pod.Name)
			}
		})
	}
}
