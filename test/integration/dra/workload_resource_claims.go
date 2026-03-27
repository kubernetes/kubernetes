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
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// testWorkloadResourceClaims creates a PodGroup with resource claims and a
// Pod referencing those claims and then checks whether those claims getting
// dropped when the DRAWorkloadResourceClaims feature is enabled.
func testWorkloadResourceClaims(tCtx ktesting.TContext, workloadAPIEnabled, workloadResourceClaimsEnabled bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)

	startScheduler(tCtx)
	startClaimController(tCtx)

	// controllerTimeout is the longest the ResourceClaim controller is expected
	// to take to respond.
	controllerTimeout := 15 * time.Second

	class, driverName := createTestClass(tCtx, namespace)

	slice := st.MakeResourceSlice("worker-0", driverName).Devices(device1, device2)
	_ = createSlice(tCtx, slice.Obj())

	podGroupResourceClaim := createClaim(tCtx, namespace, "-podgroup", class, claim)
	podResourceClaim := createClaim(tCtx, namespace, "", class, claim)

	podGroupName := "podgroup"
	schedGroup := &v1.PodSchedulingGroup{
		PodGroupName: &podGroupName,
	}
	podGroup := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podGroupName,
			Namespace: namespace,
		},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			ResourceClaims: []schedulingapi.PodGroupResourceClaim{
				{
					Name:              podGroupResourceClaim.Name,
					ResourceClaimName: &podGroupResourceClaim.Name,
				},
			},
		},
	}
	podGroup, err := tCtx.Client().SchedulingV1alpha2().PodGroups(namespace).Create(tCtx, podGroup, metav1.CreateOptions{FieldValidation: "Strict"})
	if workloadAPIEnabled {
		tCtx.ExpectNoError(err, "create PodGroup")
		if workloadResourceClaimsEnabled {
			assert.NotEmpty(tCtx, podGroup.Spec.ResourceClaims, "should store resource claims in PodGroup spec")
		} else {
			assert.Empty(tCtx, podGroup.Spec.ResourceClaims, "should drop resource claims from PodGroup spec")
		}
	} else {
		assert.True(tCtx, apierrors.IsNotFound(err), "PodGroup API should not be available")
	}

	tCtx.Log("Creating the first Pod")
	pod := podWithClaimName.DeepCopy()
	pod.Spec.SchedulingGroup = schedGroup
	pod = createPod(tCtx, namespace, "-1", pod, podResourceClaim, podGroupResourceClaim)

	waitForClaimAllocatedToDevice(tCtx, namespace, podResourceClaim.Name, schedulingTimeout)
	waitForClaim(tCtx, namespace, podResourceClaim.Name, schedulingTimeout,
		gomega.HaveField(
			"Status.ReservedFor",
			gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(pod.UID))),
		),
		"Claim should have been reserved for the Pod.",
	)
	if workloadResourceClaimsEnabled {
		waitForClaimAllocatedToDevice(tCtx, namespace, podGroupResourceClaim.Name, schedulingTimeout)
		waitForClaim(tCtx, namespace, podGroupResourceClaim.Name, schedulingTimeout,
			gomega.HaveField(
				"Status.ReservedFor",
				gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(podGroup.UID))),
			),
			"Claim should have been reserved for the PodGroup.",
		)
	}

	waitForPodScheduled(tCtx, namespace, pod.Name)

	tCtx.Log("Creating the second Pod")
	secondPod := podWithClaimName.DeepCopy()
	secondPod.Spec.SchedulingGroup = schedGroup
	secondPod = createPod(tCtx, namespace, "-2", secondPod, podResourceClaim, podGroupResourceClaim)

	waitForClaim(tCtx, namespace, podResourceClaim.Name, schedulingTimeout,
		gomega.HaveField(
			"Status.ReservedFor",
			gomega.ConsistOf(
				gomega.HaveField("UID", gomega.Equal(pod.UID)),
				gomega.HaveField("UID", gomega.Equal(secondPod.UID)),
			),
		),
		"Claim should have been reserved for both Pods.",
	)
	if workloadResourceClaimsEnabled {
		waitForClaim(tCtx, namespace, podGroupResourceClaim.Name, schedulingTimeout,
			gomega.HaveField(
				"Status.ReservedFor",
				gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(podGroup.UID))),
			),
			"Claim should stay reserved only for the PodGroup after creating another Pod.",
		)
	}

	waitForPodScheduled(tCtx, namespace, secondPod.Name)

	tCtx.Log("Deleting the first Pod")
	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pod.Name)

	waitForClaim(tCtx, namespace, podResourceClaim.Name, controllerTimeout,
		gomega.HaveField(
			"Status.ReservedFor",
			gomega.ConsistOf(
				gomega.HaveField("UID", gomega.Equal(secondPod.UID)),
			),
		),
		"Claim should only be reserved for the second Pod.",
	)
	if workloadResourceClaimsEnabled {
		tCtx.Consistently(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, podGroupResourceClaim.Name, metav1.GetOptions{})
			return c, err
		}).
			WithTimeout(controllerTimeout).
			WithPolling(time.Second).
			Should(
				gomega.HaveField(
					"Status.ReservedFor",
					gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(podGroup.UID))),
				),
				"Claim should stay reserved for the PodGroup after deleting the first Pod.",
			)
	}

	tCtx.Log("Deleting the second Pod")
	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, secondPod.Name)

	waitForClaim(tCtx, namespace, podResourceClaim.Name, controllerTimeout,
		gomega.HaveField(
			"Status.ReservedFor",
			gomega.BeEmpty(),
		),
		"Claim should not be reserved after deleting the second Pod.",
	)
	if workloadResourceClaimsEnabled {
		tCtx.Consistently(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, podGroupResourceClaim.Name, metav1.GetOptions{})
			return c, err
		}).
			WithTimeout(controllerTimeout).
			WithPolling(time.Second).
			Should(
				gomega.HaveField(
					"Status.ReservedFor",
					gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(podGroup.UID))),
				),
				"Claim should stay reserved for the PodGroup after deleting all of its Pods.",
			)

		tCtx.Log("Deleting the PodGroup")
		// Remove the finalizer. The podgroupprotection controller isn't running
		// to remove it.
		patch := []byte(`{"metadata": {"finalizers": null}}`)
		podGroup, err = tCtx.Client().SchedulingV1alpha2().PodGroups(namespace).Patch(tCtx, podGroup.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
		tCtx.ExpectNoError(err)
		deleteAndWait(tCtx, tCtx.Client().SchedulingV1alpha2().PodGroups(namespace).Delete, tCtx.Client().SchedulingV1alpha2().PodGroups(namespace).Get, podGroup.Name)

		waitForClaim(tCtx, namespace, podGroupResourceClaim.Name, controllerTimeout,
			gomega.HaveField(
				"Status.ReservedFor",
				gomega.BeEmpty(),
			),
			"Claim should not be reserved for the PodGroup after deleting it.",
		)
	}
}
