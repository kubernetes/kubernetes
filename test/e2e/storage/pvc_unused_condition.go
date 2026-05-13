/*
Copyright 2026 The Kubernetes Authors.

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

package storage

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// waitForPVCUnusedCondition polls a PVC until it has an Unused condition with the expected status.
// Returns the PVC with the matching condition or an error if the timeout is reached.
func waitForPVCUnusedCondition(ctx context.Context, c clientset.Interface, ns, pvcName string, expectedStatus v1.ConditionStatus, timeout time.Duration) (*v1.PersistentVolumeClaim, error) {
	var pvc *v1.PersistentVolumeClaim
	err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(ctx, pvcName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range pvc.Status.Conditions {
			if condition.Type == v1.PersistentVolumeClaimUnused {
				return condition.Status == expectedStatus, nil
			}
		}
		return false, nil
	})
	return pvc, err
}

// findPVCCondition returns the condition of the given type from a PVC, or nil.
func findPVCCondition(pvc *v1.PersistentVolumeClaim, condType v1.PersistentVolumeClaimConditionType) *v1.PersistentVolumeClaimCondition {
	for i := range pvc.Status.Conditions {
		if pvc.Status.Conditions[i].Type == condType {
			return &pvc.Status.Conditions[i]
		}
	}
	return nil
}

var _ = utils.SIGDescribe("PVC Unused Condition", framework.WithFeatureGate(features.PersistentVolumeClaimUnusedSinceTime), func() {
	f := framework.NewDefaultFramework("pvc-unused-condition")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var (
		client    clientset.Interface
		namespace string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, client, f.Timeouts.NodeSchedulable))
		e2epv.SkipIfNoDefaultStorageClass(ctx, client)
	})

	// createTestPVC creates a PVC using the default storage class and returns it.
	createTestPVC := func(ctx context.Context) *v1.PersistentVolumeClaim {
		t := testsuites.StorageClassTest{
			Timeouts:  f.Timeouts,
			ClaimSize: "1Gi",
		}
		pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			NamePrefix: "pvc-unused-",
			ClaimSize:  t.ClaimSize,
			VolumeMode: &t.VolumeMode,
		}, namespace)
		pvc, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(ctx, pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating PVC")
		return pvc
	}

	ginkgo.It("should set Unused=True condition when PVC has no referencing pods", func(ctx context.Context) {
		ginkgo.By("Creating a PVC with no pod referencing it")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Waiting for PVC to have Unused=True condition")
		pvc, err := waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition")

		ginkgo.By("Verifying condition details")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil(), "Unused condition should exist")
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionTrue))
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
		gomega.Expect(condition.LastTransitionTime.IsZero()).To(gomega.BeFalse(), "lastTransitionTime should be set")
	})

	ginkgo.It("should set Unused=False when a running pod references the PVC", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Creating a Pod that uses the PVC and waiting for it to be Running")
		pod, err := e2epod.CreatePod(ctx, client, namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")
		defer func() {
			err := e2epod.DeletePodWithWait(ctx, client, pod)
			framework.ExpectNoError(err, "Error deleting pod")
		}()

		ginkgo.By("Waiting for PVC to have Unused=False condition")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition")

		ginkgo.By("Verifying condition details")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil(), "Unused condition should exist")
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionFalse))
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should transition Unused from False to True when pod is deleted", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Creating a Pod that uses the PVC")
		pod, err := e2epod.CreatePod(ctx, client, namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")

		ginkgo.By("Waiting for Unused=False while pod is running")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition")
		conditionBeforeDeletion := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(conditionBeforeDeletion).ToNot(gomega.BeNil())
		transitionTimeBefore := conditionBeforeDeletion.LastTransitionTime

		ginkgo.By("Deleting the pod")
		err = e2epod.DeletePodWithWait(ctx, client, pod)
		framework.ExpectNoError(err, "Error deleting pod")

		ginkgo.By("Waiting for Unused=True after pod deletion")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition after pod deletion")

		ginkgo.By("Verifying lastTransitionTime was updated")
		conditionAfterDeletion := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(conditionAfterDeletion).ToNot(gomega.BeNil())
		gomega.Expect(conditionAfterDeletion.LastTransitionTime.After(transitionTimeBefore.Time)).To(gomega.BeTrue(),
			"lastTransitionTime should be updated after transition from False to True")
	})

	ginkgo.It("should transition Unused from True to False when new pod starts", func(ctx context.Context) {
		ginkgo.By("Creating a PVC with no pod")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Waiting for Unused=True since no pod is using the PVC")
		pvc, err := waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition")

		ginkgo.By("Creating a Pod that uses the PVC")
		pod, err := e2epod.CreatePod(ctx, client, namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")
		defer func() {
			err := e2epod.DeletePodWithWait(ctx, client, pod)
			framework.ExpectNoError(err, "Error deleting pod")
		}()

		ginkgo.By("Waiting for Unused=False after pod creation")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition after pod creation")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should show Unused=True when only terminated pods reference the PVC", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Creating a Pod that completes quickly")
		pod := e2epod.MakePod(namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "echo done")
		pod.Spec.RestartPolicy = v1.RestartPolicyNever
		pod, err := client.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pod")
		defer func() {
			err := e2epod.DeletePodWithWait(ctx, client, pod)
			framework.ExpectNoError(err, "Error deleting pod")
		}()

		ginkgo.By("Waiting for pod to succeed")
		err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, client, pod.Name, namespace, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Pod did not succeed")

		ginkgo.By("Waiting for Unused=True since terminated pods do not count as using the PVC")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition with terminated pod")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
	})

	ginkgo.It("should show Unused=False for pending unscheduled pod", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Creating a Pod with an impossible nodeSelector so it stays Pending")
		pod := e2epod.MakePod(namespace, map[string]string{"non-existent-node": "true"}, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		pod, err := client.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating unschedulable pod")
		defer func() {
			err := e2epod.DeletePodWithWait(ctx, client, pod)
			framework.ExpectNoError(err, "Error deleting unschedulable pod")
		}()

		ginkgo.By("Waiting for pod to be Unschedulable")
		err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, client, pod.Name, namespace)
		framework.ExpectNoError(err, "Pod did not become Unschedulable")

		ginkgo.By("Waiting for Unused=False since unscheduled non-terminal pods still count as using the PVC")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition for unschedulable pod")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should track Unused correctly with multiple pods", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc := createTestPVC(ctx)
		defer func() {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, namespace), "Error deleting PVC")
		}()

		ginkgo.By("Creating Pod1 that uses the PVC")
		pod1, err := e2epod.CreatePod(ctx, client, namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		framework.ExpectNoError(err, "Error creating Pod1")

		ginkgo.By("Verifying PVC has Unused=False with Pod1 running")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False with Pod1")

		ginkgo.By("Creating Pod2 that also uses the same PVC")
		pod2, err := e2epod.CreatePod(ctx, client, namespace, nil, []*v1.PersistentVolumeClaim{pvc}, f.NamespacePodSecurityLevel, "")
		framework.ExpectNoError(err, "Error creating Pod2")

		ginkgo.By("Deleting Pod1")
		err = e2epod.DeletePodWithWait(ctx, client, pod1)
		framework.ExpectNoError(err, "Error deleting Pod1")

		ginkgo.By("Verifying PVC is still Unused=False because Pod2 is still using it")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "PVC should still be Unused=False with Pod2 running")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionFalse))

		ginkgo.By("Deleting Pod2")
		err = e2epod.DeletePodWithWait(ctx, client, pod2)
		framework.ExpectNoError(err, "Error deleting Pod2")

		ginkgo.By("Verifying PVC transitions to Unused=True after all pods are deleted")
		pvc, err = waitForPVCUnusedCondition(ctx, client, namespace, pvc.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True after deleting all pods")
		condition = findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
	})
})
