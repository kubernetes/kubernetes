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

package csimock

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
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

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

func findPVCCondition(pvc *v1.PersistentVolumeClaim, condType v1.PersistentVolumeClaimConditionType) *v1.PersistentVolumeClaimCondition {
	for i := range pvc.Status.Conditions {
		if pvc.Status.Conditions[i].Type == condType {
			return &pvc.Status.Conditions[i]
		}
	}
	return nil
}

var _ = utils.SIGDescribe("CSI Mock PVC Unused Condition", framework.WithFeatureGate(features.PersistentVolumeClaimUnusedSinceTime), func() {
	f := framework.NewDefaultFramework("csi-mock-pvc-unused")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.It("should set Unused=True condition when PVC has no referencing pods", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC with no pod referencing it")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Waiting for PVC to have Unused=True condition")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition")

		ginkgo.By("Verifying condition details")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil(), "Unused condition should exist")
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionTrue))
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
		gomega.Expect(condition.LastTransitionTime.IsZero()).To(gomega.BeFalseBecause("lastTransitionTime should be set"))
	})

	ginkgo.It("should set Unused=False when a running pod references the PVC", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Creating a Pod that uses the PVC and waiting for it to be Running")
		pod, err := m.createPodWithPVC(claim)
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")
		err = e2epod.WaitForPodRunningInNamespace(ctx, m.cs, pod)
		framework.ExpectNoError(err, "Error waiting for pod to be Running")

		ginkgo.By("Waiting for PVC to have Unused=False condition")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition")

		ginkgo.By("Verifying condition details")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil(), "Unused condition should exist")
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionFalse))
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should transition Unused from False to True when pod is deleted", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Creating a Pod that uses the PVC")
		pod, err := m.createPodWithPVC(claim)
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")
		err = e2epod.WaitForPodRunningInNamespace(ctx, m.cs, pod)
		framework.ExpectNoError(err, "Error waiting for pod to be Running")

		ginkgo.By("Waiting for Unused=False while pod is running")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition")
		conditionBeforeDeletion := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(conditionBeforeDeletion).ToNot(gomega.BeNil())
		transitionTimeBefore := conditionBeforeDeletion.LastTransitionTime

		ginkgo.By("Deleting the pod")
		err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
		framework.ExpectNoError(err, "Error deleting pod")

		ginkgo.By("Waiting for Unused=True after pod deletion")
		pvc, err = waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition after pod deletion")

		ginkgo.By("Verifying lastTransitionTime was updated")
		conditionAfterDeletion := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(conditionAfterDeletion).ToNot(gomega.BeNil())
		gomega.Expect(conditionAfterDeletion.LastTransitionTime.After(transitionTimeBefore.Time)).To(gomega.BeTrueBecause("lastTransitionTime should be updated after transition from False to True"))
	})

	ginkgo.It("should transition Unused from True to False when new pod starts", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC with no pod")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Waiting for Unused=True since no pod is using the PVC")
		_, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition")

		ginkgo.By("Creating a Pod that uses the PVC")
		pod, err := m.createPodWithPVC(claim)
		framework.ExpectNoError(err, "Error creating pod that uses the PVC")
		err = e2epod.WaitForPodRunningInNamespace(ctx, m.cs, pod)
		framework.ExpectNoError(err, "Error waiting for pod to be Running")

		ginkgo.By("Waiting for Unused=False after pod creation")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition after pod creation")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should show Unused=True when only terminated pods reference the PVC", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Creating a Pod that completes quickly")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "pvc-unused-terminated-",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "volume-tester",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "echo done"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "my-volume",
								MountPath: "/mnt/test",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "my-volume",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: claim.Name,
								ReadOnly:  false,
							},
						},
					},
				},
			},
		}
		e2epod.SetNodeSelection(&pod.Spec, m.config.ClientNodeSelection)
		pod, err := m.cs.CoreV1().Pods(claim.Namespace).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pod")
		m.pods = append(m.pods, pod)

		ginkgo.By("Waiting for pod to succeed")
		err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, m.cs, pod.Name, claim.Namespace, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Pod did not succeed")

		ginkgo.By("Waiting for Unused=True since terminated pods do not count as using the PVC")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True condition with terminated pod")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
	})

	ginkgo.It("should show Unused=False for pending unscheduled pod", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Creating a Pod with an impossible nodeSelector so it stays Pending")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "pvc-unused-unschedulable-",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "volume-tester",
						Image: imageutils.GetE2EImage(imageutils.Pause),
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "my-volume",
								MountPath: "/mnt/test",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeSelector:  map[string]string{"non-existent-node": "true"},
				Volumes: []v1.Volume{
					{
						Name: "my-volume",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: claim.Name,
								ReadOnly:  false,
							},
						},
					},
				},
			},
		}
		pod, err := m.cs.CoreV1().Pods(claim.Namespace).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating unschedulable pod")
		m.pods = append(m.pods, pod)

		ginkgo.By("Waiting for pod to be Unschedulable")
		err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, m.cs, pod.Name, claim.Namespace)
		framework.ExpectNoError(err, "Pod did not become Unschedulable")

		ginkgo.By("Waiting for Unused=False since unscheduled non-terminal pods still count as using the PVC")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False condition for unschedulable pod")

		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("PodUsingPVC"))
	})

	ginkgo.It("should track Unused correctly with multiple pods", func(ctx context.Context) {
		m.init(ctx, testParameters{registerDriver: true})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a PVC")
		_, claim := m.createPVC(ctx)

		ginkgo.By("Creating Pod1 that uses the PVC")
		pod1, err := m.createPodWithPVC(claim)
		framework.ExpectNoError(err, "Error creating Pod1")
		err = e2epod.WaitForPodRunningInNamespace(ctx, m.cs, pod1)
		framework.ExpectNoError(err, "Error waiting for Pod1 to be Running")

		ginkgo.By("Verifying PVC has Unused=False with Pod1 running")
		_, err = waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=False with Pod1")

		ginkgo.By("Creating Pod2 that also uses the same PVC")
		pod2, err := m.createPodWithPVC(claim)
		framework.ExpectNoError(err, "Error creating Pod2")
		err = e2epod.WaitForPodRunningInNamespace(ctx, m.cs, pod2)
		framework.ExpectNoError(err, "Error waiting for Pod2 to be Running")

		ginkgo.By("Deleting Pod1")
		err = e2epod.DeletePodWithWait(ctx, m.cs, pod1)
		framework.ExpectNoError(err, "Error deleting Pod1")

		ginkgo.By("Verifying PVC is still Unused=False because Pod2 is still using it")
		pvc, err := waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionFalse, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "PVC should still be Unused=False with Pod2 running")
		condition := findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Status).To(gomega.Equal(v1.ConditionFalse))

		ginkgo.By("Deleting Pod2")
		err = e2epod.DeletePodWithWait(ctx, m.cs, pod2)
		framework.ExpectNoError(err, "Error deleting Pod2")

		ginkgo.By("Verifying PVC transitions to Unused=True after all pods are deleted")
		pvc, err = waitForPVCUnusedCondition(ctx, m.cs, claim.Namespace, claim.Name, v1.ConditionTrue, f.Timeouts.ClaimUnusedCondition)
		framework.ExpectNoError(err, "Timed out waiting for Unused=True after deleting all pods")
		condition = findPVCCondition(pvc, v1.PersistentVolumeClaimUnused)
		gomega.Expect(condition).ToNot(gomega.BeNil())
		gomega.Expect(condition.Reason).To(gomega.Equal("NoPodsUsingPVC"))
	})
})
