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

package scheduling

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("GangScheduling", framework.WithFeatureGate(features.GenericWorkload), func() {
	f := framework.NewDefaultFramework("gang-scheduling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	removePodGroup := func(ctx context.Context, pgName string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		ginkgo.By("Deleting PodGroup")
		err := cs.SchedulingV1beta1().PodGroups(ns).Delete(ctx, pgName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PodGroup")
	}

	f.It("should schedule pods only when quorum is reached", func(ctx context.Context) {
		cs := f.ClientSet
		ns := f.Namespace.Name

		ginkgo.By("Creating a PodGroup with MinCount=2")
		pgName := "test-pg"
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
					Gang: &schedulingv1beta1.GangSchedulingPolicy{
						MinCount: 2,
					},
				},
			},
		}
		_, err := cs.SchedulingV1beta1().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup")
		defer removePodGroup(ctx, pgName)

		ginkgo.By("Creating first pod in the gang")
		p1 := e2epod.MakePod(ns, nil, nil, admissionapi.LevelPrivileged, "")
		p1.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &pgName,
		}
		p1, err = cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod p1")

		ginkgo.By("Verifying pod p1 remains Pending")
		gomega.Consistently(ctx, func() v1.PodPhase {
			pod, err := cs.CoreV1().Pods(ns).Get(ctx, p1.Name, metav1.GetOptions{})
			if err != nil {
				return v1.PodUnknown
			}
			return pod.Status.Phase
		}, 10*time.Second, 1*time.Second).Should(gomega.Equal(v1.PodPending))

		ginkgo.By("Creating second pod in the gang")
		p2 := e2epod.MakePod(ns, nil, nil, admissionapi.LevelPrivileged, "")
		p2.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &pgName,
		}
		p2, err = cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod p2")

		ginkgo.By("Verifying both pods are scheduled")
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p1.Name, ns), "pod p1 failed to run")
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p2.Name, ns), "pod p2 failed to run")
	})

	f.It("should schedule pods with basic scheduling policy", func(ctx context.Context) {
		cs := f.ClientSet
		ns := f.Namespace.Name

		ginkgo.By("Creating a PodGroup with Basic policy")
		pgName := "test-pg"
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
					Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
				},
			},
		}
		_, err := cs.SchedulingV1beta1().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup")

		ginkgo.By("Creating first pod in the group")
		p1 := e2epod.MakePod(ns, nil, nil, admissionapi.LevelPrivileged, "")
		p1.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &pgName,
		}
		p1, err = cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod p1")

		ginkgo.By("Verifying first pod is scheduled immediately")
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p1.Name, ns), "pod p1 failed to run")

		ginkgo.By("Creating second pod in the group")
		p2 := e2epod.MakePod(ns, nil, nil, admissionapi.LevelPrivileged, "")
		p2.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &pgName,
		}
		p2, err = cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod p2")

		ginkgo.By("Verifying second pod is scheduled immediately")
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p2.Name, ns), "pod p2 failed to run")
	})

	f.It("should not preempt lower-priority pods when high-priority PodGroup has PreemptionPolicy: PreemptNever", framework.WithFeatureGate(features.PodGroupPreemptionPolicy), func(ctx context.Context) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		extendedResourceName := v1.ResourceName("example.com/" + ns)

		lowPriorityName := "low-priority-" + ns
		highPriorityName := "high-priority-never-" + ns

		lowPC := &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
			Value:      100,
		}
		_, err := cs.SchedulingV1().PriorityClasses().Create(ctx, lowPC, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			_ = cs.SchedulingV1().PriorityClasses().Delete(ctx, lowPriorityName, metav1.DeleteOptions{})
		}()

		preemptNever := v1.PreemptNever
		highPC := &schedulingv1.PriorityClass{
			ObjectMeta:       metav1.ObjectMeta{Name: highPriorityName},
			Value:            1000,
			PreemptionPolicy: &preemptNever,
		}
		_, err = cs.SchedulingV1().PriorityClasses().Create(ctx, highPC, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			_ = cs.SchedulingV1().PriorityClasses().Delete(ctx, highPriorityName, metav1.DeleteOptions{})
		}()

		node, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err)
		e2enode.AddExtendedResource(ctx, cs, node.Name, extendedResourceName, resource.MustParse("2"))
		defer e2enode.RemoveExtendedResource(ctx, cs, node.Name, extendedResourceName)

		ginkgo.By("Creating low-priority running pods saturating capacity")
		var lowPods []*v1.Pod
		for i := 1; i <= 2; i++ {
			p := e2epod.MakePod(ns, map[string]string{"kubernetes.io/hostname": node.Name}, nil, admissionapi.LevelPrivileged, "")
			p.ObjectMeta.GenerateName = fmt.Sprintf("low-pod-%d-", i)
			p.Spec.PriorityClassName = lowPriorityName
			p.Spec.Containers[0].Resources.Requests = v1.ResourceList{extendedResourceName: resource.MustParse("1")}
			p.Spec.Containers[0].Resources.Limits = v1.ResourceList{extendedResourceName: resource.MustParse("1")}
			createdPod, err := cs.CoreV1().Pods(ns).Create(ctx, p, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			lowPods = append(lowPods, createdPod)
		}

		for _, p := range lowPods {
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
		}

		ginkgo.By("Creating high-priority gang PodGroup with PreemptionPolicy: PreemptNever")
		pgNeverPolicy := schedulingv1beta1.PreemptNever
		pgName := "hp-never-pg"
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1beta1.PodGroupSpec{
				PriorityClassName: highPriorityName,
				PreemptionPolicy:  &pgNeverPolicy,
				SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
					Gang: &schedulingv1beta1.GangSchedulingPolicy{
						MinCount: 2,
					},
				},
			},
		}
		_, err = cs.SchedulingV1beta1().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer removePodGroup(ctx, pgName)

		ginkgo.By("Creating high-priority gang pods with PreemptionPolicy: PreemptNever")
		var hpPods []*v1.Pod
		for i := 1; i <= 2; i++ {
			p := e2epod.MakePod(ns, map[string]string{"kubernetes.io/hostname": node.Name}, nil, admissionapi.LevelPrivileged, "")
			p.ObjectMeta.GenerateName = fmt.Sprintf("hp-never-pod-%d-", i)
			p.Spec.PriorityClassName = highPriorityName
			p.Spec.PreemptionPolicy = &preemptNever
			p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
				PodGroupName: &pgName,
			}
			p.Spec.Containers[0].Resources.Requests = v1.ResourceList{extendedResourceName: resource.MustParse("1")}
			p.Spec.Containers[0].Resources.Limits = v1.ResourceList{extendedResourceName: resource.MustParse("1")}
			createdPod, err := cs.CoreV1().Pods(ns).Create(ctx, p, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			hpPods = append(hpPods, createdHPod(createdPod))
		}

		ginkgo.By("Verifying high-priority gang pods remain Pending and fail to schedule")
		gomega.Consistently(ctx, func() bool {
			for _, p := range hpPods {
				pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
				if err != nil || pod.Status.Phase != v1.PodPending {
					return false
				}
			}
			return true
		}, 10*time.Second, 1*time.Second).Should(gomega.BeTrue(), "High priority gang pods with PreemptNever should remain Pending")

		ginkgo.By("Verifying no low-priority pods were preempted")
		for _, p := range lowPods {
			pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.DeletionTimestamp).To(gomega.BeNil(), "Low priority pod must not be preempted")
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		}
	})
})

func createdHPod(p *v1.Pod) *v1.Pod {
	return p
}
