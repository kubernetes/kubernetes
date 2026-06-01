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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("GangScheduling", framework.WithFeatureGate(features.GangScheduling), func() {
	f := framework.NewDefaultFramework("gang-scheduling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	removePodGroup := func(ctx context.Context, pgName string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		ginkgo.By("Deleting PodGroup")
		err := cs.SchedulingV1alpha3().PodGroups(ns).Delete(ctx, pgName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PodGroup")
	}

	f.It("should schedule pods only when quorum is reached", func(ctx context.Context) {
		cs := f.ClientSet
		ns := f.Namespace.Name

		ginkgo.By("Creating a PodGroup with MinCount=2")
		pgName := "test-pg"
		pg := &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1alpha3.PodGroupSpec{
				SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
					Gang: &schedulingv1alpha3.GangSchedulingPolicy{
						MinCount: 2,
					},
				},
			},
		}
		_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
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
		pg := &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1alpha3.PodGroupSpec{
				SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
					Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
				},
			},
		}
		_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
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
})
