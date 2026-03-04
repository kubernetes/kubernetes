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

package auth

import (
	"context"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("ContainerUlimits PSA Integration",
	framework.WithFeatureGate(features.ContainerUlimits),
	func() {
		f := framework.NewDefaultFramework("container-ulimits-psa")
		f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

		createNamespaceWithPSALevel := func(ctx context.Context, level admissionapi.Level) string {
			oldEnforce := f.NamespacePodSecurityEnforceLevel
			f.NamespacePodSecurityEnforceLevel = level
			ns, err := f.CreateNamespace(ctx, "container-ulimits", nil)
			f.NamespacePodSecurityEnforceLevel = oldEnforce
			framework.ExpectNoError(err)
			return ns.Name
		}

		newPodWithUlimits := func(namespace, name string, ulimits []v1.Ulimit) *v1.Pod {
			containerSecurityContext := e2epod.GetRestrictedContainerSecurityContext()
			containerSecurityContext.Ulimits = ulimits

			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: namespace,
					Name:      name,
				},
				Spec: v1.PodSpec{
					SecurityContext: e2epod.GetRestrictedPodSecurityContext(),
					RestartPolicy:   v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:            "main",
							Image:           imageutils.GetE2EImage(imageutils.BusyBox),
							Command:         []string{"sh", "-c", "echo ok"},
							SecurityContext: containerSecurityContext,
						},
					},
				},
			}
		}

		ginkgo.It("should allow valid container ulimits in privileged PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelPrivileged)
			pod := newPodWithUlimits(ns, "ulimits-privileged-allow", []v1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should reject invalid container ulimit values in privileged PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelPrivileged)
			pod := newPodWithUlimits(ns, "ulimits-privileged-reject-invalid", []v1.Ulimit{{Name: "nproc", Soft: 1024, Hard: 2048}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsInvalid(err)).To(gomega.BeTrueBecause("expected API validation to reject invalid ulimit name"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})

		ginkgo.It("should reject container ulimits in baseline PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelBaseline)
			pod := newPodWithUlimits(ns, "ulimits-baseline-reject", []v1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsForbidden(err)).To(gomega.BeTrueBecause("expected PSA baseline policy to reject pod"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})

		ginkgo.It("should reject container ulimits in restricted PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelRestricted)
			pod := newPodWithUlimits(ns, "ulimits-restricted-reject", []v1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsForbidden(err)).To(gomega.BeTrueBecause("expected PSA restricted policy to reject pod"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})
	})
