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
	"k8s.io/utils/ptr"
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

		newPodWithUlimits := func(namespace, name string, ulimits *v1.Ulimits) *v1.Pod {
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
			pod := newPodWithUlimits(ns, "ulimits-privileged-allow", &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](1024), Hard: ptr.To[int64](2048)}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should reject invalid container ulimit configuration in privileged PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelPrivileged)
			pod := newPodWithUlimits(ns, "ulimits-privileged-reject-invalid", &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](255), Hard: ptr.To[int64](256)}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsInvalid(err)).To(gomega.BeTrueBecause("expected API validation to reject invalid ulimit values"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})

		ginkgo.It("should allow boundary ulimit values in privileged PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelPrivileged)

			minPod := newPodWithUlimits(ns, "ulimits-privileged-min-boundary", &v1.Ulimits{
				Nofile:  &v1.Ulimit{Soft: ptr.To[int64](256), Hard: ptr.To[int64](256)},
				Nice:    &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)},
				Rtprio:  &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)},
				Stack:   &v1.Ulimit{Soft: ptr.To[int64](262144), Hard: ptr.To[int64](262144)},
				Memlock: &v1.Ulimit{Soft: ptr.To[int64](8192), Hard: ptr.To[int64](8192)},
				Core:    &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)},
			})
			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, minPod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err)

			maxPod := newPodWithUlimits(ns, "ulimits-privileged-max-boundary", &v1.Ulimits{
				Nofile:  &v1.Ulimit{Soft: ptr.To[int64](65536), Hard: ptr.To[int64](65536)},
				Nice:    &v1.Ulimit{Soft: ptr.To[int64](40), Hard: ptr.To[int64](40)},
				Rtprio:  &v1.Ulimit{Soft: ptr.To[int64](99), Hard: ptr.To[int64](99)},
				Stack:   &v1.Ulimit{Soft: ptr.To[int64](17179869184), Hard: ptr.To[int64](17179869184)},
				Memlock: &v1.Ulimit{Soft: ptr.To[int64](17179869184), Hard: ptr.To[int64](17179869184)},
				Core:    &v1.Ulimit{Soft: ptr.To[int64](17179869184), Hard: ptr.To[int64](17179869184)},
			})
			_, err = f.ClientSet.CoreV1().Pods(ns).Create(ctx, maxPod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err)

			unlimitedPod := newPodWithUlimits(ns, "ulimits-privileged-unlimited", &v1.Ulimits{
				Nofile:  &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
				Nice:    &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
				Rtprio:  &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
				Stack:   &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
				Memlock: &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
				Core:    &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)},
			})
			_, err = f.ClientSet.CoreV1().Pods(ns).Create(ctx, unlimitedPod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should reject out-of-range ulimit values in privileged PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelPrivileged)
			testCases := []struct {
				name    string
				ulimits *v1.Ulimits
			}{
				{
					name:    "nofile-below-min",
					ulimits: &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](255), Hard: ptr.To[int64](256)}},
				},
				{
					name:    "nofile-above-max",
					ulimits: &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](65536), Hard: ptr.To[int64](65537)}},
				},
				{
					name:    "nice-above-max",
					ulimits: &v1.Ulimits{Nice: &v1.Ulimit{Soft: ptr.To[int64](40), Hard: ptr.To[int64](41)}},
				},
				{
					name:    "rtprio-above-max",
					ulimits: &v1.Ulimits{Rtprio: &v1.Ulimit{Soft: ptr.To[int64](99), Hard: ptr.To[int64](100)}},
				},
				{
					name:    "stack-below-min",
					ulimits: &v1.Ulimits{Stack: &v1.Ulimit{Soft: ptr.To[int64](262143), Hard: ptr.To[int64](262144)}},
				},
				{
					name:    "memlock-below-min",
					ulimits: &v1.Ulimits{Memlock: &v1.Ulimit{Soft: ptr.To[int64](8191), Hard: ptr.To[int64](8192)}},
				},
				{
					name:    "core-above-max",
					ulimits: &v1.Ulimits{Core: &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](17179869185)}},
				},
			}

			for _, tc := range testCases {
				pod := newPodWithUlimits(ns, "ulimits-privileged-reject-"+tc.name, tc.ulimits)
				_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
				gomega.Expect(err).To(gomega.HaveOccurred(), "expected invalid ulimit values to be rejected for case %s", tc.name)
				gomega.Expect(apierrors.IsInvalid(err)).To(gomega.BeTrueBecause("expected API validation to reject ulimit values for case %s", tc.name))
				gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
			}
		})

		ginkgo.It("should reject container ulimits in baseline PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelBaseline)
			pod := newPodWithUlimits(ns, "ulimits-baseline-reject", &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](1024), Hard: ptr.To[int64](2048)}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsForbidden(err)).To(gomega.BeTrueBecause("expected PSA baseline policy to reject pod"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})

		ginkgo.It("should reject container ulimits in restricted PSA namespace", func(ctx context.Context) {
			ns := createNamespaceWithPSALevel(ctx, admissionapi.LevelRestricted)
			pod := newPodWithUlimits(ns, "ulimits-restricted-reject", &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](1024), Hard: ptr.To[int64](2048)}})

			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(apierrors.IsForbidden(err)).To(gomega.BeTrueBecause("expected PSA restricted policy to reject pod"))
			gomega.Expect(err.Error()).To(gomega.ContainSubstring("ulimits"))
		})
	})
