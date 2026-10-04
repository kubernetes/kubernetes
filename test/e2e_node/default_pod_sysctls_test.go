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

package e2enode

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("DefaultPodSysctls [LinuxOnly]", framework.WithSerial(), framework.WithDisruptive(), framework.WithFeatureGate(features.DefaultPodSysctls), func() {
	f := framework.NewDefaultFramework("default-pod-sysctls-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("with DefaultPodSysctls configured", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.DefaultPodSysctls)] = true
			initialConfig.DefaultPodSysctls = map[string]string{
				"net.ipv4.ip_forward":    "1",
				"kernel.shm_rmid_forced": "1",
			}
		})

		ginkgo.It("should apply default sysctls to pods", func(ctx context.Context) {
			ginkgo.By("creating a pod without sysctls")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-default-sysctls",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   busyboxImage,
							Command: []string{"sh", "-c", "sysctl net.ipv4.ip_forward && sysctl kernel.shm_rmid_forced"},
						},
					},
				},
			}
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("checking pod logs for default sysctl values")
			gomega.Eventually(ctx, func() (string, error) {
				return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
			}, 1*time.Minute, 2*time.Second).Should(gomega.And(
				gomega.ContainSubstring("net.ipv4.ip_forward = 1"),
				gomega.ContainSubstring("kernel.shm_rmid_forced = 1"),
			))
		})

		ginkgo.It("should allow pod-level securityContext to override default sysctls", func(ctx context.Context) {
			ginkgo.By("creating a pod with overriding sysctl in securityContext")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-override-sysctls",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					SecurityContext: &v1.PodSecurityContext{
						Sysctls: []v1.Sysctl{
							{
								Name:  "kernel.shm_rmid_forced",
								Value: "0",
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   busyboxImage,
							Command: []string{"sh", "-c", "sysctl net.ipv4.ip_forward && sysctl kernel.shm_rmid_forced"},
						},
					},
				},
			}
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("checking pod logs for overridden sysctl values")
			gomega.Eventually(ctx, func() (string, error) {
				return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
			}, 1*time.Minute, 2*time.Second).Should(gomega.And(
				gomega.ContainSubstring("net.ipv4.ip_forward = 1"),    // inherited
				gomega.ContainSubstring("kernel.shm_rmid_forced = 0"), // overridden
			))
		})
	})
})
