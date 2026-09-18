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

package e2enode

import (
	"context"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Hermetic", feature.Hermetic, func() {
	f := framework.NewDefaultFramework("hermetic-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should run a hermetic pod without IP address", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-pod",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod")
		createdPod := podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying pod status")
		fetchedPod, err := podClient.Get(ctx, createdPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")

		gomega.Expect(fetchedPod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(fetchedPod.Status.PodIP).To(gomega.BeEmpty(), "podIP should be empty for hermetic pod")
		gomega.Expect(fetchedPod.Status.PodIPs).To(gomega.BeEmpty(), "podIPs should be empty for hermetic pod")
		gomega.Expect(fetchedPod.Spec.DNSPolicy).To(gomega.Equal(v1.DNSNone), "DNSPolicy should default to None")
		gomega.Expect(fetchedPod.Spec.EnableServiceLinks).To(gomega.Equal(ptr.To(false)), "EnableServiceLinks should default to false")
	})

	ginkgo.It("should only have loopback interface in hermetic pod sandbox", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-netns-node",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod")
		createdPod := podClient.CreateSync(ctx, pod)

		ginkgo.By("Checking network interfaces inside hermetic container")
		stdout := e2epod.ExecShellInPod(ctx, f, createdPod.Name, "ip -o link show")
		lines := strings.Split(strings.TrimSpace(stdout), "\n")
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if len(trimmed) == 0 {
				continue
			}
			gomega.Expect(trimmed).To(gomega.ContainSubstring("lo:"), "unexpected network interface in hermetic sandbox: %s", trimmed)
		}

		ginkgo.By("Verifying loopback connectivity inside hermetic container")
		stdout = e2epod.ExecShellInPod(ctx, f, createdPod.Name, "ping -c 1 -W 2 127.0.0.1")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"), "loopback ping failed")
	})

	ginkgo.It("should run exec probe on a hermetic pod", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-exec-probe-node",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{
								Command: []string{"echo", "healthy"},
							},
						},
						InitialDelaySeconds: 1,
						PeriodSeconds:       2,
					},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod with Exec probe")
		createdPod := podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying container restart count remains 0")
		time.Sleep(5 * time.Second)
		fetchedPod, err := podClient.Get(ctx, createdPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")
		gomega.Expect(fetchedPod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		gomega.Expect(fetchedPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
	})
})
