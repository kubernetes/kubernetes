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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

func networkIsolatedPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone),
			Containers: []v1.Container{{
				Name:    "agnhost",
				Image:   imageutils.GetE2EImage(imageutils.Agnhost),
				Command: []string{"sleep", "3600"},
			}},
		},
	}
}

// The tests require a container runtime that reports the default_network_none
// capability; otherwise the kubelet rejects the pods with PodFeatureUnsupported.
var _ = SIGDescribe("Network Isolated Pods", feature.PodDefaultNetwork, framework.WithFeatureGate(features.PodDefaultNetwork), func() {
	f := framework.NewDefaultFramework("network-isolated-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should create the sandbox without a pod IP and with only a loopback interface", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod")
		pod := podClient.CreateSync(ctx, networkIsolatedPod("isolated"))

		ginkgo.By("Verifying the pod status")
		gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(pod.Status.PodIP).To(gomega.BeEmpty(), "status.podIP must be empty")
		gomega.Expect(pod.Status.PodIPs).To(gomega.BeEmpty(), "status.podIPs must be empty")
		gomega.Expect(pod.Status.HostIP).NotTo(gomega.BeEmpty(), "status.hostIP must be reported")
		gomega.Expect(pod.Status.Conditions).To(gomega.ContainElement(gomega.And(
			gomega.HaveField("Type", v1.PodReadyToStartContainers),
			gomega.HaveField("Status", v1.ConditionTrue),
		)), "PodReadyToStartContainers must be true without a pod IP")

		ginkgo.By("Verifying only the loopback interface exists in the sandbox")
		stdout := e2epod.ExecShellInPod(ctx, f, pod.Name, "ip -o link show")
		for _, line := range strings.Split(strings.TrimSpace(stdout), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			gomega.Expect(line).To(gomega.ContainSubstring(" lo: "), "unexpected network interface in isolated sandbox: %s", line)
		}
		stdout = e2epod.ExecShellInPod(ctx, f, pod.Name, "ping -c 1 -W 2 127.0.0.1")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"))
	})

	ginkgo.It("should run exec probes", func(ctx context.Context) {
		pod := networkIsolatedPod("isolated-exec-probe")
		pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
			ProbeHandler:        v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}},
			InitialDelaySeconds: 1,
			PeriodSeconds:       1,
		}

		ginkgo.By("Creating a network isolated pod with an exec liveness probe")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying the container is not restarted")
		gomega.Consistently(ctx, func(ctx context.Context) (int32, error) {
			p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return 0, err
			}
			return p.Status.ContainerStatuses[0].RestartCount, nil
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(gomega.BeZero())
	})

	ginkgo.Context("when the kubelet restarts", framework.WithSerial(), framework.WithDisruptive(), func() {
		ginkgo.It("should keep the sandbox without a pod IP", func(ctx context.Context) {
			ginkgo.By("Creating a network isolated pod")
			pod := podClient.CreateSync(ctx, networkIsolatedPod("isolated-restart"))
			containerID := pod.Status.ContainerStatuses[0].ContainerID
			gomega.Expect(containerID).NotTo(gomega.BeEmpty())

			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx, true)
			waitForKubeletToStart(ctx, f)

			// PodSandboxChanged must not recreate the sandbox because it has
			// no IP; the container keeps running with the same ID.
			ginkgo.By("Verifying the container was not recreated")
			gomega.Consistently(ctx, func(ctx context.Context) (string, error) {
				p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return "", err
				}
				if p.Status.ContainerStatuses[0].RestartCount != 0 {
					return "", nil
				}
				return p.Status.ContainerStatuses[0].ContainerID, nil
			}).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Equal(containerID))
		})
	})

	// Pods in node e2e bypass the scheduler (spec.nodeName is set), so this
	// exercises the kubelet admission path directly.
	ginkgo.Context("when the kubelet does not declare the PodDefaultNetworkNone feature", framework.WithSerial(), framework.WithDisruptive(), func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(features.PodDefaultNetwork)] = false
		})

		ginkgo.It("should reject the pod instead of attaching it to the pod network", func(ctx context.Context) {
			ginkgo.By("Verifying the node does not declare the feature")
			node := getLocalNode(ctx, f)
			gomega.Expect(node.Status.DeclaredFeatures).NotTo(gomega.ContainElement("PodDefaultNetworkNone"))

			ginkgo.By("Creating a network isolated pod")
			pod := podClient.Create(ctx, networkIsolatedPod("isolated-rejected"))

			ginkgo.By("Waiting for the pod to fail admission")
			err := e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, lifecycle.PodFeatureUnsupported, framework.PodStartShortTimeout)
			framework.ExpectNoError(err, "expected the pod to be rejected with reason %s", lifecycle.PodFeatureUnsupported)

			pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Status.Message).To(gomega.ContainSubstring("PodDefaultNetworkNone"))
			gomega.Expect(pod.Status.PodIP).To(gomega.BeEmpty(), "a rejected pod must not get a pod IP")
		})
	})
})
