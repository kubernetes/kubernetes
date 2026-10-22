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
	"slices"
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
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	networkIsolatedPodsFeature = "NetworkIsolatedPods"
	isolatedServerPort         = "8080"
)

// networkIsolatedPod runs an HTTP server on loopback and becomes Ready once
// the server accepts connections, so tests can dial it right away.
func networkIsolatedPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			DefaultNetwork: new(v1.PodDefaultNetworkNone),
			Containers: []v1.Container{{
				Name:  "agnhost",
				Image: imageutils.GetE2EImage(imageutils.Agnhost),
				Args:  []string{"netexec", "--http-port=" + isolatedServerPort},
				ReadinessProbe: &v1.Probe{
					ProbeHandler:  v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"/agnhost", "connect", "--timeout=2s", "127.0.0.1:" + isolatedServerPort}}},
					PeriodSeconds: 1,
				},
			}},
		},
	}
}

// expectPodRejectedByAdmission creates an isolated pod and expects the kubelet
// to reject it with PodFeatureUnsupported instead of attaching it to the pod
// network. Pods in node e2e bypass the scheduler (spec.nodeName is set), so
// this exercises the kubelet admission path directly.
func expectPodRejectedByAdmission(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, name string) {
	ginkgo.GinkgoHelper()

	ginkgo.By("Verifying the node does not declare the feature")
	gomega.Eventually(ctx, func(ctx context.Context) []string {
		return getLocalNode(ctx, f).Status.DeclaredFeatures
	}).WithTimeout(2*time.Minute).WithPolling(2*time.Second).ShouldNot(gomega.ContainElement(networkIsolatedPodsFeature), "node status must stop declaring the feature")

	ginkgo.By("Creating a network isolated pod")
	pod := podClient.Create(ctx, networkIsolatedPod(name))

	ginkgo.By("Waiting for the pod to fail admission")
	err := e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, lifecycle.PodFeatureUnsupported, framework.PodStartShortTimeout)
	framework.ExpectNoError(err, "expected the pod to be rejected with reason %s", lifecycle.PodFeatureUnsupported)

	pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	gomega.Expect(pod.Status.Message).To(gomega.ContainSubstring(networkIsolatedPodsFeature))
	gomega.Expect(pod.Status.PodIP).To(gomega.BeEmpty(), "a rejected pod must not get a pod IP")
}

// setNetworkIsolatedPodsGate restarts the kubelet with the NetworkIsolatedPods gate
// set to enabled and restores the previous configuration at cleanup.
func setNetworkIsolatedPodsGate(ctx context.Context, f *framework.Framework, enabled bool) {
	ginkgo.GinkgoHelper()
	oldCfg, err := getCurrentKubeletConfig(ctx)
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = map[string]bool{}
	}
	newCfg.FeatureGates[string(features.NetworkIsolatedPods)] = enabled
	updateKubeletConfig(ctx, f, newCfg, true)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})
}

// These tests need a container runtime that reports the default_network_none
// CRI capability; the kubelet rejects the pods otherwise.
var _ = SIGDescribe("Network Isolated Pods", feature.NetworkIsolatedPodsSupport, framework.WithFeatureGate(features.NetworkIsolatedPods), func() {
	f := framework.NewDefaultFramework("network-isolated-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func(ctx context.Context) {
		podClient = e2epod.NewPodClient(f)
		node := getLocalNode(ctx, f)
		if !slices.Contains(node.Status.DeclaredFeatures, networkIsolatedPodsFeature) {
			e2eskipper.Skipf("node %s does not declare %s; the container runtime does not support default_network_none", node.Name, networkIsolatedPodsFeature)
		}
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
		for line := range strings.SplitSeq(strings.TrimSpace(stdout), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			gomega.Expect(line).To(gomega.ContainSubstring(" lo: "), "unexpected network interface in isolated sandbox: %s", line)
		}

		ginkgo.By("Verifying the server in the pod is reachable over loopback")
		// The readiness probe already dials it; this makes the check explicit.
		e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/agnhost", "connect", "--timeout=2s", "127.0.0.1:"+isolatedServerPort)
	})

	ginkgo.It("should get its resolver configuration from dnsConfig only", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod with the default dnsPolicy")
		pod := podClient.CreateSync(ctx, networkIsolatedPod("isolated-dns-default"))
		gomega.Expect(pod.Spec.DNSPolicy).To(gomega.Equal(v1.DNSNone), "dnsPolicy must default to None")

		ginkgo.By("Verifying /etc/resolv.conf is empty")
		stdout := e2epod.ExecShellInPod(ctx, f, pod.Name, "cat /etc/resolv.conf")
		gomega.Expect(strings.TrimSpace(stdout)).To(gomega.BeEmpty(), "resolv.conf must not fall back to the host configuration")

		ginkgo.By("Creating a network isolated pod with a loopback resolver in dnsConfig")
		pod = networkIsolatedPod("isolated-dns-loopback")
		pod.Spec.DNSConfig = &v1.PodDNSConfig{
			Nameservers: []string{"127.0.0.53"},
			Searches:    []string{"isolated.local"},
			Options:     []v1.PodDNSConfigOption{{Name: "ndots", Value: new("2")}},
		}
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying /etc/resolv.conf contains exactly the dnsConfig")
		stdout = e2epod.ExecShellInPod(ctx, f, pod.Name, "cat /etc/resolv.conf")
		gomega.Expect(stdout).To(gomega.ContainSubstring("nameserver 127.0.0.53"))
		gomega.Expect(stdout).To(gomega.ContainSubstring("search isolated.local"))
		gomega.Expect(stdout).To(gomega.ContainSubstring("options ndots:2"))
		for line := range strings.SplitSeq(strings.TrimSpace(stdout), "\n") {
			gomega.Expect(line).To(gomega.Or(
				gomega.HavePrefix("nameserver 127.0.0.53"),
				gomega.HavePrefix("search isolated.local"),
				gomega.HavePrefix("options ndots:2"),
			), "unexpected resolv.conf entry: %s", line)
		}
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
		// containerIDOf returns the container ID while the container has not
		// been restarted, and "" once it has.
		containerIDOf := func(ctx context.Context, name string) (string, error) {
			p, err := podClient.Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			if len(p.Status.ContainerStatuses) == 0 || p.Status.ContainerStatuses[0].RestartCount != 0 {
				return "", nil
			}
			return p.Status.ContainerStatuses[0].ContainerID, nil
		}

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
				return containerIDOf(ctx, pod.Name)
			}).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Equal(containerID))
		})

		ginkgo.It("should keep running isolated pods when the NetworkIsolatedPods gate is disabled", func(ctx context.Context) {
			ginkgo.By("Creating a network isolated pod")
			pod := podClient.CreateSync(ctx, networkIsolatedPod("isolated-gate-off"))
			containerID := pod.Status.ContainerStatuses[0].ContainerID
			gomega.Expect(containerID).NotTo(gomega.BeEmpty())

			ginkgo.By("Restarting the kubelet with the NetworkIsolatedPods gate disabled")
			setNetworkIsolatedPodsGate(ctx, f, false)
			gomega.Eventually(ctx, func(ctx context.Context) []string {
				return getLocalNode(ctx, f).Status.DeclaredFeatures
			}).WithTimeout(2*time.Minute).WithPolling(2*time.Second).ShouldNot(gomega.ContainElement(networkIsolatedPodsFeature), "node status must stop declaring the feature")

			// Declared features only gate new pods: a pod admitted before
			// the restart keeps running instead of being re-admitted and
			// failed with PodFeatureUnsupported.
			ginkgo.By("Verifying the pod keeps running")
			gomega.Consistently(ctx, func(ctx context.Context) (string, error) {
				return containerIDOf(ctx, pod.Name)
			}).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Equal(containerID))
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
			gomega.Expect(pod.Status.PodIP).To(gomega.BeEmpty())

			ginkgo.By("Verifying new isolated pods are rejected")
			expectPodRejectedByAdmission(ctx, f, podClient, "isolated-gate-off-new")
		})
	})
})

// Fail-closed admission does not need runtime support, so these tests run
// against any container runtime.
var _ = SIGDescribe("Network Isolated Pods admission", framework.WithFeatureGate(features.NetworkIsolatedPods), framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("network-isolated-node-admission")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should reject the pod when the container runtime does not support default_network_none", func(ctx context.Context) {
		node := getLocalNode(ctx, f)
		if slices.Contains(node.Status.DeclaredFeatures, networkIsolatedPodsFeature) {
			e2eskipper.Skipf("node %s declares %s; the container runtime supports default_network_none", node.Name, networkIsolatedPodsFeature)
		}
		expectPodRejectedByAdmission(ctx, f, podClient, "isolated-rejected-runtime")
	})

	ginkgo.Context("when the kubelet does not enable the NetworkIsolatedPods gate", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(features.NetworkIsolatedPods)] = false
		})

		ginkgo.It("should reject the pod instead of attaching it to the pod network", func(ctx context.Context) {
			expectPodRejectedByAdmission(ctx, f, podClient, "isolated-rejected-gate")
		})
	})
})
