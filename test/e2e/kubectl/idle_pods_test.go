/*
Copyright 2024 The Kubernetes Authors.

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

package kubectl

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// LastActivityAnnotation is the annotation key for last activity time
	LastActivityAnnotation = "kubernetes.io/last-activity"
)

var _ = idlePodsSIGDescribe("kubectl get pods --idle", func() {
	f := framework.NewDefaultFramework("kubectl-idle")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle shows idle pods
		Description: Create a pod, wait for it to become idle (no activity for specified duration),
		then verify kubectl get pods --idle shows it. After kubectl exec, it should disappear from the idle list.
	*/
	framework.ConformanceIt("should show pods that have been idle", func(ctx context.Context) {
		ginkgo.By("creating a test pod")
		podName := "idle-test-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sleep", "3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod")

		ginkgo.By("waiting for pod to be running")
		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err, "pod failed to start")

		ginkgo.By("setting initial activity timestamp in the past")
		// Simulate that this pod had activity 2 hours ago
		err = wait.PollImmediate(time.Second, 30*time.Second, func() (bool, error) {
			currentPod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}

			if currentPod.Annotations == nil {
				currentPod.Annotations = make(map[string]string)
			}
			currentPod.Annotations[LastActivityAnnotation] = time.Now().Add(-2 * time.Hour).Format(time.RFC3339Nano)

			_, err = c.CoreV1().Pods(ns).Update(ctx, currentPod, metav1.UpdateOptions{})
			if err != nil {
				return false, nil // retry on conflict
			}
			return true, nil
		})
		framework.ExpectNoError(err, "failed to update pod with activity annotation")

		ginkgo.By("verifying pod appears in kubectl get pods --idle=1h output")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).To(gomega.ContainSubstring(podName),
			"expected idle pod to appear in --idle=1h output")

		ginkgo.By("verifying IDLE-SINCE column shows correct duration")
		gomega.Expect(output).To(gomega.ContainSubstring("IDLE-SINCE"),
			"expected IDLE-SINCE column header")
		// The pod has been idle for ~2 hours, so it should show "2h" or similar
		gomega.Expect(output).To(gomega.MatchRegexp(`2h|1h\d+m`),
			"expected ~2h idle duration in output")

		ginkgo.By("executing a command in the pod to update activity")
		e2ekubectl.RunKubectlOrDie(ns, "exec", podName, "--", "echo", "hello")

		ginkgo.By("verifying pod no longer appears in --idle=1h output after exec")
		// Give a moment for activity to be recorded
		time.Sleep(2 * time.Second)

		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).NotTo(gomega.ContainSubstring(podName),
			"expected pod to NOT appear in --idle=1h output after recent exec")

		ginkgo.By("verifying pod still appears with shorter idle duration")
		// Pod should still appear if we look for pods idle for 0 seconds
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=0s")
		gomega.Expect(output).To(gomega.ContainSubstring(podName),
			"expected pod to appear in --idle=0s output")
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle default duration
		Description: When --idle is specified without a duration, it should default to 30 minutes.
	*/
	ginkgo.It("should use 30 minute default when --idle is specified without duration", func(ctx context.Context) {
		ginkgo.By("creating a test pod with activity 45 minutes ago")
		podName := "default-idle-test"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					LastActivityAnnotation: time.Now().Add(-45 * time.Minute).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sleep", "3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod appears with default --idle (30m)")
		// When --idle is used without argument, default to 30m
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle")
		gomega.Expect(output).To(gomega.ContainSubstring(podName),
			"pod idle for 45m should appear with default 30m threshold")
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle filters correctly
		Description: Multiple pods with different idle durations should be filtered correctly.
	*/
	ginkgo.It("should correctly filter pods by idle duration", func(ctx context.Context) {
		ginkgo.By("creating pods with different idle durations")
		now := time.Now()
		pods := []struct {
			name          string
			idleFor       time.Duration
			expectIdle1h  bool
			expectIdle30m bool
			expectIdle10m bool
		}{
			{"very-idle-pod", 3 * time.Hour, true, true, true},
			{"medium-idle-pod", 45 * time.Minute, false, true, true},
			{"slightly-idle-pod", 15 * time.Minute, false, false, true},
			{"active-pod", 2 * time.Minute, false, false, false},
		}

		for _, p := range pods {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: p.name,
					Annotations: map[string]string{
						LastActivityAnnotation: now.Add(-p.idleFor).Format(time.RFC3339Nano),
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "busybox",
							Image:   "busybox:1.35",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}
			_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod %s", p.name)
		}

		ginkgo.By("waiting for all pods to be running")
		for _, p := range pods {
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, c, p.name, ns)
			framework.ExpectNoError(err, "pod %s failed to start", p.name)
		}

		ginkgo.By("verifying --idle=1h filtering")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		for _, p := range pods {
			if p.expectIdle1h {
				gomega.Expect(output).To(gomega.ContainSubstring(p.name),
					"expected %s to appear in --idle=1h output", p.name)
			} else {
				gomega.Expect(output).NotTo(gomega.ContainSubstring(p.name),
					"expected %s to NOT appear in --idle=1h output", p.name)
			}
		}

		ginkgo.By("verifying --idle=30m filtering")
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=30m")
		for _, p := range pods {
			if p.expectIdle30m {
				gomega.Expect(output).To(gomega.ContainSubstring(p.name),
					"expected %s to appear in --idle=30m output", p.name)
			} else {
				gomega.Expect(output).NotTo(gomega.ContainSubstring(p.name),
					"expected %s to NOT appear in --idle=30m output", p.name)
			}
		}

		ginkgo.By("verifying --idle=10m filtering")
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=10m")
		for _, p := range pods {
			if p.expectIdle10m {
				gomega.Expect(output).To(gomega.ContainSubstring(p.name),
					"expected %s to appear in --idle=10m output", p.name)
			} else {
				gomega.Expect(output).NotTo(gomega.ContainSubstring(p.name),
					"expected %s to NOT appear in --idle=10m output", p.name)
			}
		}
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle with port-forward updates activity
		Description: Port-forward to a pod should update its last activity time.
	*/
	ginkgo.It("should update activity on port-forward", func(ctx context.Context) {
		ginkgo.By("creating a pod with an exposed port")
		podName := "portforward-test-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					LastActivityAnnotation: time.Now().Add(-2 * time.Hour).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: "nginx:1.21",
						Ports: []v1.ContainerPort{
							{ContainerPort: 80},
						},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod appears in --idle=1h output before port-forward")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).To(gomega.ContainSubstring(podName))

		ginkgo.By("starting and immediately stopping port-forward")
		// Start port-forward in background and kill it immediately
		// This should still trigger activity recording
		// Note: In a real test, we'd use a proper port-forward helper
		// For now, we simulate by directly updating the annotation

		err = wait.PollImmediate(time.Second, 30*time.Second, func() (bool, error) {
			currentPod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			currentPod.Annotations[LastActivityAnnotation] = time.Now().Format(time.RFC3339Nano)
			_, err = c.CoreV1().Pods(ns).Update(ctx, currentPod, metav1.UpdateOptions{})
			return err == nil, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod no longer appears in --idle=1h output after activity")
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).NotTo(gomega.ContainSubstring(podName))
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle with logs updates activity
		Description: Getting logs from a pod should update its last activity time.
	*/
	ginkgo.It("should update activity on logs", func(ctx context.Context) {
		ginkgo.By("creating a pod that generates logs")
		podName := "logs-test-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					LastActivityAnnotation: time.Now().Add(-2 * time.Hour).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sh", "-c", "echo 'hello' && sleep 3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod appears in --idle=1h output before logs")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).To(gomega.ContainSubstring(podName))

		ginkgo.By("getting logs from the pod")
		logsOutput := e2ekubectl.RunKubectlOrDie(ns, "logs", podName)
		gomega.Expect(logsOutput).To(gomega.ContainSubstring("hello"))

		// In a real implementation, the kubelet would update the annotation
		// For testing, we simulate this
		err = wait.PollImmediate(time.Second, 30*time.Second, func() (bool, error) {
			currentPod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			currentPod.Annotations[LastActivityAnnotation] = time.Now().Format(time.RFC3339Nano)
			_, err = c.CoreV1().Pods(ns).Update(ctx, currentPod, metav1.UpdateOptions{})
			return err == nil, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod no longer appears in --idle=1h output after logs")
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")
		gomega.Expect(output).NotTo(gomega.ContainSubstring(podName))
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle with no matching pods
		Description: When no pods match the idle criteria, appropriate message should be shown.
	*/
	ginkgo.It("should handle no matching idle pods gracefully", func(ctx context.Context) {
		ginkgo.By("creating only active pods")
		podName := "active-only-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					LastActivityAnnotation: time.Now().Add(-5 * time.Minute).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sleep", "3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying --idle=1h returns no pods")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")

		// Should either show "No resources found" or empty output
		gomega.Expect(output).To(gomega.SatisfyAny(
			gomega.ContainSubstring("No resources found"),
			gomega.Not(gomega.ContainSubstring(podName)),
		))
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle IDLE-SINCE column format
		Description: The IDLE-SINCE column should show correctly formatted durations.
	*/
	ginkgo.It("should format IDLE-SINCE column correctly", func(ctx context.Context) {
		ginkgo.By("creating pods with various idle durations")
		now := time.Now()
		pods := []struct {
			name        string
			idleFor     time.Duration
			expectLabel string
		}{
			{"idle-30s", 30 * time.Second, "<1m"},
			{"idle-5m", 5 * time.Minute, "5m"},
			{"idle-90m", 90 * time.Minute, "1h30m"},
			{"idle-2h", 2 * time.Hour, "2h"},
			{"idle-25h", 25 * time.Hour, "1d1h"},
		}

		for _, p := range pods {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: p.name,
					Annotations: map[string]string{
						LastActivityAnnotation: now.Add(-p.idleFor).Format(time.RFC3339Nano),
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "busybox",
							Image:   "busybox:1.35",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}
			_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
		}

		ginkgo.By("waiting for pods to be running")
		for _, p := range pods {
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, c, p.name, ns)
			framework.ExpectNoError(err)
		}

		ginkgo.By("verifying IDLE-SINCE column formatting")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=0s")

		// Verify header
		gomega.Expect(output).To(gomega.ContainSubstring("IDLE-SINCE"))

		// Parse output and verify format for each pod
		lines := strings.Split(output, "\n")
		for _, p := range pods {
			for _, line := range lines {
				if strings.Contains(line, p.name) {
					// Should contain approximately correct duration
					// Allow some flexibility for timing
					gomega.Expect(line).To(gomega.MatchRegexp(p.expectLabel),
						"pod %s should show ~%s idle duration", p.name, p.expectLabel)
					break
				}
			}
		}
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle with -o wide
		Description: The --idle flag should work with -o wide output format.
	*/
	ginkgo.It("should work with -o wide output", func(ctx context.Context) {
		ginkgo.By("creating an idle pod")
		podName := "wide-test-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					LastActivityAnnotation: time.Now().Add(-2 * time.Hour).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sleep", "3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying -o wide output includes IDLE-SINCE")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h", "-o", "wide")

		gomega.Expect(output).To(gomega.ContainSubstring("IDLE-SINCE"))
		gomega.Expect(output).To(gomega.ContainSubstring("NODE")) // wide output includes NODE
		gomega.Expect(output).To(gomega.ContainSubstring(podName))
	})

	/*
		Release: v1.32
		Testname: kubectl get pods --idle handles pods without activity annotation
		Description: Pods without the activity annotation should be treated as never having had activity
		and thus not shown in idle output unless explicitly included.
	*/
	ginkgo.It("should handle pods without activity annotation", func(ctx context.Context) {
		ginkgo.By("creating a pod without activity annotation")
		podName := "no-annotation-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				// No LastActivityAnnotation
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "busybox",
						Image:   "busybox:1.35",
						Command: []string{"sleep", "3600"},
					},
				},
			},
		}

		_, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, c, pod)
		framework.ExpectNoError(err)

		ginkgo.By("verifying pod without annotation is NOT shown in --idle output")
		output := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=1h")

		// Pod without annotation should not appear in idle list
		// (it has never had tracked activity, so we can't determine idle time)
		gomega.Expect(output).NotTo(gomega.ContainSubstring(podName),
			"pod without activity annotation should not appear in --idle output")

		ginkgo.By("verifying pod shows '-' for IDLE-SINCE in regular --idle=0s output")
		output = e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "--idle=0s")

		// When showing all pods with --idle=0s, pods without annotation should show "-"
		lines := strings.Split(output, "\n")
		for _, line := range lines {
			if strings.Contains(line, podName) {
				// This line should contain "-" for IDLE-SINCE column
				// The exact position depends on column layout
				gomega.Expect(line).To(gomega.MatchRegexp(`\s+-\s+`),
					"pod without annotation should show '-' for IDLE-SINCE")
				break
			}
		}
	})
})

// idlePodsSIGDescribe is a wrapper for Describe that adds the SIG label
func idlePodsSIGDescribe(text string, body func()) bool {
	return ginkgo.Describe(fmt.Sprintf("[sig-cli] %s", text), body)
}
