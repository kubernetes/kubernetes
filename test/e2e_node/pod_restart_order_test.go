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

package e2enode

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Pod Restart with PodStartingOrderByPriority Featuregate", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pod-restart-order-by-priority")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	const (
		// podAmount is the number of pods created per priority class.
		podAmount = 50
		// podTotal is the total number of pods across both priority classes.
		podTotal = podAmount * 2
	)

	var (
		customPriorityClassHigh = newPriorityClass("high-priority", 1000000)
		customPriorityClassLow  = newPriorityClass("low-priority", 100)
	)

	// setupPriorityClasses creates the custom priority classes and waits for them to be available.
	setupPriorityClasses := func(ctx context.Context) {
		ginkgo.By("Wait for node to be ready")
		gomega.Expect(e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 5*time.Minute)).To(gomega.Succeed())
		ginkgo.By("Create custom priority classes")
		customClasses := []*schedulingv1.PriorityClass{customPriorityClassHigh, customPriorityClassLow}
		for _, customClass := range customClasses {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, customClass, metav1.CreateOptions{})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				framework.ExpectNoError(err)
			}
		}
		gomega.Eventually(ctx, func(ctx context.Context) error {
			for _, customClass := range customClasses {
				_, err := f.ClientSet.SchedulingV1().PriorityClasses().Get(ctx, customClass.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
			}
			return nil
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(gomega.Succeed())
	}

	// cleanupAllPods deletes all pods in the test namespace and waits for them to be gone.
	cleanupAllPods := func(ctx context.Context) {
		ginkgo.By("Cleaning up all test pods")
		err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).DeleteCollection(ctx,
			metav1.DeleteOptions{},
			metav1.ListOptions{})
		framework.ExpectNoError(err)
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			if err != nil {
				return err
			}
			if len(podList.Items) > 0 {
				return fmt.Errorf("still %d pods remaining", len(podList.Items))
			}
			return nil
		}).WithTimeout(2 * time.Minute).WithPolling(time.Second).Should(gomega.Succeed())
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		setupPriorityClasses(ctx)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		cleanupAllPods(ctx)
	})

	// runRestartOrderTest creates low-priority pods first (older creation timestamps),
	// then high-priority pods, simulates a node reboot, and returns the median CRI
	// sandbox creation timestamps for each priority class.
	runRestartOrderTest := func(ctx context.Context) (medianHigh, medianLow int64) {
		nodeName := getNodeName(ctx, f)
		nodeSelector := fields.Set{
			"spec.nodeName": nodeName,
		}.AsSelector().String()

		ginkgo.By("Create low-priority pods")
		lowPrioPods := []*v1.Pod{}
		for i := 0; i < podAmount; i++ {
			lowPrioPods = append(lowPrioPods, getPodWithPriorityAndResources(fmt.Sprintf("priority-low-%d", i), nodeName, customPriorityClassLow.Name, v1.ResourceRequirements{}))
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, lowPrioPods)

		ginkgo.By("Waiting for all low-priority pods to be running initially")
		gomega.Eventually(func() error {
			podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}
			for _, pod := range podList.Items {
				if pod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("pod %s not running yet", pod.Name)
				}
			}
			return nil
		}).WithTimeout(30 * time.Second).WithPolling(time.Second).Should(gomega.BeNil())

		ginkgo.By("Create high-priority pods once low priority pods are running")
		// Wait to ensure low-priority pods have older creation timestamps.
		time.Sleep(5 * time.Second)
		highPrioPods := []*v1.Pod{}
		for i := 0; i < podAmount; i++ {
			highPrioPods = append(highPrioPods, getPodWithPriorityAndResources(fmt.Sprintf("priority-high-%d", i), nodeName, customPriorityClassHigh.Name, v1.ResourceRequirements{}))
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, highPrioPods)

		ginkgo.By("Waiting for all pods to be running")
		gomega.Eventually(func() error {
			podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}
			for _, pod := range podList.Items {
				if pod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("pod %s not running yet", pod.Name)
				}
			}
			return nil
		}).WithTimeout(30 * time.Second).WithPolling(time.Second).Should(gomega.BeNil())

		ginkgo.By("Stopping the kubelet")
		restartKubelet := mustStopKubelet(ctx, f)

		ginkgo.By("Stop all containers using CRI")
		rs, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err)
		sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
		framework.ExpectNoError(err)
		for _, sandbox := range sandboxes {
			gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
			ginkgo.By(fmt.Sprintf("stopping pod sandbox using CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))
			err := rs.StopPodSandbox(ctx, sandbox.Id)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Restarting the kubelet")
		restartKubelet(ctx)

		ginkgo.By("Check pod restart order by using CRI sandbox creation timestamps")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			if err != nil {
				return err
			}
			runningCount := 0
			for _, pod := range podList.Items {
				if len(pod.Status.ContainerStatuses) > 0 &&
					pod.Status.ContainerStatuses[0].State.Running != nil &&
					pod.Status.ContainerStatuses[0].RestartCount >= 1 {
					runningCount++
				}
				// restart count should be at least 1
				if pod.Status.ContainerStatuses[0].RestartCount < 1 {
					framework.Logf("Expecting pod to have restarted at least once, but it has not. Pod: (%v/%v), Restart Count: %d", pod.Namespace, pod.Name, pod.Status.ContainerStatuses[0].RestartCount)
					return fmt.Errorf("pod should have restarted at least once, restart count: %d", pod.Status.ContainerStatuses[0].RestartCount)
				}
			}
			if runningCount < podTotal {
				return fmt.Errorf("waiting for all pods to reach Running state, current: %d/%d", runningCount, podTotal)
			}
			return nil
		}).WithTimeout(5 * time.Minute).WithPolling(time.Second).Should(gomega.BeNil())

		ginkgo.By("Querying CRI for sandbox creation timestamps (nanosecond precision)")
		// Re-acquire CRI client after kubelet restart.
		rs, _, err = getCRIClient(ctx)
		framework.ExpectNoError(err)
		sandboxes, err = rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{
			State: &runtimeapi.PodSandboxStateValue{
				State: runtimeapi.PodSandboxState_SANDBOX_READY,
			},
		})
		framework.ExpectNoError(err)

		// Collect CRI sandbox CreatedAt (nanoseconds) per pod name
		type sandboxInfo struct {
			name      string
			createdAt int64
			priority  string
		}
		var sandboxInfos []sandboxInfo

		ns := f.Namespace.Name
		for _, sb := range sandboxes {
			if sb.Metadata == nil {
				continue
			}
			if sb.Metadata.Namespace != ns {
				continue
			}
			name := sb.Metadata.Name
			if !strings.Contains(name, "priority-") {
				continue
			}
			prio := "low"
			if strings.Contains(name, "priority-high") {
				prio = "high"
			}
			sandboxInfos = append(sandboxInfos, sandboxInfo{
				name:      name,
				createdAt: sb.CreatedAt,
				priority:  prio,
			})
			framework.Logf("CRI sandbox %s (priority=%s) CreatedAt=%d ns (%v)",
				name, prio, sb.CreatedAt, time.Unix(0, sb.CreatedAt))
		}

		// Sort all sandboxes by CRI creation timestamp (nanoseconds)
		sort.Slice(sandboxInfos, func(i, j int) bool {
			return sandboxInfos[i].createdAt < sandboxInfos[j].createdAt
		})

		framework.Logf("Sandbox creation order (by CRI nanosecond timestamps):")
		for i, si := range sandboxInfos {
			framework.Logf("  %d. %s (priority=%s, created=%d)", i, si.name, si.priority, si.createdAt)
		}

		var highTimestamps []int64
		var lowTimestamps []int64

		for _, si := range sandboxInfos {
			switch si.priority {
			case "high":
				highTimestamps = append(highTimestamps, si.createdAt)
			case "low":
				lowTimestamps = append(lowTimestamps, si.createdAt)
			}
		}

		gomega.Expect(highTimestamps).To(gomega.HaveLen(podAmount), "expected all high-priority sandboxes")
		gomega.Expect(lowTimestamps).To(gomega.HaveLen(podAmount), "expected all low-priority sandboxes")

		sort.Slice(highTimestamps, func(i, j int) bool { return highTimestamps[i] < highTimestamps[j] })
		sort.Slice(lowTimestamps, func(i, j int) bool { return lowTimestamps[i] < lowTimestamps[j] })

		// Use median timestamps to compare start order. Median is more robust than
		// comparing min/max since a few pods may overlap due to concurrent sandbox creation.
		medianHigh = highTimestamps[len(highTimestamps)/2]
		medianLow = lowTimestamps[len(lowTimestamps)/2]

		framework.Logf("Median high-priority sandbox CreatedAt: %d ns (%v)", medianHigh, time.Unix(0, medianHigh))
		framework.Logf("Median low-priority sandbox CreatedAt:  %d ns (%v)", medianLow, time.Unix(0, medianLow))
		framework.Logf("Median difference (low - high): %d ns", medianLow-medianHigh)

		return medianHigh, medianLow
	}

	// runNewPodRejectionTest fills ~70% CPU with low-priority pods, stops kubelet,
	// creates a new high-priority pod requesting ~70% CPU, restarts kubelet, and
	// verifies that previously-running low-priority pods are re-admitted first
	// while the new high-priority pod fails admission due to insufficient CPU.
	runNewPodRejectionTest := func(ctx context.Context) {
		nodeName := getNodeName(ctx, f)

		ginkgo.By("Getting node allocatable CPU")
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
		allocatableCPU := nodeList.Items[0].Status.Allocatable[v1.ResourceCPU]
		cpuMillis := allocatableCPU.MilliValue()
		framework.Logf("Node allocatable CPU: %dm", cpuMillis)

		const numLowPrioPods = 10
		lowPrioCPUPerPod := (cpuMillis * 70) / (100 * numLowPrioPods)
		highPrioCPU := (cpuMillis * 70) / 100

		ginkgo.By("Creating low-priority pods that fill ~70% of CPU")
		lowPrioPods := []*v1.Pod{}
		for i := 0; i < numLowPrioPods; i++ {
			lowPrioPods = append(lowPrioPods, getPodWithPriorityAndResources(
				fmt.Sprintf("cpu-low-%d", i), nodeName, customPriorityClassLow.Name,
				v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(lowPrioCPUPerPod, resource.DecimalSI),
					},
				},
			))
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, lowPrioPods)

		ginkgo.By("Waiting for all low-priority pods to be running")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: fields.Set{"spec.nodeName": nodeName}.AsSelector().String(),
			})
			if err != nil {
				return err
			}
			for _, pod := range podList.Items {
				if pod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("pod %s not running yet (phase: %s)", pod.Name, pod.Status.Phase)
				}
			}
			return nil
		}).WithTimeout(60 * time.Second).WithPolling(time.Second).Should(gomega.Succeed())

		ginkgo.By("Stopping the kubelet")
		restartKubelet := mustStopKubelet(ctx, f)

		ginkgo.By("Stop all containers using CRI")
		rs, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err)
		sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
		framework.ExpectNoError(err)
		for _, sandbox := range sandboxes {
			gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
			ginkgo.By(fmt.Sprintf("stopping pod sandbox using CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))
			err := rs.StopPodSandbox(ctx, sandbox.Id)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Creating a high-priority pod via API while kubelet is stopped")
		highPrioPod := getPodWithPriorityAndResources(
			"cpu-high-0",
			nodeName,
			customPriorityClassHigh.Name,
			v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(highPrioCPU, resource.DecimalSI),
				},
			},
		)
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, highPrioPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create high-priority pod while kubelet is stopped")

		ginkgo.By("Restarting the kubelet")
		restartKubelet(ctx)

		ginkgo.By("Waiting for kubelet to process all pods")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			if err != nil {
				return err
			}
			lowRunning := 0
			for _, p := range podList.Items {
				if strings.HasPrefix(p.Name, "cpu-low-") && p.Status.Phase == v1.PodRunning {
					lowRunning++
				}
			}
			if lowRunning < numLowPrioPods {
				return fmt.Errorf("waiting for all low-priority pods to be running: %d/%d", lowRunning, numLowPrioPods)
			}
			for _, p := range podList.Items {
				if p.Name == "cpu-high-0" && p.Status.Phase == v1.PodFailed {
					return nil
				}
			}
			return fmt.Errorf("high-priority pod not failed yet")
		}).WithTimeout(5 * time.Minute).WithPolling(2 * time.Second).Should(gomega.Succeed())

		ginkgo.By("Verifying pod states")
		podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, p := range podList.Items {
			switch {
			case p.Name == "cpu-high-0":
				gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodFailed), "high-priority pod should be Failed")
				gomega.Expect(p.Status.Reason).To(gomega.Equal("OutOfcpu"),
					fmt.Sprintf("high-priority pod rejection reason should be OutOfcpu, got: %s", p.Status.Reason))
				gomega.Expect(p.Status.Message).To(gomega.HavePrefix("Pod was rejected: "),
					fmt.Sprintf("high-priority pod rejection message should start with 'Pod was rejected: ', got: %s", p.Status.Message))
				framework.Logf("High-priority pod %s is Failed. Reason: %s, Message: %s", p.Name, p.Status.Reason, p.Status.Message)
			case strings.HasPrefix(p.Name, "cpu-low-"):
				gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodRunning),
					fmt.Sprintf("Low-priority pod %s should still be running", p.Name))
			}
		}
	}

	// runCriticalPodPreemptionTest fills ~70% CPU with low-priority pods, stops kubelet,
	// creates a system-critical pod (system-node-critical, priority >= 2 billion)
	// requesting ~70% CPU, restarts kubelet, and verifies the critical pod admission
	// handler preempts enough low-priority pods to make room.
	runCriticalPodPreemptionTest := func(ctx context.Context) {
		nodeName := getNodeName(ctx, f)

		ginkgo.By("Getting node allocatable CPU")
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
		allocatableCPU := nodeList.Items[0].Status.Allocatable[v1.ResourceCPU]
		cpuMillis := allocatableCPU.MilliValue()
		framework.Logf("Node allocatable CPU: %dm", cpuMillis)

		const numLowPrioPods = 10
		lowPrioCPUPerPod := (cpuMillis * 70) / (100 * numLowPrioPods)
		criticalCPU := (cpuMillis * 70) / 100

		ginkgo.By("Creating low-priority pods that fill ~70% of CPU")
		lowPrioPods := []*v1.Pod{}
		for i := 0; i < numLowPrioPods; i++ {
			lowPrioPods = append(lowPrioPods, getPodWithPriorityAndResources(
				fmt.Sprintf("critical-low-%d", i), nodeName, customPriorityClassLow.Name,
				v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(lowPrioCPUPerPod, resource.DecimalSI),
					},
				},
			))
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, lowPrioPods)

		ginkgo.By("Waiting for all low-priority pods to be running")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{
				FieldSelector: fields.Set{"spec.nodeName": nodeName}.AsSelector().String(),
			})
			if err != nil {
				return err
			}
			for _, pod := range podList.Items {
				if pod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("pod %s not running yet (phase: %s)", pod.Name, pod.Status.Phase)
				}
			}
			return nil
		}).WithTimeout(60 * time.Second).WithPolling(time.Second).Should(gomega.Succeed())

		ginkgo.By("Stopping the kubelet")
		restartKubelet := mustStopKubelet(ctx, f)

		ginkgo.By("Stop all containers using CRI")
		rs, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err)
		sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
		framework.ExpectNoError(err)
		for _, sandbox := range sandboxes {
			gomega.Expect(sandbox.Metadata).ToNot(gomega.BeNil())
			ginkgo.By(fmt.Sprintf("stopping pod sandbox using CRI: %s/%s -> %s", sandbox.Metadata.Namespace, sandbox.Metadata.Name, sandbox.Id))
			err := rs.StopPodSandbox(ctx, sandbox.Id)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Creating a system-critical pod via API while kubelet is stopped")
		criticalPod := getPodWithPriorityAndResources(
			"critical-pod-0",
			nodeName,
			"system-node-critical",
			v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(criticalCPU, resource.DecimalSI),
				},
			},
		)
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, criticalPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create critical pod while kubelet is stopped")

		ginkgo.By("Restarting the kubelet")
		restartKubelet(ctx)

		ginkgo.By("Waiting for the critical pod to be running")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			if err != nil {
				return err
			}
			for _, p := range podList.Items {
				if p.Name == "critical-pod-0" && p.Status.Phase == v1.PodRunning {
					return nil
				}
			}
			return fmt.Errorf("critical pod not running yet")
		}).WithTimeout(5 * time.Minute).WithPolling(2 * time.Second).Should(gomega.Succeed())

		ginkgo.By("Verifying pod states")
		podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)

		failedCount := 0
		for _, p := range podList.Items {
			if strings.HasPrefix(p.Name, "critical-low-") && p.Status.Phase == v1.PodFailed {
				failedCount++
				gomega.Expect(p.Status.Reason).To(gomega.Equal("Preempting"),
					fmt.Sprintf("preempted pod %s reason should be Preempting, got: %s", p.Name, p.Status.Reason))
				gomega.Expect(p.Status.Message).To(gomega.Equal("Preempted in order to admit critical pod"),
					fmt.Sprintf("preempted pod %s message should match, got: %s", p.Name, p.Status.Message))
				framework.Logf("Low-priority pod %s was preempted. Reason: %s, Message: %s", p.Name, p.Status.Reason, p.Status.Message)
			}
		}
		gomega.Expect(failedCount).To(gomega.BeNumerically(">", 0),
			"At least some low-priority pods should have been preempted by the system-critical pod")

		for _, p := range podList.Items {
			if p.Name == "critical-pod-0" {
				gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodRunning),
					"system-critical pod should be running after preempting low-priority pods")
			}
		}
	}

	// Test ordering: with the gate enabled, high-priority pods should be started
	// before low-priority pods after a kubelet restart.
	f.Context("with PodStartingOrderByPriority enabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): true,
			}
		})

		ginkgo.It("should start high-priority pods before low-priority pods after kubelet restart", func(ctx context.Context) {
			medianHigh, medianLow := runRestartOrderTest(ctx)
			gomega.Expect(medianHigh).To(gomega.BeNumerically("<", medianLow),
				fmt.Sprintf("With PodStartingOrderByPriority enabled, high-priority pods should start before low-priority pods. "+
					"Median high: %d ns, Median low: %d ns, Diff: %d ns",
					medianHigh, medianLow, medianLow-medianHigh))
		})
	})

	// Test ordering: with the gate disabled, pods are started in creation-time order,
	// so the older low-priority pods should start before the newer high-priority pods.
	f.Context("without PodStartingOrderByPriority (default creation-time ordering)", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): false,
			}
		})

		ginkgo.It("should start low-priority pods before high-priority pods after kubelet restart because low-priority pods have older creation timestamps", func(ctx context.Context) {
			medianHigh, medianLow := runRestartOrderTest(ctx)
			gomega.Expect(medianLow).To(gomega.BeNumerically("<", medianHigh),
				fmt.Sprintf("Without PodStartingOrderByPriority, low-priority pods (created first) should start before high-priority pods. "+
					"Median low: %d ns, Median high: %d ns, Diff: %d ns",
					medianLow, medianHigh, medianHigh-medianLow))
		})
	})

	// Test CPU pressure: previously-running pods are re-admitted first (regardless of
	// the gate setting), so a new high-priority pod created while the kubelet is down
	// should fail admission because the running pods already consumed the CPU budget.
	f.Context("new pod rejection with PodStartingOrderByPriority enabled under CPU pressure", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): true,
			}
		})

		ginkgo.It("should reject a new high-priority pod that was never running and re-admit previously-running low-priority pods after kubelet restart", func(ctx context.Context) {
			runNewPodRejectionTest(ctx)
		})
	})

	f.Context("new pod rejection without PodStartingOrderByPriority under CPU pressure", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): false,
			}
		})

		ginkgo.It("should reject a new high-priority pod that was never running and re-admit previously-running low-priority pods after kubelet restart with creation-time ordering", func(ctx context.Context) {
			runNewPodRejectionTest(ctx)
		})
	})

	// Test critical preemption: a new system-node-critical pod (priority >= 2 billion)
	// triggers the critical pod admission handler, which preempts enough low-priority
	// pods to make room—even though those pods were previously running.
	f.Context("critical pod preemption with PodStartingOrderByPriority enabled under CPU pressure", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): true,
			}
		})

		ginkgo.It("should preempt previously-running low-priority pods when a new system-critical pod needs resources after kubelet restart", func(ctx context.Context) {
			runCriticalPodPreemptionTest(ctx)
		})
	})

	f.Context("critical pod preemption without PodStartingOrderByPriority under CPU pressure", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodStartingOrderByPriority): false,
			}
		})

		ginkgo.It("should preempt previously-running low-priority pods when a new system-critical pod needs resources after kubelet restart with creation-time ordering", func(ctx context.Context) {
			runCriticalPodPreemptionTest(ctx)
		})
	})
})

// getPodWithPriorityAndResources returns a pod pinned to the given node with the
// specified priority class and resource requirements.
func getPodWithPriorityAndResources(name, node, priorityClassName string, resources v1.ResourceRequirements) *v1.Pod {
	gracePeriod := int64(30)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      name,
					Image:     busyboxImage,
					Command:   []string{"sh", "-c"},
					Resources: resources,
					Args: []string{`
					sleep 9999999 &
					PID=$!
					_term() {
						echo "Caught SIGTERM signal!"
						wait $PID
					}

					trap _term SIGTERM
					wait $PID
					`},
				},
			},
			PriorityClassName:             priorityClassName,
			TerminationGracePeriodSeconds: &gracePeriod,
			NodeName:                      node,
			RestartPolicy:                 "Always",
		},
	}
}

// newPriorityClass returns a non-preempting PriorityClass with the given name and value.
func newPriorityClass(name string, value int32) *schedulingv1.PriorityClass {
	return &schedulingv1.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Value: value,
	}
}
