/*
Copyright 2016 The Kubernetes Authors.

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

package autoscaling

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	yaml "go.yaml.in/yaml/v2"
	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/scheduling"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	defaultTimeout         = 3 * time.Minute
	scaleUpTimeout         = 5 * time.Minute
	scaleUpTriggerTimeout  = 2 * time.Minute
	scaleDownTimeout       = 20 * time.Minute
	podTimeout             = 2 * time.Minute
	rcCreationRetryTimeout = 4 * time.Minute
	rcCreationRetryDelay   = 20 * time.Second
	makeSchedulableTimeout = 10 * time.Minute
	makeSchedulableDelay   = 20 * time.Second
	freshStatusLimit       = 20 * time.Second

	disabledTaint           = "DisabledForAutoscalingTest"
	criticalAddonsOnlyTaint = "CriticalAddonsOnly"

	caNoScaleUpStatus      = "NoActivity"
	caOngoingScaleUpStatus = "InProgress"
	timestampFormat        = "2006-01-02 15:04:05.999999999 -0700 MST"

	expendablePriorityClassName = "expendable-priority"
	highPriorityClassName       = "high-priority"

	nonExistingBypassedSchedulerName = "non-existing-bypassed-scheduler"
)

// Test assumes that the cluster has a minimum number of nodes at the start of the test.
// Example command to start the test cluster:
// kubetest2 gce -v 2   --repo-root <k/k repo root>   --gcp-project <projct_name>   \
//   --legacy-mode --build --up --env=ENABLE_CUSTOM_METRICS=true --env=KUBE_ENABLE_CLUSTER_AUTOSCALER=true \
//   --env=KUBE_AUTOSCALER_MIN_NODES=3 --env=KUBE_AUTOSCALER_MAX_NODES=6 --env=KUBE_AUTOSCALER_ENABLE_SCALE_DOWN=true \
//   --env=KUBE_ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota,Priority --env=ENABLE_POD_PRIORITY=true

// If you run the test in development consider changing values of the flags
// to speed up the scale down so that the cluster restores its initial size faster.
// Flags that can be adjusted:
// * unremovable-node-recheck-timeout
// * scale-down-unneeded-time
// * scale-down-delay-after-add

var _ = SIGDescribe("Cluster size autoscaling", framework.WithSlow(), func() {
	f := framework.NewDefaultFramework("autoscaling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var c clientset.Interface
	var nodeCount int
	var memAllocatableMb int

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		_, err := c.CoreV1().ConfigMaps("kube-system").Get(ctx, "cluster-autoscaler-status", metav1.GetOptions{})
		if err != nil {
			e2eskipper.Skipf("test expects Cluster Autoscaler to be enabled")
		}

		framework.ExpectNoError(addKubeSystemPdbs(ctx, f))

		nodes, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		nodeCount = len(nodes.Items)
		ginkgo.By(fmt.Sprintf("Initial number of schedulable nodes: %v", nodeCount))
		gomega.Expect(nodes.Items).ToNot(gomega.BeEmpty())
		mem := nodes.Items[0].Status.Allocatable[v1.ResourceMemory]
		memAllocatableMb = int((&mem).Value() / 1024 / 1024)
		// As the last deferred cleanup ensure that the state is restored.
		// AfterEach does not allow for this because it runs before other deferred
		// cleanups happen, and they are blocking cluster restoring its initial size.
		ginkgo.DeferCleanup(func(ctx context.Context) {
			ginkgo.By("Restoring the state after test")
			framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, c, nodeCount, scaleDownTimeout))
			nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)

			s := time.Now()
		makeSchedulableLoop:
			for start := time.Now(); time.Since(start) < makeSchedulableTimeout; time.Sleep(makeSchedulableDelay) {
				var criticalAddonsOnlyErrorType *CriticalAddonsOnlyError
				for _, n := range nodes.Items {
					err = makeNodeSchedulable(ctx, c, &n, true)
					if err != nil && errors.As(err, &criticalAddonsOnlyErrorType) {
						continue makeSchedulableLoop
					} else if err != nil {
						klog.Infof("Error during cleanup: %v", err)
					}
				}
				break
			}
			klog.Infof("Made nodes schedulable again in %v", time.Since(s).String())
		})
	})

	f.It("shouldn't increase cluster size if pending pod is too large", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		ginkgo.By("Creating unschedulable pod")
		ReserveMemory(ctx, f, "memory-reservation", 1, int(1.1*float64(memAllocatableMb)), false, defaultTimeout)
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "memory-reservation")

		ginkgo.By("Waiting for scale up hoping it won't happen")
		// Verify that the appropriate event was generated
		eventFound := false
	EventsLoop:
		for start := time.Now(); time.Since(start) < scaleUpTimeout; time.Sleep(20 * time.Second) {
			ginkgo.By("Waiting for NotTriggerScaleUp event")
			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)

			for _, e := range events.Items {
				if e.InvolvedObject.Kind == "Pod" && e.Reason == "NotTriggerScaleUp" {
					ginkgo.By("NotTriggerScaleUp event found")
					eventFound = true
					break EventsLoop
				}
			}
		}
		if !eventFound {
			framework.Failf("Expected event with kind 'Pod' and reason 'NotTriggerScaleUp' not found.")
		}
		// Verify that cluster size is not changed
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size <= nodeCount }, time.Second))
	})

	simpleScaleUpTest := func(ctx context.Context, unready int) {
		ReserveMemory(ctx, f, "memory-reservation", 100, nodeCount*memAllocatableMb, false, 1*time.Second)
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "memory-reservation")

		// Verify that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(ctx, f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout, unready))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
	}

	f.It("should increase cluster size if pending pods are small", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		simpleScaleUpTest(ctx, 0)
	})

	f.It("shouldn't trigger additional scale-ups during processing scale-up", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		// Wait for the situation to stabilize - CA should be running and have up-to-date node readiness info.
		status, err := waitForScaleUpStatus(ctx, c, func(s *scaleUpStatus) bool {
			return s.ready == s.target && s.ready <= nodeCount
		}, scaleUpTriggerTimeout)
		framework.ExpectNoError(err)

		unmanagedNodes := nodeCount - status.ready

		ginkgo.By("Schedule more pods than can fit and wait for cluster to scale-up")
		ReserveMemory(ctx, f, "memory-reservation", 100, nodeCount*memAllocatableMb, false, 1*time.Second)
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "memory-reservation")

		status, err = waitForScaleUpStatus(ctx, c, func(s *scaleUpStatus) bool {
			return s.status == caOngoingScaleUpStatus
		}, scaleUpTriggerTimeout)
		framework.ExpectNoError(err)
		target := status.target
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		ginkgo.By("Expect no more scale-up to be happening after all pods are scheduled")

		// wait for a while until scale-up finishes; we cannot read CA status immediately
		// after pods are scheduled as status config map is updated by CA once every loop iteration
		status, err = waitForScaleUpStatus(ctx, c, func(s *scaleUpStatus) bool {
			return s.status == caNoScaleUpStatus
		}, 2*freshStatusLimit)
		framework.ExpectNoError(err)

		if status.target != target {
			klog.Warningf("Final number of nodes (%v) does not match initial scale-up target (%v).", status.target, target)
		}
		gomega.Expect(status.timestamp.Add(freshStatusLimit)).To(gomega.BeTemporally(">=", time.Now()))
		gomega.Expect(status.status).To(gomega.Equal(caNoScaleUpStatus))
		gomega.Expect(status.ready).To(gomega.Equal(status.target))
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		gomega.Expect(nodes.Items).To(gomega.HaveLen(status.target + unmanagedNodes))
	})

	f.It("should increase cluster size if pods are pending due to host port conflict", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		scheduling.CreateHostPortPods(ctx, f, "host-port", nodeCount+2, false)
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "host-port")

		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size >= nodeCount+2 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
	})

	f.It("should increase cluster size if pods are pending due to pod anti-affinity", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		pods := nodeCount
		newPods := 2
		labels := map[string]string{
			"anti-affinity": "yes",
		}
		ginkgo.By("starting a pod with anti-affinity on each node")
		framework.ExpectNoError(runAntiAffinityPods(ctx, f, f.Namespace.Name, pods, "anti-affinity-pod", labels, labels))
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "anti-affinity-pod")
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		ginkgo.By("scheduling extra pods with anti-affinity to existing ones")
		framework.ExpectNoError(runAntiAffinityPods(ctx, f, f.Namespace.Name, newPods, "extra-pod", labels, labels))
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "extra-pod")

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, c, nodeCount+newPods, scaleUpTimeout))
	})

	f.It("should increase cluster size if pod requesting EmptyDir volume is pending", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		ginkgo.By("creating pods")
		pods := nodeCount
		newPods := 1
		labels := map[string]string{
			"anti-affinity": "yes",
		}
		framework.ExpectNoError(runAntiAffinityPods(ctx, f, f.Namespace.Name, pods, "anti-affinity-pod", labels, labels))
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "anti-affinity-pod")

		ginkgo.By("waiting for all pods before triggering scale up")
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		ginkgo.By("creating a pod requesting EmptyDir")
		framework.ExpectNoError(runVolumeAntiAffinityPods(ctx, f, f.Namespace.Name, newPods, "extra-pod", labels, labels, emptyDirVolumes))
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, "extra-pod")

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, c, nodeCount+newPods, scaleUpTimeout))
	})

	f.It("should correctly scale down after a node is not needed", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		ginkgo.By("Increase cluster size")
		cleanupFunc := increaseClusterSize(ctx, f, c, nodeCount+2)

		ginkgo.By("Remove the RC to make nodes not needed any more")
		framework.ExpectNoError(cleanupFunc())

		ginkgo.By("Some uneeded nodes should be removed")
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(ctx, f.ClientSet,
			func(size int) bool { return size < nodeCount+2 }, scaleDownTimeout, 0))
	})

	f.It("should be able to scale down when rescheduling a pod is required and pdb allows for it", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		runDrainTest(ctx, f, c, nodeCount, f.Namespace.Name, 1, 1, func(increasedSize int) {
			ginkgo.By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	f.It("shouldn't be able to scale down when rescheduling a pod is required, but pdb doesn't allow drain", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		runDrainTest(ctx, f, c, nodeCount, f.Namespace.Name, 1, 0, func(increasedSize int) {
			ginkgo.By("No nodes should be removed")
			time.Sleep(scaleDownTimeout)
			nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			gomega.Expect(nodes.Items).To(gomega.HaveLen(increasedSize))
		})
	})

	f.It("should be able to scale down by draining multiple pods one by one as dictated by pdb", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		runDrainTest(ctx, f, c, nodeCount, f.Namespace.Name, 2, 1, func(increasedSize int) {
			ginkgo.By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	f.It("should be able to scale down by draining system pods with pdb", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		runDrainTest(ctx, f, c, nodeCount, "kube-system", 2, 1, func(increasedSize int) {
			ginkgo.By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	f.It("shouldn't scale up when expendable pod is created", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		createPriorityClasses(ctx, f)
		// Create nodesCountAfterResize+1 pods allocating 0.7 allocatable on present nodes. One more node will have to be created.
		ginkgo.DeferCleanup(ReserveMemoryWithPriority, f, "memory-reservation", nodeCount+1, int(float64(nodeCount+1)*float64(0.7)*float64(memAllocatableMb)), false, time.Second, expendablePriorityClassName)
		ginkgo.By(fmt.Sprintf("Waiting for scale up hoping it won't happen, sleep for %s", scaleUpTimeout.String()))
		time.Sleep(scaleUpTimeout)
		// Verify that cluster size is not changed
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size == nodeCount }, time.Second))
	})

	f.It("should scale up when non expendable pod is created", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		createPriorityClasses(ctx, f)
		// Create nodesCountAfterResize+1 pods allocating 0.7 allocatable on present nodes. One more node will have to be created.
		cleanupFunc := ReserveMemoryWithPriority(ctx, f, "memory-reservation", nodeCount+1, int(float64(nodeCount+1)*float64(0.7)*float64(memAllocatableMb)), true, scaleUpTimeout, highPriorityClassName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		// Verify that cluster size is not changed
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size > nodeCount }, time.Second))
	})

	f.It("shouldn't scale up when expendable pod is preempted", feature.ClusterSizeAutoscalingScaleUp, func(ctx context.Context) {
		createPriorityClasses(ctx, f)
		// Create nodesCountAfterResize pods allocating 0.7 allocatable on present nodes - one pod per node.
		cleanupFunc1 := ReserveMemoryWithPriority(ctx, f, "memory-reservation1", nodeCount, int(float64(nodeCount)*float64(0.7)*float64(memAllocatableMb)), true, defaultTimeout, expendablePriorityClassName)
		defer func() {
			framework.ExpectNoError(cleanupFunc1())
		}()
		// Create nodesCountAfterResize pods allocating 0.7 allocatable on present nodes - one pod per node. Pods created here should preempt pods created above.
		cleanupFunc2 := ReserveMemoryWithPriority(ctx, f, "memory-reservation2", nodeCount, int(float64(nodeCount)*float64(0.7)*float64(memAllocatableMb)), true, defaultTimeout, highPriorityClassName)
		defer func() {
			framework.ExpectNoError(cleanupFunc2())
		}()
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size == nodeCount }, time.Second))
	})

	f.It("should scale down when expendable pod is running", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		createPriorityClasses(ctx, f)
		increasedSize := nodeCount + 2
		cleanupIncreaseFunc := increaseClusterSize(ctx, f, c, increasedSize)
		// Create increasedSize pods allocating 0.7 allocatable on present nodes - one pod per node.
		cleanupFunc := ReserveMemoryWithPriority(ctx, f, "memory-reservation", increasedSize, int(float64(increasedSize)*float64(0.7)*float64(memAllocatableMb)), true, scaleUpTimeout, expendablePriorityClassName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		ginkgo.By("Remove pods that increased the cluster size")
		framework.ExpectNoError(cleanupIncreaseFunc())
		ginkgo.By("Waiting for scale down")
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size == nodeCount }, scaleDownTimeout))
	})

	f.It("shouldn't scale down when non expendable pod is running", feature.ClusterSizeAutoscalingScaleDown, func(ctx context.Context) {
		createPriorityClasses(ctx, f)
		increasedSize := nodeCount + 2
		cleanupIncreased := increaseClusterSize(ctx, f, c, increasedSize)
		// Create increasedSize pods allocating 0.7 allocatable on present nodes - one pod per node.
		cleanupFunc := ReserveMemoryWithPriority(ctx, f, "memory-reservation", increasedSize, int(float64(increasedSize)*float64(0.7)*float64(memAllocatableMb)), true, scaleUpTimeout, highPriorityClassName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		framework.ExpectNoError(cleanupIncreased())
		ginkgo.By(fmt.Sprintf("Waiting for scale down hoping it won't happen, sleep for %s", scaleDownTimeout.String()))
		time.Sleep(scaleDownTimeout)
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size == increasedSize }, time.Second))
	})

	f.It("should scale up when unprocessed pod is created and is going to be unschedulable", feature.ClusterScaleUpBypassScheduler, func(ctx context.Context) {
		// 70% of allocatable memory of a single node * replica count, forcing a scale up in case of normal pods
		replicaCount := 2 * nodeCount
		reservedMemory := int(float64(replicaCount) * float64(0.7) * float64(memAllocatableMb))
		cleanupFunc := ReserveMemoryWithSchedulerName(ctx, f, "memory-reservation", replicaCount, reservedMemory, false, 1, nonExistingBypassedSchedulerName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		// Verify that cluster size is increased
		ginkgo.By("Waiting for cluster scale-up")
		sizeFunc := func(size int) bool {
			// Softly checks scale-up since other types of machines can be added which would affect #nodes
			return size > nodeCount
		}
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(ctx, f.ClientSet, sizeFunc, scaleUpTimeout, 0))
	})

	f.It("shouldn't scale up when unprocessed pod is created and is going to be schedulable", feature.ClusterScaleUpBypassScheduler, func(ctx context.Context) {
		// 50% of allocatable memory of a single node, so that no scale up would trigger in normal cases
		replicaCount := 1
		reservedMemory := int(float64(0.5) * float64(memAllocatableMb))
		cleanupFunc := ReserveMemoryWithSchedulerName(ctx, f, "memory-reservation", replicaCount, reservedMemory, false, 1, nonExistingBypassedSchedulerName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		// Verify that cluster size is the same
		ginkgo.By(fmt.Sprintf("Waiting for scale up hoping it won't happen, polling cluster size for %s", scaleUpTimeout.String()))
		sizeFunc := func(size int) bool {
			return size == nodeCount
		}
		gomega.Consistently(ctx, func() error {
			return WaitForClusterSizeFunc(ctx, f.ClientSet, sizeFunc, time.Second)
		}).WithTimeout(scaleUpTimeout).WithPolling(framework.Poll).ShouldNot(gomega.HaveOccurred())
	})

	f.It("shouldn't scale up when unprocessed pod is created and scheduler is not specified to be bypassed", feature.ClusterScaleUpBypassScheduler, func(ctx context.Context) {
		// 70% of allocatable memory of a single node * replica count, forcing a scale up in case of normal pods
		replicaCount := 2 * nodeCount
		reservedMemory := int(float64(replicaCount) * float64(0.7) * float64(memAllocatableMb))
		schedulerName := "non-existent-scheduler-" + f.UniqueName
		cleanupFunc := ReserveMemoryWithSchedulerName(ctx, f, "memory-reservation", replicaCount, reservedMemory, false, 1, schedulerName)
		defer func() {
			framework.ExpectNoError(cleanupFunc())
		}()
		// Verify that cluster size is the same
		ginkgo.By(fmt.Sprintf("Waiting for scale up hoping it won't happen, polling cluster size for %s", scaleUpTimeout.String()))
		sizeFunc := func(size int) bool {
			return size == nodeCount
		}
		gomega.Consistently(ctx, func() error {
			return WaitForClusterSizeFunc(ctx, f.ClientSet, sizeFunc, time.Second)
		}).WithTimeout(scaleUpTimeout).WithPolling(framework.Poll).ShouldNot(gomega.HaveOccurred())
	})

})

func runDrainTest(ctx context.Context, f *framework.Framework, c clientset.Interface, nodeCount int, namespace string, podsPerNode, pdbSize int, verifyFunction func(int)) {
	increasedCount := nodeCount + 2

	cleanupIncreasedSizeFunc := increaseClusterSize(ctx, f, c, increasedCount)

	nodes, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{
		"spec.unschedulable": "false",
	}.AsSelector().String()})
	framework.ExpectNoError(err)
	numPods := len(nodes.Items) * podsPerNode
	testID := string(uuid.NewUUID()) // So that we can label and find pods
	labelMap := map[string]string{"test_id": testID}
	makeNodesSchedulable, err := runReplicatedPodOnEachNode(ctx, f, nodes.Items, namespace, podsPerNode, "reschedulable-pods", labelMap, 0)
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, namespace, "reschedulable-pods")

	ginkgo.By("Create a PodDisruptionBudget")
	minAvailable := intstr.FromInt32(int32(numPods - pdbSize))
	pdb := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test_pdb",
			Namespace: namespace,
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: labelMap},
			MinAvailable: &minAvailable,
		},
	}
	_, err = f.ClientSet.PolicyV1().PodDisruptionBudgets(namespace).Create(ctx, pdb, metav1.CreateOptions{})

	ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.PolicyV1().PodDisruptionBudgets(namespace).Delete), pdb.Name, metav1.DeleteOptions{})

	framework.ExpectNoError(err)
	framework.ExpectNoError(cleanupIncreasedSizeFunc())
	framework.ExpectNoError(makeNodesSchedulable())
	verifyFunction(increasedCount)
}

func reserveMemory(ctx context.Context, f *framework.Framework, id string, replicas, megabytes int, expectRunning bool, timeout time.Duration, selector map[string]string, tolerations []v1.Toleration, priorityClassName, schedulerName string) func() error {
	ginkgo.By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &testutils.RCConfig{
		Client:            f.ClientSet,
		Name:              id,
		Namespace:         f.Namespace.Name,
		Timeout:           timeout,
		Image:             imageutils.GetPauseImageName(),
		Replicas:          replicas,
		MemRequest:        request,
		NodeSelector:      selector,
		Tolerations:       tolerations,
		PriorityClassName: priorityClassName,
		SchedulerName:     schedulerName,
	}
	for start := time.Now(); time.Since(start) < rcCreationRetryTimeout; time.Sleep(rcCreationRetryDelay) {
		err := e2erc.RunRC(ctx, *config)
		if err != nil && strings.Contains(err.Error(), "Error creating replication controller") {
			klog.Warningf("Failed to create memory reservation: %v", err)
			continue
		}
		if expectRunning {
			framework.ExpectNoError(err)
		}
		return func() error {
			return e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, id)
		}
	}
	framework.Failf("Failed to reserve memory within timeout")
	return nil
}

// ReserveMemoryWithPriority creates a replication controller with pods with priority that, in summation,
// request the specified amount of memory.
func ReserveMemoryWithPriority(ctx context.Context, f *framework.Framework, id string, replicas, megabytes int, expectRunning bool, timeout time.Duration, priorityClassName string) func() error {
	return reserveMemory(ctx, f, id, replicas, megabytes, expectRunning, timeout, nil, nil, priorityClassName, "")
}

// ReserveMemoryWithSchedulerName creates a replication controller with pods with scheduler name that, in summation,
// request the specified amount of memory.
func ReserveMemoryWithSchedulerName(ctx context.Context, f *framework.Framework, id string, replicas, megabytes int, expectRunning bool, timeout time.Duration, schedulerName string) func() error {
	return reserveMemory(ctx, f, id, replicas, megabytes, expectRunning, timeout, nil, nil, "", schedulerName)
}

// ReserveMemory creates a replication controller with pods that, in summation,
// request the specified amount of memory.
func ReserveMemory(ctx context.Context, f *framework.Framework, id string, replicas, megabytes int, expectRunning bool, timeout time.Duration) func() error {
	return reserveMemory(ctx, f, id, replicas, megabytes, expectRunning, timeout, nil, nil, "", "")
}

// WaitForClusterSizeFunc waits until the cluster size matches the given function.
func WaitForClusterSizeFunc(ctx context.Context, c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration) error {
	return WaitForClusterSizeFuncWithUnready(ctx, c, sizeFunc, timeout, 0)
}

// WaitForClusterSizeFuncWithUnready waits until the cluster size matches the given function and assumes some unready nodes.
func WaitForClusterSizeFuncWithUnready(ctx context.Context, c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration, expectedUnready int) error {
	for start := time.Now(); time.Since(start) < timeout && ctx.Err() == nil; time.Sleep(20 * time.Second) {
		nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			klog.Warningf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		e2enode.Filter(nodes, func(node v1.Node) bool {
			return e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true)
		})
		numReady := len(nodes.Items)

		if numNodes == numReady+expectedUnready && sizeFunc(numNodes) {
			klog.Infof("Cluster has reached the desired size. Current size %d, not ready nodes %d", numNodes, numNodes-numReady)
			return nil
		}
		klog.Infof("Waiting for cluster with func, current size %d, not ready nodes %d", numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for appropriate cluster size", timeout)
}

func waitForCaPodsReadyInNamespace(ctx context.Context, f *framework.Framework, c clientset.Interface, tolerateUnreadyCount int) error {
	var notready []string
	for start := time.Now(); time.Now().Before(start.Add(scaleUpTimeout)) && ctx.Err() == nil; time.Sleep(20 * time.Second) {
		pods, err := c.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("failed to get pods: %w", err)
		}
		notready = make([]string, 0)
		for _, pod := range pods.Items {
			ready := false
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.PodReady && c.Status == v1.ConditionTrue {
					ready = true
				}
			}
			// Failed pods in this context generally mean that they have been
			// double scheduled onto a node, but then failed a constraint check.
			if pod.Status.Phase == v1.PodFailed {
				klog.Warningf("Pod has failed: %v", pod)
			}
			if !ready && pod.Status.Phase != v1.PodFailed {
				notready = append(notready, pod.Name)
			}
		}
		if len(notready) <= tolerateUnreadyCount {
			klog.Infof("sufficient number of pods ready. Tolerating %d unready", tolerateUnreadyCount)
			return nil
		}
		klog.Infof("Too many pods are not ready yet: %v", notready)
	}
	klog.Info("Timeout on waiting for pods being ready")
	klog.Info(e2ekubectl.RunKubectlOrDie(f.Namespace.Name, "get", "pods", "-o", "json", "--all-namespaces"))
	klog.Info(e2ekubectl.RunKubectlOrDie(f.Namespace.Name, "get", "nodes", "-o", "json"))

	// Some pods are still not running.
	return fmt.Errorf("too many pods are still not running: %v", notready)
}

func waitForAllCaPodsReadyInNamespace(ctx context.Context, f *framework.Framework, c clientset.Interface) error {
	return waitForCaPodsReadyInNamespace(ctx, f, c, 0)
}

func makeNodeUnschedulable(ctx context.Context, c clientset.Interface, node *v1.Node) error {
	ginkgo.By(fmt.Sprintf("Taint node %s", node.Name))
	for j := 0; j < 3; j++ {
		freshNode, err := c.CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		for _, taint := range freshNode.Spec.Taints {
			if taint.Key == disabledTaint {
				return nil
			}
		}
		freshNode.Spec.Taints = append(freshNode.Spec.Taints, v1.Taint{
			Key:    disabledTaint,
			Value:  "DisabledForTest",
			Effect: v1.TaintEffectNoSchedule,
		})
		_, err = c.CoreV1().Nodes().Update(ctx, freshNode, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		klog.Warningf("Got 409 conflict when trying to taint node, retries left: %v", 3-j)
	}
	return fmt.Errorf("failed to taint node in allowed number of retries")
}

// CriticalAddonsOnlyError implements the `error` interface, and signifies the
// presence of the `CriticalAddonsOnly` taint on the node.
type CriticalAddonsOnlyError struct{}

func (CriticalAddonsOnlyError) Error() string {
	return "CriticalAddonsOnly taint found on node"
}

func makeNodeSchedulable(ctx context.Context, c clientset.Interface, node *v1.Node, failOnCriticalAddonsOnly bool) error {
	ginkgo.By(fmt.Sprintf("Remove taint from node %s", node.Name))
	for j := 0; j < 3; j++ {
		freshNode, err := c.CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		var newTaints []v1.Taint
		for _, taint := range freshNode.Spec.Taints {
			if failOnCriticalAddonsOnly && taint.Key == criticalAddonsOnlyTaint {
				return CriticalAddonsOnlyError{}
			}
			if taint.Key != disabledTaint {
				newTaints = append(newTaints, taint)
			}
		}

		if len(newTaints) == len(freshNode.Spec.Taints) {
			return nil
		}
		freshNode.Spec.Taints = newTaints
		_, err = c.CoreV1().Nodes().Update(ctx, freshNode, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		klog.Warningf("Got 409 conflict when trying to taint node, retries left: %v", 3-j)
	}
	return fmt.Errorf("failed to remove taint from node in allowed number of retries")
}

// Create an RC running a given number of pods with anti-affinity
func runAntiAffinityPods(ctx context.Context, f *framework.Framework, namespace string, pods int, id string, podLabels, antiAffinityLabels map[string]string) error {
	config := &testutils.RCConfig{
		Affinity:  buildAntiAffinity(antiAffinityLabels),
		Client:    f.ClientSet,
		Name:      id,
		Namespace: namespace,
		Timeout:   scaleUpTimeout,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  pods,
		Labels:    podLabels,
	}
	err := e2erc.RunRC(ctx, *config)
	if err != nil {
		return err
	}
	_, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func runVolumeAntiAffinityPods(ctx context.Context, f *framework.Framework, namespace string, pods int, id string, podLabels, antiAffinityLabels map[string]string, volumes []v1.Volume) error {
	config := &testutils.RCConfig{
		Affinity:  buildAntiAffinity(antiAffinityLabels),
		Volumes:   volumes,
		Client:    f.ClientSet,
		Name:      id,
		Namespace: namespace,
		Timeout:   scaleUpTimeout,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  pods,
		Labels:    podLabels,
	}
	err := e2erc.RunRC(ctx, *config)
	if err != nil {
		return err
	}
	_, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

var emptyDirVolumes = []v1.Volume{
	{
		Name: "empty-volume",
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	},
}

func buildAntiAffinity(labels map[string]string) *v1.Affinity {
	return &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchLabels: labels,
					},
					TopologyKey: "kubernetes.io/hostname",
				},
			},
		},
	}
}

// Create an RC running a given number of pods on each node without adding any constraint forcing
// such pod distribution. This is meant to create a bunch of underutilized (but not unused) nodes
// with pods that can be rescheduled on different nodes.
// This is achieved using the following method:
// 1. disable scheduling on each node
// 2. create an empty RC
// 3. for each node:
// 3a. enable scheduling on that node
// 3b. increase number of replicas in RC by podsPerNode
// Return the function to enable back scheduling on each node
func runReplicatedPodOnEachNode(ctx context.Context, f *framework.Framework, nodes []v1.Node, namespace string, podsPerNode int, id string, labels map[string]string, memRequest int64) (func() error, error) {
	ginkgo.By("Run a pod on each node")
	for _, node := range nodes {
		err := makeNodeUnschedulable(ctx, f.ClientSet, &node)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := makeNodeSchedulable(ctx, f.ClientSet, &node, false)
			klog.Infof("Error during cleanup: %v", err)
		})

		if err != nil {
			return nil, err
		}
	}
	config := &testutils.RCConfig{
		Client:     f.ClientSet,
		Name:       id,
		Namespace:  namespace,
		Timeout:    defaultTimeout,
		Image:      imageutils.GetPauseImageName(),
		Replicas:   0,
		Labels:     labels,
		MemRequest: memRequest,
	}
	err := e2erc.RunRC(ctx, *config)
	if err != nil {
		return nil, err
	}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	for i, node := range nodes {
		err = makeNodeSchedulable(ctx, f.ClientSet, &node, false)
		if err != nil {
			return nil, err
		}

		// Update replicas count, to create new pods that will be allocated on node
		// (we retry 409 errors in case rc reference got out of sync)
		for j := 0; j < 3; j++ {
			*rc.Spec.Replicas = int32((i + 1) * podsPerNode)
			rc, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Update(ctx, rc, metav1.UpdateOptions{})
			if err == nil {
				break
			}
			if !apierrors.IsConflict(err) {
				return nil, err
			}
			klog.Warningf("Got 409 conflict when trying to scale RC, retries left: %v", 3-j)
			rc, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
		}

		err = wait.PollUntilContextTimeout(ctx, 5*time.Second, podTimeout, true, func(ctx context.Context) (bool, error) {
			rc, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
			if err != nil || rc.Status.ReadyReplicas < int32((i+1)*podsPerNode) {
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			return nil, fmt.Errorf("failed to coerce RC into spawning a pod on node %s within timeout", node.Name)
		}
		err = makeNodeUnschedulable(ctx, f.ClientSet, &node)
		if err != nil {
			return nil, err
		}
	}
	makeNodesSchedulable := func() error {
		for _, node := range nodes {
			err = makeNodeSchedulable(ctx, f.ClientSet, &node, false)
			if err != nil {
				return err
			}
		}
		return nil
	}
	return makeNodesSchedulable, nil
}

// Increase cluster size by creating pods with anti-affinity.
// Returns a function that removes the pods.
// Adds the same to deferred cleanup in case the function was not called.
func increaseClusterSizeWithTimeout(ctx context.Context, f *framework.Framework, c clientset.Interface, targetNodeCount int, timeout time.Duration) func() error {
	labels := map[string]string{
		"anti-affinity": "yes",
	}
	framework.ExpectNoError(runAntiAffinityPods(ctx, f, f.Namespace.Name, targetNodeCount, "increase-size-pod", labels, labels))
	cleanupFunc := func() error {
		return e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, "increase-size-pod")
	}

	ginkgo.DeferCleanup(func(ctx context.Context) {
		klog.Infof("Cleaning up RC and pods if test did not clean them up")
		err := cleanupFunc()
		klog.Infof("Error during cleanup: %v", err)
	})

	// Verify that cluster size is increased
	framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(ctx, f.ClientSet,
		func(size int) bool { return size >= targetNodeCount }, timeout, 0))
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
	return cleanupFunc
}

func increaseClusterSize(ctx context.Context, f *framework.Framework, c clientset.Interface, targetNodeCount int) func() error {
	return increaseClusterSizeWithTimeout(ctx, f, c, targetNodeCount, scaleUpTimeout)
}

type scaleUpStatus struct {
	status    string
	ready     int
	target    int
	timestamp time.Time
}

// Try to get timestamp from status.
func getStatusTimestamp(parsedStatus map[interface{}]interface{}) (time.Time, error) {
	timestamp := parsedStatus["time"]
	if timestamp == nil {
		return time.Time{}, fmt.Errorf("failed to parse CA status timestamp, parsed status: %v", parsedStatus)
	}

	timestampParsed, err := time.Parse(timestampFormat, timestamp.(string))
	if err != nil {
		return time.Time{}, err
	}
	return timestampParsed, nil
}

// Try to get scaleup statuses of all groups
func getScaleUpStatus(ctx context.Context, c clientset.Interface) (*scaleUpStatus, error) {
	configMap, err := c.CoreV1().ConfigMaps("kube-system").Get(ctx, "cluster-autoscaler-status", metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	status, ok := configMap.Data["status"]
	if !ok {
		return nil, fmt.Errorf("status information not found in configmap")
	}

	parsedStatus := make(map[interface{}]interface{})
	err = yaml.Unmarshal([]byte(status), &parsedStatus)
	if err != nil {
		return nil, fmt.Errorf("failed to parse the autoscaler status: %w", err)
	}

	timestamp, err := getStatusTimestamp(parsedStatus)
	if err != nil {
		return nil, err
	}

	result := scaleUpStatus{
		status:    caNoScaleUpStatus,
		ready:     0,
		target:    0,
		timestamp: timestamp,
	}
	for _, nodeGroup := range parsedStatus["nodeGroups"].([]interface{}) {
		if nodeGroup.(map[interface{}]interface{})["scaleUp"].(map[interface{}]interface{})["status"].(string) == caOngoingScaleUpStatus {
			result.status = caOngoingScaleUpStatus
		}
		newReady := nodeGroup.(map[interface{}]interface{})["health"].(map[interface{}]interface{})["nodeCounts"].(map[interface{}]interface{})["registered"].(map[interface{}]interface{})["ready"].(int)
		if err != nil {
			return nil, err
		}
		result.ready += newReady
		newTarget := nodeGroup.(map[interface{}]interface{})["health"].(map[interface{}]interface{})["cloudProviderTarget"].(int)
		if err != nil {
			return nil, err
		}
		result.target += newTarget
	}
	klog.Infof("Cluster-Autoscaler scale-up status: %v (%v, %v)", result.status, result.ready, result.target)
	return &result, nil
}

func waitForScaleUpStatus(ctx context.Context, c clientset.Interface, cond func(s *scaleUpStatus) bool, timeout time.Duration) (*scaleUpStatus, error) {
	var finalErr error
	var status *scaleUpStatus
	err := wait.PollUntilContextTimeout(ctx, 5*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		status, finalErr = getScaleUpStatus(ctx, c)
		if finalErr != nil {
			klog.Infof("Error getting the autoscaler status: %v", finalErr)
			return false, nil
		}
		if status.timestamp.Add(freshStatusLimit).Before(time.Now()) {
			// stale status
			finalErr = fmt.Errorf("status too old")
			return false, nil
		}
		return cond(status), nil
	})
	if err != nil {
		err = fmt.Errorf("failed to find expected scale up status: %v, last status: %v, final err: %v", err, status, finalErr)
	}
	return status, err
}

func addKubeSystemPdbs(ctx context.Context, f *framework.Framework) error {
	ginkgo.By("Create PodDisruptionBudgets for kube-system components, so they can be migrated if required")

	var newPdbs []string
	cleanup := func(ctx context.Context) {
		var finalErr error
		for _, newPdbName := range newPdbs {
			ginkgo.By(fmt.Sprintf("Delete PodDisruptionBudget %v", newPdbName))
			err := f.ClientSet.PolicyV1().PodDisruptionBudgets("kube-system").Delete(ctx, newPdbName, metav1.DeleteOptions{})
			if err != nil {
				// log error, but attempt to remove other pdbs
				klog.Errorf("Failed to delete PodDisruptionBudget %v, err: %v", newPdbName, err)
				finalErr = err
			}
		}
		if finalErr != nil {
			framework.Failf("Error during PodDisruptionBudget cleanup: %v", finalErr)
		}
	}
	ginkgo.DeferCleanup(cleanup)

	type pdbInfo struct {
		label        string
		minAvailable int
	}
	pdbsToAdd := []pdbInfo{
		{label: "kube-dns", minAvailable: 1},
		{label: "kube-dns-autoscaler", minAvailable: 0},
		{label: "metrics-server", minAvailable: 0},
		{label: "kubernetes-dashboard", minAvailable: 0},
		{label: "glbc", minAvailable: 0},
		{label: "volume-snapshot-controller", minAvailable: 0},
		{label: "fluentd-gcp", minAvailable: 0},
		{label: "fluentd-gcp-scaler", minAvailable: 0},
		{label: "event-exporter", minAvailable: 0},
		{label: "cloud-controller-manager", minAvailable: 0},
	}
	for _, pdbData := range pdbsToAdd {
		ginkgo.By(fmt.Sprintf("Create PodDisruptionBudget for %v", pdbData.label))
		labelMap := map[string]string{"k8s-app": pdbData.label}
		pdbName := fmt.Sprintf("test-pdb-for-%v", pdbData.label)
		minAvailable := intstr.FromInt32(int32(pdbData.minAvailable))
		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pdbName,
				Namespace: "kube-system",
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				Selector:     &metav1.LabelSelector{MatchLabels: labelMap},
				MinAvailable: &minAvailable,
			},
		}
		_, err := f.ClientSet.PolicyV1().PodDisruptionBudgets("kube-system").Create(ctx, pdb, metav1.CreateOptions{})
		newPdbs = append(newPdbs, pdbName)

		if err != nil {
			return err
		}
	}
	return nil
}

func createPriorityClasses(ctx context.Context, f *framework.Framework) {
	priorityClasses := map[string]int32{
		expendablePriorityClassName: -15,
		highPriorityClassName:       1000,
	}
	for className, priority := range priorityClasses {
		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: className}, Value: priority}, metav1.CreateOptions{})
		if err != nil {
			klog.Errorf("Error creating priority class: %v", err)
		}
		if err != nil && !apierrors.IsAlreadyExists(err) {
			framework.Failf("unexpected error while creating priority class: %v", err)
		}
	}

	ginkgo.DeferCleanup(func(ctx context.Context) {
		for className := range priorityClasses {
			err := f.ClientSet.SchedulingV1().PriorityClasses().Delete(ctx, className, metav1.DeleteOptions{})
			if err != nil {
				klog.Errorf("Error deleting priority class: %v", err)
			}
		}
	})
}
