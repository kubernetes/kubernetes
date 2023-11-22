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
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	memoryReservationTimeout = 5 * time.Minute
	largeResizeTimeout       = 8 * time.Minute
	largeScaleUpTimeout      = 10 * time.Minute
	maxNodes                 = 1000
)

type clusterPredicates struct {
	nodes int
}

type scaleUpTestConfig struct {
	initialNodes   int
	initialPods    int
	extraPods      *testutils.RCConfig
	expectedResult *clusterPredicates
}

var _ = SIGDescribe("Cluster size autoscaler scalability", framework.WithSlow(), func() {
	f := framework.NewDefaultFramework("autoscaling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var c clientset.Interface
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int
	var originalSizes map[string]int
	var sum int

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "kubemark")

		// Check if Cloud Autoscaler is enabled by trying to get its ConfigMap.
		_, err := f.ClientSet.CoreV1().ConfigMaps("kube-system").Get(ctx, "cluster-autoscaler-status", metav1.GetOptions{})
		if err != nil {
			e2eskipper.Skipf("test expects Cluster Autoscaler to be enabled")
		}

		c = f.ClientSet
		if originalSizes == nil {
			originalSizes = make(map[string]int)
			sum = 0
			for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
				size, err := framework.GroupSize(mig)
				framework.ExpectNoError(err)
				ginkgo.By(fmt.Sprintf("Initial size of %s: %d", mig, size))
				originalSizes[mig] = size
				sum += size
			}
		}

		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, c, sum, scaleUpTimeout))

		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		nodeCount = len(nodes.Items)
		cpu := nodes.Items[0].Status.Capacity[v1.ResourceCPU]
		mem := nodes.Items[0].Status.Capacity[v1.ResourceMemory]
		coresPerNode = int((&cpu).MilliValue() / 1000)
		memCapacityMb = int((&mem).Value() / 1024 / 1024)

		gomega.Expect(nodeCount).To(gomega.Equal(sum))

		if framework.ProviderIs("gke") {
			val, err := isAutoscalerEnabled(3)
			framework.ExpectNoError(err)
			if !val {
				err = enableAutoscaler("default-pool", 3, 5)
				framework.ExpectNoError(err)
			}
		}
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("Restoring initial size of the cluster"))
		setMigSizes(originalSizes)
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, c, nodeCount, scaleDownTimeout))
		nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		s := time.Now()
	makeSchedulableLoop:
		for start := time.Now(); time.Since(start) < makeSchedulableTimeout; time.Sleep(makeSchedulableDelay) {
			for _, n := range nodes.Items {
				err = makeNodeSchedulable(ctx, c, &n, true)
				switch err.(type) {
				case CriticalAddonsOnlyError:
					continue makeSchedulableLoop
				default:
					framework.ExpectNoError(err)
				}
			}
			break
		}
		klog.Infof("Made nodes schedulable again in %v", time.Since(s).String())
	})

	f.It("should scale up at all", feature.ClusterAutoscalerScalability1, func(ctx context.Context) {
		perNodeReservation := int(float64(memCapacityMb) * 0.95)
		replicasPerNode := 10

		additionalNodes := maxNodes - nodeCount
		replicas := additionalNodes * replicasPerNode
		additionalReservation := additionalNodes * perNodeReservation

		// saturate cluster
		reservationCleanup := ReserveMemory(ctx, f, "some-pod", nodeCount*2, nodeCount*perNodeReservation, true, memoryReservationTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		// configure pending pods & expected scale up
		rcConfig := reserveMemoryRCConfig(f, "extra-pod-1", replicas, additionalReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes)
		config := createScaleUpTestConfig(nodeCount, nodeCount, rcConfig, expectedResult)

		// run test
		testCleanup := simpleScaleUpTest(ctx, f, config)
		defer testCleanup()
	})

	f.It("should scale up twice", feature.ClusterAutoscalerScalability2, func(ctx context.Context) {
		perNodeReservation := int(float64(memCapacityMb) * 0.95)
		replicasPerNode := 10
		additionalNodes1 := int(math.Ceil(0.7 * maxNodes))
		additionalNodes2 := int(math.Ceil(0.25 * maxNodes))
		if additionalNodes1+additionalNodes2 > maxNodes {
			additionalNodes2 = maxNodes - additionalNodes1
		}

		replicas1 := additionalNodes1 * replicasPerNode
		replicas2 := additionalNodes2 * replicasPerNode

		klog.Infof("cores per node: %v", coresPerNode)

		// saturate cluster
		initialReplicas := nodeCount
		reservationCleanup := ReserveMemory(ctx, f, "some-pod", initialReplicas, nodeCount*perNodeReservation, true, memoryReservationTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		klog.Infof("Reserved successfully")

		// configure pending pods & expected scale up #1
		rcConfig := reserveMemoryRCConfig(f, "extra-pod-1", replicas1, additionalNodes1*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes1)
		config := createScaleUpTestConfig(nodeCount, nodeCount, rcConfig, expectedResult)

		// run test #1
		tolerateUnreadyNodes := additionalNodes1 / 20
		tolerateUnreadyPods := (initialReplicas + replicas1) / 20
		testCleanup1 := simpleScaleUpTestWithTolerance(ctx, f, config, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup1()

		klog.Infof("Scaled up once")

		// configure pending pods & expected scale up #2
		rcConfig2 := reserveMemoryRCConfig(f, "extra-pod-2", replicas2, additionalNodes2*perNodeReservation, largeScaleUpTimeout)
		expectedResult2 := createClusterPredicates(nodeCount + additionalNodes1 + additionalNodes2)
		config2 := createScaleUpTestConfig(nodeCount+additionalNodes1, nodeCount+additionalNodes2, rcConfig2, expectedResult2)

		// run test #2
		tolerateUnreadyNodes = maxNodes / 20
		tolerateUnreadyPods = (initialReplicas + replicas1 + replicas2) / 20
		testCleanup2 := simpleScaleUpTestWithTolerance(ctx, f, config2, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup2()

		klog.Infof("Scaled up twice")
	})

	f.It("should scale down empty nodes", feature.ClusterAutoscalerScalability3, func(ctx context.Context) {
		perNodeReservation := int(float64(memCapacityMb) * 0.7)
		replicas := int(math.Ceil(maxNodes * 0.7))
		totalNodes := maxNodes

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, f.ClientSet, totalNodes, largeResizeTimeout))

		// run replicas
		rcConfig := reserveMemoryRCConfig(f, "some-pod", replicas, replicas*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(totalNodes)
		config := createScaleUpTestConfig(totalNodes, totalNodes, rcConfig, expectedResult)
		tolerateUnreadyNodes := totalNodes / 10
		tolerateUnreadyPods := replicas / 10
		testCleanup := simpleScaleUpTestWithTolerance(ctx, f, config, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup()

		// check if empty nodes are scaled down
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool {
				return size <= replicas+3 // leaving space for non-evictable kube-system pods
			}, scaleDownTimeout))
	})

	f.It("should scale down underutilized nodes", feature.ClusterAutoscalerScalability4, func(ctx context.Context) {
		perPodReservation := int(float64(memCapacityMb) * 0.01)
		// underutilizedNodes are 10% full
		underutilizedPerNodeReplicas := 10
		// fullNodes are 70% full
		fullPerNodeReplicas := 70
		totalNodes := maxNodes
		underutilizedRatio := 0.3
		maxDelta := 30

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)

		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, f.ClientSet, totalNodes, largeResizeTimeout))

		// annotate all nodes with no-scale-down
		ScaleDownDisabledKey := "cluster-autoscaler.kubernetes.io/scale-down-disabled"

		nodes, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{
			FieldSelector: fields.Set{
				"spec.unschedulable": "false",
			}.AsSelector().String(),
		})

		framework.ExpectNoError(err)
		framework.ExpectNoError(addAnnotation(ctx, f, nodes.Items, ScaleDownDisabledKey, "true"))

		// distribute pods using replication controllers taking up space that should
		// be empty after pods are distributed
		underutilizedNodesNum := int(float64(maxNodes) * underutilizedRatio)
		fullNodesNum := totalNodes - underutilizedNodesNum

		podDistribution := []podBatch{
			{numNodes: fullNodesNum, podsPerNode: fullPerNodeReplicas},
			{numNodes: underutilizedNodesNum, podsPerNode: underutilizedPerNodeReplicas}}

		distributeLoad(ctx, f, f.Namespace.Name, "10-70", podDistribution, perPodReservation,
			int(0.95*float64(memCapacityMb)), map[string]string{}, largeScaleUpTimeout)

		// enable scale down again
		framework.ExpectNoError(addAnnotation(ctx, f, nodes.Items, ScaleDownDisabledKey, "false"))

		// wait for scale down to start. Node deletion takes a long time, so we just
		// wait for maximum of 30 nodes deleted
		nodesToScaleDownCount := int(float64(totalNodes) * 0.1)
		if nodesToScaleDownCount > maxDelta {
			nodesToScaleDownCount = maxDelta
		}
		expectedSize := totalNodes - nodesToScaleDownCount
		timeout := time.Duration(nodesToScaleDownCount)*time.Minute + scaleDownTimeout
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet, func(size int) bool {
			return size <= expectedSize
		}, timeout))
	})

	f.It("shouldn't scale down with underutilized nodes due to host port conflicts", feature.ClusterAutoscalerScalability5, func(ctx context.Context) {
		fullReservation := int(float64(memCapacityMb) * 0.9)
		hostPortPodReservation := int(float64(memCapacityMb) * 0.3)
		totalNodes := maxNodes
		reservedPort := 4321

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, f.ClientSet, totalNodes, largeResizeTimeout))
		divider := int(float64(totalNodes) * 0.7)
		fullNodesCount := divider
		underutilizedNodesCount := totalNodes - fullNodesCount

		ginkgo.By("Reserving full nodes")
		// run RC1 w/o host port
		cleanup := ReserveMemory(ctx, f, "filling-pod", fullNodesCount, fullNodesCount*fullReservation, true, largeScaleUpTimeout*2)
		defer cleanup()

		ginkgo.By("Reserving host ports on remaining nodes")
		// run RC2 w/ host port
		ginkgo.DeferCleanup(createHostPortPodsWithMemory, f, "underutilizing-host-port-pod", underutilizedNodesCount, reservedPort, underutilizedNodesCount*hostPortPodReservation, largeScaleUpTimeout)

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
		// wait and check scale down doesn't occur
		ginkgo.By(fmt.Sprintf("Sleeping %v minutes...", scaleDownTimeout.Minutes()))
		time.Sleep(scaleDownTimeout)

		ginkgo.By("Checking if the number of nodes is as expected")
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		klog.Infof("Nodes: %v, expected: %v", len(nodes.Items), totalNodes)
		gomega.Expect(nodes.Items).To(gomega.HaveLen(totalNodes))
	})

	f.It("CA ignores unschedulable pods while scheduling schedulable pods", feature.ClusterAutoscalerScalability6, func(ctx context.Context) {
		// Start a number of pods saturating existing nodes.
		perNodeReservation := int(float64(memCapacityMb) * 0.80)
		replicasPerNode := 10
		initialPodReplicas := nodeCount * replicasPerNode
		initialPodsTotalMemory := nodeCount * perNodeReservation
		reservationCleanup := ReserveMemory(ctx, f, "initial-pod", initialPodReplicas, initialPodsTotalMemory, true /* wait for pods to run */, memoryReservationTimeout)
		ginkgo.DeferCleanup(reservationCleanup)
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))

		// Configure a number of unschedulable pods.
		unschedulableMemReservation := memCapacityMb * 2
		unschedulablePodReplicas := 1000
		totalMemReservation := unschedulableMemReservation * unschedulablePodReplicas
		timeToWait := 5 * time.Minute
		podsConfig := reserveMemoryRCConfig(f, "unschedulable-pod", unschedulablePodReplicas, totalMemReservation, timeToWait)
		_ = e2erc.RunRC(ctx, *podsConfig) // Ignore error (it will occur because pods are unschedulable)
		ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, podsConfig.Name)

		// Ensure that no new nodes have been added so far.
		readyNodeCount, _ := e2enode.TotalReady(ctx, f.ClientSet)
		gomega.Expect(readyNodeCount).To(gomega.Equal(nodeCount))

		// Start a number of schedulable pods to ensure CA reacts.
		additionalNodes := maxNodes - nodeCount
		replicas := additionalNodes * replicasPerNode
		totalMemory := additionalNodes * perNodeReservation
		rcConfig := reserveMemoryRCConfig(f, "extra-pod", replicas, totalMemory, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes)
		config := createScaleUpTestConfig(nodeCount, initialPodReplicas, rcConfig, expectedResult)

		// Test that scale up happens, allowing 1000 unschedulable pods not to be scheduled.
		testCleanup := simpleScaleUpTestWithTolerance(ctx, f, config, 0, unschedulablePodReplicas)
		ginkgo.DeferCleanup(testCleanup)
	})

})

func anyKey(input map[string]int) string {
	for k := range input {
		return k
	}
	return ""
}

func simpleScaleUpTestWithTolerance(ctx context.Context, f *framework.Framework, config *scaleUpTestConfig, tolerateMissingNodeCount int, tolerateMissingPodCount int) func() error {
	// resize cluster to start size
	// run rc based on config
	ginkgo.By(fmt.Sprintf("Running RC %v from config", config.extraPods.Name))
	start := time.Now()
	framework.ExpectNoError(e2erc.RunRC(ctx, *config.extraPods))
	// check results
	if tolerateMissingNodeCount > 0 {
		// Tolerate some number of nodes not to be created.
		minExpectedNodeCount := config.expectedResult.nodes - tolerateMissingNodeCount
		framework.ExpectNoError(WaitForClusterSizeFunc(ctx, f.ClientSet,
			func(size int) bool { return size >= minExpectedNodeCount }, scaleUpTimeout))
	} else {
		framework.ExpectNoError(e2enode.WaitForReadyNodes(ctx, f.ClientSet, config.expectedResult.nodes, scaleUpTimeout))
	}
	klog.Infof("cluster is increased")
	if tolerateMissingPodCount > 0 {
		framework.ExpectNoError(waitForCaPodsReadyInNamespace(ctx, f, f.ClientSet, tolerateMissingPodCount))
	} else {
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, f.ClientSet))
	}
	timeTrack(start, fmt.Sprintf("Scale up to %v", config.expectedResult.nodes))
	return func() error {
		return e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, config.extraPods.Name)
	}
}

func simpleScaleUpTest(ctx context.Context, f *framework.Framework, config *scaleUpTestConfig) func() error {
	return simpleScaleUpTestWithTolerance(ctx, f, config, 0, 0)
}

func reserveMemoryRCConfig(f *framework.Framework, id string, replicas, megabytes int, timeout time.Duration) *testutils.RCConfig {
	return &testutils.RCConfig{
		Client:     f.ClientSet,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    timeout,
		Image:      imageutils.GetPauseImageName(),
		Replicas:   replicas,
		MemRequest: int64(1024 * 1024 * megabytes / replicas),
	}
}

func createScaleUpTestConfig(nodes, pods int, extraPods *testutils.RCConfig, expectedResult *clusterPredicates) *scaleUpTestConfig {
	return &scaleUpTestConfig{
		initialNodes:   nodes,
		initialPods:    pods,
		extraPods:      extraPods,
		expectedResult: expectedResult,
	}
}

func createClusterPredicates(nodes int) *clusterPredicates {
	return &clusterPredicates{
		nodes: nodes,
	}
}

func addAnnotation(ctx context.Context, f *framework.Framework, nodes []v1.Node, key, value string) error {
	for _, node := range nodes {
		oldData, err := json.Marshal(node)
		if err != nil {
			return err
		}

		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}
		node.Annotations[key] = value

		newData, err := json.Marshal(node)
		if err != nil {
			return err
		}

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		if err != nil {
			return err
		}

		_, err = f.ClientSet.CoreV1().Nodes().Patch(ctx, string(node.Name), types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
		if err != nil {
			return err
		}
	}
	return nil
}

func createHostPortPodsWithMemory(ctx context.Context, f *framework.Framework, id string, replicas, port, megabytes int, timeout time.Duration) func() error {
	ginkgo.By(fmt.Sprintf("Running RC which reserves host port and memory"))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &testutils.RCConfig{
		Client:     f.ClientSet,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    timeout,
		Image:      imageutils.GetPauseImageName(),
		Replicas:   replicas,
		HostPorts:  map[string]int{"port1": port},
		MemRequest: request,
	}
	err := e2erc.RunRC(ctx, *config)
	framework.ExpectNoError(err)
	return func() error {
		return e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, id)
	}
}

type podBatch struct {
	numNodes    int
	podsPerNode int
}

// distributeLoad distributes the pods in the way described by podDostribution,
// assuming all pods will have the same memory reservation and all nodes the same
// memory capacity. This allows us generate the load on the cluster in the exact
// way that we want.
//
// To achieve this we do the following:
// 1. Create replication controllers that eat up all the space that should be
// empty after setup, making sure they end up on different nodes by specifying
// conflicting host port
// 2. Create target RC that will generate the load on the cluster
// 3. Remove the rcs created in 1.
func distributeLoad(ctx context.Context, f *framework.Framework, namespace string, id string, podDistribution []podBatch,
	podMemRequestMegabytes int, nodeMemCapacity int, labels map[string]string, timeout time.Duration) {
	port := 8013
	// Create load-distribution RCs with one pod per node, reserving all remaining
	// memory to force the distribution of pods for the target RCs.
	// The load-distribution RCs will be deleted on function return.
	totalPods := 0
	for i, podBatch := range podDistribution {
		totalPods += podBatch.numNodes * podBatch.podsPerNode
		remainingMem := nodeMemCapacity - podBatch.podsPerNode*podMemRequestMegabytes
		replicas := podBatch.numNodes
		cleanup := createHostPortPodsWithMemory(ctx, f, fmt.Sprintf("load-distribution%d", i), replicas, port, remainingMem*replicas, timeout)
		defer cleanup()
	}
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, f.ClientSet))
	// Create the target RC
	rcConfig := reserveMemoryRCConfig(f, id, totalPods, totalPods*podMemRequestMegabytes, timeout)
	framework.ExpectNoError(e2erc.RunRC(ctx, *rcConfig))
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, f.ClientSet))
	ginkgo.DeferCleanup(e2erc.DeleteRCAndWaitForGC, f.ClientSet, f.Namespace.Name, id)
}

func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	klog.Infof("%s took %s", name, elapsed)
}
