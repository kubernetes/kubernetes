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
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	memoryReservationTimeout = 5 * time.Minute
	largeResizeTimeout       = 8 * time.Minute
	largeScaleUpTimeout      = 10 * time.Minute
	largeScaleDownTimeout    = 20 * time.Minute
	minute                   = 1 * time.Minute

	maxNodes = 1000
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

var _ = framework.KubeDescribe("Cluster size autoscaler scalability [Slow]", func() {
	f := framework.NewDefaultFramework("autoscaling")
	var c clientset.Interface
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int
	var originalSizes map[string]int
	var sum int

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke", "kubemark")

		// Check if Cloud Autoscaler is enabled by trying to get its ConfigMap.
		_, err := f.ClientSet.CoreV1().ConfigMaps("kube-system").Get("cluster-autoscaler-status", metav1.GetOptions{})
		if err != nil {
			framework.Skipf("test expects Cluster Autoscaler to be enabled")
		}

		c = f.ClientSet
		if originalSizes == nil {
			originalSizes = make(map[string]int)
			sum = 0
			for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
				size, err := framework.GroupSize(mig)
				framework.ExpectNoError(err)
				By(fmt.Sprintf("Initial size of %s: %d", mig, size))
				originalSizes[mig] = size
				sum += size
			}
		}

		framework.ExpectNoError(framework.WaitForReadyNodes(c, sum, scaleUpTimeout))

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())
		cpu := nodes.Items[0].Status.Capacity[v1.ResourceCPU]
		mem := nodes.Items[0].Status.Capacity[v1.ResourceMemory]
		coresPerNode = int((&cpu).MilliValue() / 1000)
		memCapacityMb = int((&mem).Value() / 1024 / 1024)

		Expect(nodeCount).Should(Equal(sum))

		if framework.ProviderIs("gke") {
			val, err := isAutoscalerEnabled(3)
			framework.ExpectNoError(err)
			if !val {
				err = enableAutoscaler("default-pool", 3, 5)
				framework.ExpectNoError(err)
			}
		}
	})

	AfterEach(func() {
		By(fmt.Sprintf("Restoring initial size of the cluster"))
		setMigSizes(originalSizes)
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount, scaleDownTimeout))
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		s := time.Now()
	makeSchedulableLoop:
		for start := time.Now(); time.Since(start) < makeSchedulableTimeout; time.Sleep(makeSchedulableDelay) {
			for _, n := range nodes.Items {
				err = makeNodeSchedulable(c, &n, true)
				switch err.(type) {
				case CriticalAddonsOnlyError:
					continue makeSchedulableLoop
				default:
					framework.ExpectNoError(err)
				}
			}
			break
		}
		glog.Infof("Made nodes schedulable again in %v", time.Since(s).String())
	})

	It("should scale up at all [Feature:ClusterAutoscalerScalability1]", func() {
		perNodeReservation := int(float64(memCapacityMb) * 0.95)
		replicasPerNode := 10

		additionalNodes := maxNodes - nodeCount
		replicas := additionalNodes * replicasPerNode
		additionalReservation := additionalNodes * perNodeReservation

		// saturate cluster
		reservationCleanup := ReserveMemory(f, "some-pod", nodeCount*2, nodeCount*perNodeReservation, true, memoryReservationTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		// configure pending pods & expected scale up
		rcConfig := reserveMemoryRCConfig(f, "extra-pod-1", replicas, additionalReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes)
		config := createScaleUpTestConfig(nodeCount, nodeCount, rcConfig, expectedResult)

		// run test
		testCleanup := simpleScaleUpTest(f, config)
		defer testCleanup()
	})

	It("should scale up twice [Feature:ClusterAutoscalerScalability2]", func() {
		perNodeReservation := int(float64(memCapacityMb) * 0.95)
		replicasPerNode := 10
		additionalNodes1 := int(math.Ceil(0.7 * maxNodes))
		additionalNodes2 := int(math.Ceil(0.25 * maxNodes))
		if additionalNodes1+additionalNodes2 > maxNodes {
			additionalNodes2 = maxNodes - additionalNodes1
		}

		replicas1 := additionalNodes1 * replicasPerNode
		replicas2 := additionalNodes2 * replicasPerNode

		glog.Infof("cores per node: %v", coresPerNode)

		// saturate cluster
		initialReplicas := nodeCount
		reservationCleanup := ReserveMemory(f, "some-pod", initialReplicas, nodeCount*perNodeReservation, true, memoryReservationTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		glog.Infof("Reserved successfully")

		// configure pending pods & expected scale up #1
		rcConfig := reserveMemoryRCConfig(f, "extra-pod-1", replicas1, additionalNodes1*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes1)
		config := createScaleUpTestConfig(nodeCount, nodeCount, rcConfig, expectedResult)

		// run test #1
		tolerateUnreadyNodes := additionalNodes1 / 20
		tolerateUnreadyPods := (initialReplicas + replicas1) / 20
		testCleanup1 := simpleScaleUpTestWithTolerance(f, config, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup1()

		glog.Infof("Scaled up once")

		// configure pending pods & expected scale up #2
		rcConfig2 := reserveMemoryRCConfig(f, "extra-pod-2", replicas2, additionalNodes2*perNodeReservation, largeScaleUpTimeout)
		expectedResult2 := createClusterPredicates(nodeCount + additionalNodes1 + additionalNodes2)
		config2 := createScaleUpTestConfig(nodeCount+additionalNodes1, nodeCount+additionalNodes2, rcConfig2, expectedResult2)

		// run test #2
		tolerateUnreadyNodes = maxNodes / 20
		tolerateUnreadyPods = (initialReplicas + replicas1 + replicas2) / 20
		testCleanup2 := simpleScaleUpTestWithTolerance(f, config2, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup2()

		glog.Infof("Scaled up twice")
	})

	It("should scale down empty nodes [Feature:ClusterAutoscalerScalability3]", func() {
		perNodeReservation := int(float64(memCapacityMb) * 0.7)
		replicas := int(math.Ceil(maxNodes * 0.7))
		totalNodes := maxNodes

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(framework.WaitForReadyNodes(f.ClientSet, totalNodes, largeResizeTimeout))

		// run replicas
		rcConfig := reserveMemoryRCConfig(f, "some-pod", replicas, replicas*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(totalNodes)
		config := createScaleUpTestConfig(totalNodes, totalNodes, rcConfig, expectedResult)
		tolerateUnreadyNodes := totalNodes / 10
		tolerateUnreadyPods := replicas / 10
		testCleanup := simpleScaleUpTestWithTolerance(f, config, tolerateUnreadyNodes, tolerateUnreadyPods)
		defer testCleanup()

		// check if empty nodes are scaled down
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool {
				return size <= replicas+3 // leaving space for non-evictable kube-system pods
			}, scaleDownTimeout))
	})

	It("should scale down underutilized nodes [Feature:ClusterAutoscalerScalability4]", func() {
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

		framework.ExpectNoError(framework.WaitForReadyNodes(f.ClientSet, totalNodes, largeResizeTimeout))

		// annotate all nodes with no-scale-down
		ScaleDownDisabledKey := "cluster-autoscaler.kubernetes.io/scale-down-disabled"

		nodes, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{
			FieldSelector: fields.Set{
				"spec.unschedulable": "false",
			}.AsSelector().String(),
		})

		framework.ExpectNoError(err)
		framework.ExpectNoError(addAnnotation(f, nodes.Items, ScaleDownDisabledKey, "true"))

		// distribute pods using replication controllers taking up space that should
		// be empty after pods are distributed
		underutilizedNodesNum := int(float64(maxNodes) * underutilizedRatio)
		fullNodesNum := totalNodes - underutilizedNodesNum

		podDistribution := []podBatch{
			{numNodes: fullNodesNum, podsPerNode: fullPerNodeReplicas},
			{numNodes: underutilizedNodesNum, podsPerNode: underutilizedPerNodeReplicas}}

		cleanup := distributeLoad(f, f.Namespace.Name, "10-70", podDistribution, perPodReservation,
			int(0.95*float64(memCapacityMb)), map[string]string{}, largeScaleUpTimeout)
		defer cleanup()

		// enable scale down again
		framework.ExpectNoError(addAnnotation(f, nodes.Items, ScaleDownDisabledKey, "false"))

		// wait for scale down to start. Node deletion takes a long time, so we just
		// wait for maximum of 30 nodes deleted
		nodesToScaleDownCount := int(float64(totalNodes) * 0.1)
		if nodesToScaleDownCount > maxDelta {
			nodesToScaleDownCount = maxDelta
		}
		expectedSize := totalNodes - nodesToScaleDownCount
		timeout := time.Duration(nodesToScaleDownCount)*time.Minute + scaleDownTimeout
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet, func(size int) bool {
			return size <= expectedSize
		}, timeout))
	})

	It("shouldn't scale down with underutilized nodes due to host port conflicts [Feature:ClusterAutoscalerScalability5]", func() {
		fullReservation := int(float64(memCapacityMb) * 0.9)
		hostPortPodReservation := int(float64(memCapacityMb) * 0.3)
		totalNodes := maxNodes
		reservedPort := 4321

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(framework.WaitForReadyNodes(f.ClientSet, totalNodes, largeResizeTimeout))
		divider := int(float64(totalNodes) * 0.7)
		fullNodesCount := divider
		underutilizedNodesCount := totalNodes - fullNodesCount

		By("Reserving full nodes")
		// run RC1 w/o host port
		cleanup := ReserveMemory(f, "filling-pod", fullNodesCount, fullNodesCount*fullReservation, true, largeScaleUpTimeout*2)
		defer cleanup()

		By("Reserving host ports on remaining nodes")
		// run RC2 w/ host port
		cleanup2 := createHostPortPodsWithMemory(f, "underutilizing-host-port-pod", underutilizedNodesCount, reservedPort, underutilizedNodesCount*hostPortPodReservation, largeScaleUpTimeout)
		defer cleanup2()

		waitForAllCaPodsReadyInNamespace(f, c)
		// wait and check scale down doesn't occur
		By(fmt.Sprintf("Sleeping %v minutes...", scaleDownTimeout.Minutes()))
		time.Sleep(scaleDownTimeout)

		By("Checking if the number of nodes is as expected")
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		glog.Infof("Nodes: %v, expected: %v", len(nodes.Items), totalNodes)
		Expect(len(nodes.Items)).Should(Equal(totalNodes))
	})

	Specify("CA ignores unschedulable pods while scheduling schedulable pods [Feature:ClusterAutoscalerScalability6]", func() {
		// Start a number of pods saturating existing nodes.
		perNodeReservation := int(float64(memCapacityMb) * 0.80)
		replicasPerNode := 10
		initialPodReplicas := nodeCount * replicasPerNode
		initialPodsTotalMemory := nodeCount * perNodeReservation
		reservationCleanup := ReserveMemory(f, "initial-pod", initialPodReplicas, initialPodsTotalMemory, true /* wait for pods to run */, memoryReservationTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		// Configure a number of unschedulable pods.
		unschedulableMemReservation := memCapacityMb * 2
		unschedulablePodReplicas := 1000
		totalMemReservation := unschedulableMemReservation * unschedulablePodReplicas
		timeToWait := 5 * time.Minute
		podsConfig := reserveMemoryRCConfig(f, "unschedulable-pod", unschedulablePodReplicas, totalMemReservation, timeToWait)
		framework.RunRC(*podsConfig) // Ignore error (it will occur because pods are unschedulable)
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, podsConfig.Name)

		// Ensure that no new nodes have been added so far.
		Expect(framework.NumberOfReadyNodes(f.ClientSet)).To(Equal(nodeCount))

		// Start a number of schedulable pods to ensure CA reacts.
		additionalNodes := maxNodes - nodeCount
		replicas := additionalNodes * replicasPerNode
		totalMemory := additionalNodes * perNodeReservation
		rcConfig := reserveMemoryRCConfig(f, "extra-pod", replicas, totalMemory, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes)
		config := createScaleUpTestConfig(nodeCount, initialPodReplicas, rcConfig, expectedResult)

		// Test that scale up happens, allowing 1000 unschedulable pods not to be scheduled.
		testCleanup := simpleScaleUpTestWithTolerance(f, config, 0, unschedulablePodReplicas)
		defer testCleanup()
	})

})

func anyKey(input map[string]int) string {
	for k := range input {
		return k
	}
	return ""
}

func simpleScaleUpTestWithTolerance(f *framework.Framework, config *scaleUpTestConfig, tolerateMissingNodeCount int, tolerateMissingPodCount int) func() error {
	// resize cluster to start size
	// run rc based on config
	By(fmt.Sprintf("Running RC %v from config", config.extraPods.Name))
	start := time.Now()
	framework.ExpectNoError(framework.RunRC(*config.extraPods))
	// check results
	if tolerateMissingNodeCount > 0 {
		// Tolerate some number of nodes not to be created.
		minExpectedNodeCount := config.expectedResult.nodes - tolerateMissingNodeCount
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= minExpectedNodeCount }, scaleUpTimeout))
	} else {
		framework.ExpectNoError(framework.WaitForReadyNodes(f.ClientSet, config.expectedResult.nodes, scaleUpTimeout))
	}
	glog.Infof("cluster is increased")
	if tolerateMissingPodCount > 0 {
		framework.ExpectNoError(waitForCaPodsReadyInNamespace(f, f.ClientSet, tolerateMissingPodCount))
	} else {
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, f.ClientSet))
	}
	timeTrack(start, fmt.Sprintf("Scale up to %v", config.expectedResult.nodes))
	return func() error {
		return framework.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, config.extraPods.Name)
	}
}

func simpleScaleUpTest(f *framework.Framework, config *scaleUpTestConfig) func() error {
	return simpleScaleUpTestWithTolerance(f, config, 0, 0)
}

func reserveMemoryRCConfig(f *framework.Framework, id string, replicas, megabytes int, timeout time.Duration) *testutils.RCConfig {
	return &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        timeout,
		Image:          imageutils.GetPauseImageName(),
		Replicas:       replicas,
		MemRequest:     int64(1024 * 1024 * megabytes / replicas),
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

func addAnnotation(f *framework.Framework, nodes []v1.Node, key, value string) error {
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

		_, err = f.ClientSet.CoreV1().Nodes().Patch(string(node.Name), types.StrategicMergePatchType, patchBytes)
		if err != nil {
			return err
		}
	}
	return nil
}

func createHostPortPodsWithMemory(f *framework.Framework, id string, replicas, port, megabytes int, timeout time.Duration) func() error {
	By(fmt.Sprintf("Running RC which reserves host port and memory"))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        timeout,
		Image:          imageutils.GetPauseImageName(),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": port},
		MemRequest:     request,
	}
	err := framework.RunRC(*config)
	framework.ExpectNoError(err)
	return func() error {
		return framework.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, id)
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
// 2. Create targer RC that will generate the load on the cluster
// 3. Remove the rcs created in 1.
func distributeLoad(f *framework.Framework, namespace string, id string, podDistribution []podBatch,
	podMemRequestMegabytes int, nodeMemCapacity int, labels map[string]string, timeout time.Duration) func() error {
	port := 8013
	// Create load-distribution RCs with one pod per node, reserving all remaining
	// memory to force the distribution of pods for the target RCs.
	// The load-distribution RCs will be deleted on function return.
	totalPods := 0
	for i, podBatch := range podDistribution {
		totalPods += podBatch.numNodes * podBatch.podsPerNode
		remainingMem := nodeMemCapacity - podBatch.podsPerNode*podMemRequestMegabytes
		replicas := podBatch.numNodes
		cleanup := createHostPortPodsWithMemory(f, fmt.Sprintf("load-distribution%d", i), replicas, port, remainingMem*replicas, timeout)
		defer cleanup()
	}
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, f.ClientSet))
	// Create the target RC
	rcConfig := reserveMemoryRCConfig(f, id, totalPods, totalPods*podMemRequestMegabytes, timeout)
	framework.ExpectNoError(framework.RunRC(*rcConfig))
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, f.ClientSet))
	return func() error {
		return framework.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, id)
	}
}

func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	glog.Infof("%s took %s", name, elapsed)
}
