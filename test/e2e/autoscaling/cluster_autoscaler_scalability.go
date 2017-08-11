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

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	largeResizeTimeout    = 10 * time.Minute
	largeScaleUpTimeout   = 10 * time.Minute
	largeScaleDownTimeout = 20 * time.Minute
	minute                = 1 * time.Minute

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

		framework.ExpectNoError(framework.WaitForClusterSize(c, sum, scaleUpTimeout))

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
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount, scaleDownTimeout))
		nodes, err := c.Core().Nodes().List(metav1.ListOptions{})
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
		glog.Infof("Made nodes schedulable again in %v", time.Now().Sub(s).String())
	})

	It("should scale up at all [Feature:ClusterAutoscalerScalability1]", func() {
		perNodeReservation := int(float64(memCapacityMb) * 0.95)
		replicasPerNode := 10

		additionalNodes := maxNodes - nodeCount
		replicas := additionalNodes * replicasPerNode
		additionalReservation := additionalNodes * perNodeReservation

		// saturate cluster
		reservationCleanup := ReserveMemory(f, "some-pod", nodeCount*2, nodeCount*perNodeReservation, true, scaleUpTimeout)
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
		additionalNodes1 := int(0.7 * maxNodes)
		additionalNodes2 := int(0.25 * maxNodes)

		replicas1 := additionalNodes1 * replicasPerNode
		replicas2 := additionalNodes2 * replicasPerNode

		glog.Infof("cores per node: %v", coresPerNode)

		// saturate cluster
		reservationCleanup := ReserveMemory(f, "some-pod", nodeCount, nodeCount*perNodeReservation, true, scaleUpTimeout)
		defer reservationCleanup()
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		glog.Infof("Reserved successfully")

		// configure pending pods & expected scale up #1
		rcConfig := reserveMemoryRCConfig(f, "extra-pod-1", replicas1, additionalNodes1*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(nodeCount + additionalNodes1)
		config := createScaleUpTestConfig(nodeCount, nodeCount, rcConfig, expectedResult)

		epsilon := 0.05

		// run test #1
		testCleanup1 := simpleScaleUpTestWithEpsilon(f, config, epsilon)
		defer testCleanup1()

		glog.Infof("Scaled up once")

		// configure pending pods & expected scale up #2
		rcConfig2 := reserveMemoryRCConfig(f, "extra-pod-2", replicas2, additionalNodes2*perNodeReservation, largeScaleUpTimeout)
		expectedResult2 := createClusterPredicates(nodeCount + additionalNodes1 + additionalNodes2)
		config2 := createScaleUpTestConfig(nodeCount+additionalNodes1, nodeCount+additionalNodes2, rcConfig2, expectedResult2)

		// run test #2
		testCleanup2 := simpleScaleUpTestWithEpsilon(f, config2, epsilon)
		defer testCleanup2()

		glog.Infof("Scaled up twice")
	})

	It("should scale down empty nodes [Feature:ClusterAutoscalerScalability3]", func() {
		perNodeReservation := int(float64(memCapacityMb) * 0.7)
		replicas := int(float64(maxNodes) * 0.7)
		totalNodes := maxNodes

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(framework.WaitForClusterSize(f.ClientSet, totalNodes, largeResizeTimeout))

		// run replicas
		rcConfig := reserveMemoryRCConfig(f, "some-pod", replicas, replicas*perNodeReservation, largeScaleUpTimeout)
		expectedResult := createClusterPredicates(totalNodes)
		config := createScaleUpTestConfig(totalNodes, totalNodes, rcConfig, expectedResult)
		testCleanup := simpleScaleUpTestWithEpsilon(f, config, 0.1)
		defer testCleanup()

		// check if empty nodes are scaled down
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool {
				return size <= replicas+3 // leaving space for non-evictable kube-system pods
			}, scaleDownTimeout))
	})

	It("should scale down underutilized nodes [Feature:ClusterAutoscalerScalability4]", func() {
		underutilizedReservation := int64(float64(memCapacityMb) * 0.01)
		fullReservation := int64(float64(memCapacityMb) * 0.8)
		perNodeReplicas := 10
		totalNodes := maxNodes

		// resize cluster to totalNodes
		newSizes := map[string]int{
			anyKey(originalSizes): totalNodes,
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(framework.WaitForClusterSize(f.ClientSet, totalNodes, largeResizeTimeout))

		// annotate all nodes with no-scale-down
		ScaleDownDisabledKey := "cluster-autoscaler.kubernetes.io/scale-down-disabled"

		nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{
			FieldSelector: fields.Set{
				"spec.unschedulable": "false",
			}.AsSelector().String(),
		})

		framework.ExpectNoError(addAnnotation(f, nodes.Items, ScaleDownDisabledKey, "true"))

		// distribute pods (using taints)
		divider := int(float64(len(nodes.Items)) * 0.7)

		fullNodes := nodes.Items[:divider]
		underutilizedNodes := nodes.Items[divider:]

		framework.ExpectNoError(makeUnschedulable(f, underutilizedNodes))

		testId2 := "full"
		labels2 := map[string]string{"test_id": testId2}
		cleanup2, err := runReplicatedPodOnEachNodeWithCleanup(f, fullNodes, f.Namespace.Name, 1, "filling-pod", labels2, fullReservation)
		defer cleanup2()
		framework.ExpectNoError(err)

		framework.ExpectNoError(makeUnschedulable(f, fullNodes))

		testId := "underutilized"
		labels := map[string]string{"test_id": testId}
		cleanup, err := runReplicatedPodOnEachNodeWithCleanup(f, underutilizedNodes, f.Namespace.Name, perNodeReplicas, "underutilizing-pod", labels, underutilizedReservation)
		defer cleanup()
		framework.ExpectNoError(err)

		framework.ExpectNoError(makeSchedulable(f, nodes.Items))
		framework.ExpectNoError(addAnnotation(f, nodes.Items, ScaleDownDisabledKey, "false"))

		// wait for scale down
		expectedSize := int(float64(totalNodes) * 0.85)
		nodesToScaleDownCount := totalNodes - expectedSize
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
		framework.ExpectNoError(framework.WaitForClusterSize(f.ClientSet, totalNodes, largeResizeTimeout))
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

})

func makeUnschedulable(f *framework.Framework, nodes []v1.Node) error {
	for _, node := range nodes {
		err := makeNodeUnschedulable(f.ClientSet, &node)
		if err != nil {
			return err
		}
	}
	return nil
}

func makeSchedulable(f *framework.Framework, nodes []v1.Node) error {
	for _, node := range nodes {
		err := makeNodeSchedulable(f.ClientSet, &node, false)
		if err != nil {
			return err
		}
	}
	return nil
}

func anyKey(input map[string]int) string {
	for k := range input {
		return k
	}
	return ""
}

func simpleScaleUpTestWithEpsilon(f *framework.Framework, config *scaleUpTestConfig, epsilon float64) func() error {
	// resize cluster to start size
	// run rc based on config
	By(fmt.Sprintf("Running RC %v from config", config.extraPods.Name))
	start := time.Now()
	framework.ExpectNoError(framework.RunRC(*config.extraPods))
	// check results
	if epsilon > 0 && epsilon < 1 {
		// Tolerate some number of nodes not to be created.
		minExpectedNodeCount := int(float64(config.expectedResult.nodes) - epsilon*float64(config.expectedResult.nodes))
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= minExpectedNodeCount }, scaleUpTimeout))
	} else {
		framework.ExpectNoError(framework.WaitForClusterSize(f.ClientSet, config.expectedResult.nodes, scaleUpTimeout))
	}
	glog.Infof("cluster is increased")
	if epsilon > 0 && epsilon < 0 {
		framework.ExpectNoError(waitForCaPodsReadyInNamespace(f, f.ClientSet, int(epsilon*float64(config.extraPods.Replicas)+1)))
	} else {
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, f.ClientSet))
	}
	timeTrack(start, fmt.Sprintf("Scale up to %v", config.expectedResult.nodes))
	return func() error {
		return framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, config.extraPods.Name)
	}
}

func simpleScaleUpTest(f *framework.Framework, config *scaleUpTestConfig) func() error {
	return simpleScaleUpTestWithEpsilon(f, config, 0)
}

func reserveMemoryRCConfig(f *framework.Framework, id string, replicas, megabytes int, timeout time.Duration) *testutils.RCConfig {
	return &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        timeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
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

		_, err = f.ClientSet.Core().Nodes().Patch(string(node.Name), types.StrategicMergePatchType, patchBytes)
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
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": port},
		MemRequest:     request,
	}
	err := framework.RunRC(*config)
	framework.ExpectNoError(err)
	return func() error {
		return framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, id)
	}
}

func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	glog.Infof("%s took %s", name, elapsed)
}
