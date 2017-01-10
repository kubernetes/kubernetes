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

package e2e

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	defaultTimeout   = 3 * time.Minute
	resizeTimeout    = 5 * time.Minute
	scaleUpTimeout   = 5 * time.Minute
	scaleDownTimeout = 15 * time.Minute

	gkeEndpoint      = "https://test-container.sandbox.googleapis.com"
	gkeUpdateTimeout = 15 * time.Minute
)

var _ = framework.KubeDescribe("Cluster size autoscaling [Slow]", func() {
	f := framework.NewDefaultFramework("autoscaling")
	var c clientset.Interface
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int
	var originalSizes map[string]int

	BeforeEach(func() {
		c = f.ClientSet
		framework.SkipUnlessProviderIs("gce", "gke")

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())
		cpu := nodes.Items[0].Status.Capacity[v1.ResourceCPU]
		mem := nodes.Items[0].Status.Capacity[v1.ResourceMemory]
		coresPerNode = int((&cpu).MilliValue() / 1000)
		memCapacityMb = int((&mem).Value() / 1024 / 1024)

		originalSizes = make(map[string]int)
		sum := 0
		for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			size, err := GroupSize(mig)
			framework.ExpectNoError(err)
			By(fmt.Sprintf("Initial size of %s: %d", mig, size))
			originalSizes[mig] = size
			sum += size
		}
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
	})

	It("shouldn't increase cluster size if pending pod is too large [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		By("Creating unschedulable pod")
		ReserveMemory(f, "memory-reservation", 1, memCapacityMb, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		By("Waiting for scale up hoping it won't happen")
		// Verfiy, that the appropreate event was generated.
		eventFound := false
	EventsLoop:
		for start := time.Now(); time.Since(start) < scaleUpTimeout; time.Sleep(20 * time.Second) {
			By("Waiting for NotTriggerScaleUp event")
			events, err := f.ClientSet.Core().Events(f.Namespace.Name).List(v1.ListOptions{})
			framework.ExpectNoError(err)

			for _, e := range events.Items {
				if e.InvolvedObject.Kind == "Pod" && e.Reason == "NotTriggerScaleUp" && strings.Contains(e.Message, "it wouldn't fit if a new node is added") {
					By("NotTriggerScaleUp event found")
					eventFound = true
					break EventsLoop
				}
			}
		}
		Expect(eventFound).Should(Equal(true))
		// Verify, that cluster size is not changed.
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size <= nodeCount }, time.Second))
	})

	It("should increase cluster size if pending pods are small [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		ReserveMemory(f, "memory-reservation", 100, nodeCount*memCapacityMb, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Verify, that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("should increase cluster size if pending pods are small and there is another node pool that is not autoscaled [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Creating new node-pool with one n1-standard-4 machine")
		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-4", 1)
		defer deleteNodePool(extraPoolName)
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount+1, resizeTimeout))
		glog.Infof("Not enabling cluster autoscaler for the node pool (on purpose).")

		ReserveMemory(f, "memory-reservation", 100, nodeCount*memCapacityMb, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Verify, that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("should disable node pool autoscaling [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Creating new node-pool with one n1-standard-4 machine")
		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-4", 1)
		defer deleteNodePool(extraPoolName)
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount+1, resizeTimeout))
		framework.ExpectNoError(enableAutoscaler(extraPoolName, 1, 2))
		framework.ExpectNoError(disableAutoscaler(extraPoolName, 1, 2))
	})

	It("should increase cluster size if pods are pending due to host port conflict [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		CreateHostPortPods(f, "host-port", nodeCount+2, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "host-port")

		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+2 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("should add node to the particular mig [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		labels := map[string]string{"cluster-autoscaling-test.special-node": "true"}

		By("Finding the smallest MIG")
		minMig := ""
		minSize := nodeCount
		for mig, size := range originalSizes {
			if size <= minSize {
				minMig = mig
				minSize = size
			}
		}

		removeLabels := func(nodesToClean sets.String) {
			By("Removing labels from nodes")
			updateNodeLabels(c, nodesToClean, nil, labels)
		}

		nodes, err := GetGroupNodes(minMig)
		framework.ExpectNoError(err)
		nodesSet := sets.NewString(nodes...)
		defer removeLabels(nodesSet)
		By(fmt.Sprintf("Annotating nodes of the smallest MIG(%s): %v", minMig, nodes))
		updateNodeLabels(c, nodesSet, labels, nil)

		CreateNodeSelectorPods(f, "node-selector", minSize+1, labels, false)

		By("Waiting for new node to appear and annotating it")
		WaitForGroupSize(minMig, int32(minSize+1))
		// Verify, that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout))

		newNodes, err := GetGroupNodes(minMig)
		framework.ExpectNoError(err)
		newNodesSet := sets.NewString(newNodes...)
		newNodesSet.Delete(nodes...)
		if len(newNodesSet) > 1 {
			By(fmt.Sprintf("Spotted following new nodes in %s: %v", minMig, newNodesSet))
			glog.Infof("Usually only 1 new node is expected, investigating")
			glog.Infof("Kubectl:%s\n", framework.RunKubectlOrDie("get", "nodes", "-o", "json"))
			if output, err := exec.Command("gcloud", "compute", "instances", "list",
				"--project="+framework.TestContext.CloudConfig.ProjectID,
				"--zone="+framework.TestContext.CloudConfig.Zone).Output(); err == nil {
				glog.Infof("Gcloud compute instances list: %s", output)
			} else {
				glog.Errorf("Failed to get instances list: %v", err)
			}

			for newNode := range newNodesSet {
				if output, err := exec.Command("gcloud", "compute", "instances", "describe",
					newNode,
					"--project="+framework.TestContext.CloudConfig.ProjectID,
					"--zone="+framework.TestContext.CloudConfig.Zone).Output(); err == nil {
					glog.Infof("Gcloud compute instances describe: %s", output)
				} else {
					glog.Errorf("Failed to get instances describe: %v", err)
				}
			}

			// TODO: possibly remove broken node from newNodesSet to prevent removeLabel from crashing.
			// However at this moment we DO WANT it to crash so that we don't check all test runs for the
			// rare behavior, but only the broken ones.
		}
		By(fmt.Sprintf("New nodes: %v\n", newNodesSet))
		registeredNodes := sets.NewString()
		for nodeName := range newNodesSet {
			node, err := f.ClientSet.Core().Nodes().Get(nodeName, metav1.GetOptions{})
			if err == nil && node != nil {
				registeredNodes.Insert(nodeName)
			} else {
				glog.Errorf("Failed to get node %v: %v", nodeName, err)
			}
		}
		By(fmt.Sprintf("Setting labels for registered new nodes: %v", registeredNodes.List()))
		updateNodeLabels(c, registeredNodes, labels, nil)
		defer removeLabels(registeredNodes)

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
		framework.ExpectNoError(framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "node-selector"))
	})

	It("should scale up correct target pool [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Creating new node-pool with one n1-standard-4 machine")
		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-4", 1)
		defer deleteNodePool(extraPoolName)
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount+1, resizeTimeout))
		framework.ExpectNoError(enableAutoscaler(extraPoolName, 1, 2))

		By("Creating rc with 2 pods too big to fit default-pool but fitting extra-pool")
		ReserveMemory(f, "memory-reservation", 2, 2*memCapacityMb, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Apparently GKE master is restarted couple minutes after the node pool is added
		// reseting all the timers in scale down code. Adding 5 extra minutes to workaround
		// this issue.
		// TODO: Remove the extra time when GKE restart is fixed.
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount+2, scaleUpTimeout+5*time.Minute))
	})

	It("should correctly scale down after a node is not needed [Feature:ClusterSizeAutoscalingScaleDown]", func() {
		By("Manually increase cluster size")
		increasedSize := 0
		newSizes := make(map[string]int)
		for key, val := range originalSizes {
			newSizes[key] = val + 2
			increasedSize += val + 2
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= increasedSize }, scaleUpTimeout))

		By("Some node should be removed")
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size < increasedSize }, scaleDownTimeout))
	})

	It("should correctly scale down after a node is not needed when there is non autoscaled pool[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Manually increase cluster size")
		increasedSize := 0
		newSizes := make(map[string]int)
		for key, val := range originalSizes {
			newSizes[key] = val + 2
			increasedSize += val + 2
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= increasedSize }, scaleUpTimeout))

		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-1", 3)
		defer deleteNodePool(extraPoolName)

		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= increasedSize+3 }, scaleUpTimeout))

		By("Some node should be removed")
		// Apparently GKE master is restarted couple minutes after the node pool is added
		// reseting all the timers in scale down code. Adding 10 extra minutes to workaround
		// this issue.
		// TODO: Remove the extra time when GKE restart is fixed.
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size < increasedSize+3 }, scaleDownTimeout+10*time.Minute))
	})
})

func getGKEClusterUrl() string {
	out, err := exec.Command("gcloud", "auth", "print-access-token").Output()
	framework.ExpectNoError(err)
	token := strings.Replace(string(out), "\n", "", -1)

	return fmt.Sprintf("%s/v1/projects/%s/zones/%s/clusters/%s?access_token=%s",
		gkeEndpoint,
		framework.TestContext.CloudConfig.ProjectID,
		framework.TestContext.CloudConfig.Zone,
		framework.TestContext.CloudConfig.Cluster,
		token)
}

func isAutoscalerEnabled(expectedMinNodeCountInTargetPool int) (bool, error) {
	resp, err := http.Get(getGKEClusterUrl())
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return false, err
	}
	strBody := string(body)
	glog.Infof("Cluster config %s", strBody)

	if strings.Contains(strBody, "\"minNodeCount\": "+strconv.Itoa(expectedMinNodeCountInTargetPool)) {
		return true, nil
	}
	return false, nil
}

func enableAutoscaler(nodePool string, minCount, maxCount int) error {

	if nodePool == "default-pool" {
		glog.Infof("Using gcloud to enable autoscaling for pool %s", nodePool)

		output, err := exec.Command("gcloud", "alpha", "container", "clusters", "update", framework.TestContext.CloudConfig.Cluster,
			"--enable-autoscaling",
			"--min-nodes="+strconv.Itoa(minCount),
			"--max-nodes="+strconv.Itoa(maxCount),
			"--node-pool="+nodePool,
			"--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).Output()

		if err != nil {
			return fmt.Errorf("Failed to enable autoscaling: %v", err)
		}
		glog.Infof("Config update result: %s", output)

	} else {
		glog.Infof("Using direct api access to enable autoscaling for pool %s", nodePool)
		updateRequest := "{" +
			" \"update\": {" +
			"  \"desiredNodePoolId\": \"" + nodePool + "\"," +
			"  \"desiredNodePoolAutoscaling\": {" +
			"   \"enabled\": \"true\"," +
			"   \"minNodeCount\": \"" + strconv.Itoa(minCount) + "\"," +
			"   \"maxNodeCount\": \"" + strconv.Itoa(maxCount) + "\"" +
			"  }" +
			" }" +
			"}"

		url := getGKEClusterUrl()
		glog.Infof("Using gke api url %s", url)
		putResult, err := doPut(url, updateRequest)
		if err != nil {
			return fmt.Errorf("Failed to put %s: %v", url, err)
		}
		glog.Infof("Config update result: %s", putResult)
	}

	for startTime := time.Now(); startTime.Add(gkeUpdateTimeout).After(time.Now()); time.Sleep(30 * time.Second) {
		if val, err := isAutoscalerEnabled(minCount); err == nil && val {
			return nil
		}
	}
	return fmt.Errorf("autoscaler not enabled")
}

func disableAutoscaler(nodePool string, minCount, maxCount int) error {

	if nodePool == "default-pool" {
		glog.Infof("Using gcloud to disable autoscaling for pool %s", nodePool)

		output, err := exec.Command("gcloud", "alpha", "container", "clusters", "update", framework.TestContext.CloudConfig.Cluster,
			"--no-enable-autoscaling",
			"--node-pool="+nodePool,
			"--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).Output()

		if err != nil {
			return fmt.Errorf("Failed to enable autoscaling: %v", err)
		}
		glog.Infof("Config update result: %s", output)

	} else {
		glog.Infof("Using direct api access to disable autoscaling for pool %s", nodePool)
		updateRequest := "{" +
			" \"update\": {" +
			"  \"desiredNodePoolId\": \"" + nodePool + "\"," +
			"  \"desiredNodePoolAutoscaling\": {" +
			"   \"enabled\": \"false\"," +
			"  }" +
			" }" +
			"}"

		url := getGKEClusterUrl()
		glog.Infof("Using gke api url %s", url)
		putResult, err := doPut(url, updateRequest)
		if err != nil {
			return fmt.Errorf("Failed to put %s: %v", url, err)
		}
		glog.Infof("Config update result: %s", putResult)
	}

	for startTime := time.Now(); startTime.Add(gkeUpdateTimeout).After(time.Now()); time.Sleep(30 * time.Second) {
		if val, err := isAutoscalerEnabled(minCount); err == nil && !val {
			return nil
		}
	}
	return fmt.Errorf("autoscaler still enabled")
}

func addNodePool(name string, machineType string, numNodes int) {
	output, err := exec.Command("gcloud", "alpha", "container", "node-pools", "create", name, "--quiet",
		"--machine-type="+machineType,
		"--num-nodes="+strconv.Itoa(numNodes),
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	framework.ExpectNoError(err)
	glog.Infof("Creating node-pool %s: %s", name, output)
}

func deleteNodePool(name string) {
	glog.Infof("Deleting node pool %s", name)
	output, err := exec.Command("gcloud", "alpha", "container", "node-pools", "delete", name, "--quiet",
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	if err != nil {
		glog.Infof("Error: %v", err)
	}
	glog.Infof("Node-pool deletion output: %s", output)
}

func doPut(url, content string) (string, error) {
	req, err := http.NewRequest("PUT", url, bytes.NewBuffer([]byte(content)))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	strBody := string(body)
	return strBody, nil
}

func CreateNodeSelectorPods(f *framework.Framework, id string, replicas int, nodeSelector map[string]string, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves host port and defines node selector"))

	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": 4321},
		NodeSelector:   nodeSelector,
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}

func CreateHostPortPods(f *framework.Framework, id string, replicas int, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves host port"))
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": 4321},
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}

func ReserveCpu(f *framework.Framework, id string, replicas, millicores int) {
	By(fmt.Sprintf("Running RC which reserves %v millicores", millicores))
	request := int64(millicores / replicas)
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		CpuRequest:     request,
	}
	framework.ExpectNoError(framework.RunRC(*config))
}

func ReserveMemory(f *framework.Framework, id string, replicas, megabytes int, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		MemRequest:     request,
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}

// WaitForClusterSize waits until the cluster size matches the given function.
func WaitForClusterSizeFunc(c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.Core().Nodes().List(v1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			glog.Warningf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		framework.FilterNodes(nodes, func(node v1.Node) bool {
			return framework.IsNodeConditionSetAsExpected(&node, v1.NodeReady, true)
		})
		numReady := len(nodes.Items)

		if numNodes == numReady && sizeFunc(numReady) {
			glog.Infof("Cluster has reached the desired size")
			return nil
		}
		glog.Infof("Waiting for cluster, current size %d, not ready nodes %d", numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for appropriate cluster size", timeout)
}

func waitForAllCaPodsReadyInNamespace(f *framework.Framework, c clientset.Interface) error {
	var notready []string
	for start := time.Now(); time.Now().Before(start.Add(scaleUpTimeout)); time.Sleep(20 * time.Second) {
		pods, err := c.Core().Pods(f.Namespace.Name).List(v1.ListOptions{})
		if err != nil {
			return fmt.Errorf("failed to get pods: %v", err)
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
				glog.Warningf("Pod has failed: %v", pod)
			}
			if !ready && pod.Status.Phase != v1.PodFailed {
				notready = append(notready, pod.Name)
			}
		}
		if len(notready) == 0 {
			glog.Infof("All pods ready")
			return nil
		}
		glog.Infof("Some pods are not ready yet: %v", notready)
	}
	glog.Info("Timeout on waiting for pods being ready")
	glog.Info(framework.RunKubectlOrDie("get", "pods", "-o", "json", "--all-namespaces"))
	glog.Info(framework.RunKubectlOrDie("get", "nodes", "-o", "json"))

	// Some pods are still not running.
	return fmt.Errorf("Some pods are still not running: %v", notready)
}

func setMigSizes(sizes map[string]int) {
	for mig, desiredSize := range sizes {
		currentSize, err := GroupSize(mig)
		framework.ExpectNoError(err)
		if desiredSize != currentSize {
			By(fmt.Sprintf("Setting size of %s to %d", mig, desiredSize))
			err = ResizeGroup(mig, int32(desiredSize))
			framework.ExpectNoError(err)
		}
	}
}
