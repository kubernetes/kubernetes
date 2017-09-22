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
	"bytes"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/scheduling"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	defaultTimeout         = 3 * time.Minute
	resizeTimeout          = 5 * time.Minute
	scaleUpTimeout         = 5 * time.Minute
	scaleUpTriggerTimeout  = 2 * time.Minute
	scaleDownTimeout       = 20 * time.Minute
	podTimeout             = 2 * time.Minute
	nodesRecoverTimeout    = 5 * time.Minute
	rcCreationRetryTimeout = 4 * time.Minute
	rcCreationRetryDelay   = 20 * time.Second
	makeSchedulableTimeout = 10 * time.Minute
	makeSchedulableDelay   = 20 * time.Second
	freshStatusLimit       = 20 * time.Second

	gkeEndpoint      = "https://test-container.sandbox.googleapis.com"
	gkeUpdateTimeout = 15 * time.Minute

	disabledTaint             = "DisabledForAutoscalingTest"
	criticalAddonsOnlyTaint   = "CriticalAddonsOnly"
	newNodesForScaledownTests = 2
	unhealthyClusterThreshold = 4

	caNoScaleUpStatus      = "NoActivity"
	caOngoingScaleUpStatus = "InProgress"
	timestampFormat        = "2006-01-02 15:04:05 -0700 MST"
)

var _ = SIGDescribe("Cluster size autoscaling [Slow]", func() {
	f := framework.NewDefaultFramework("autoscaling")
	var c clientset.Interface
	var nodeCount int
	var memAllocatableMb int
	var originalSizes map[string]int

	BeforeEach(func() {
		c = f.ClientSet
		framework.SkipUnlessProviderIs("gce", "gke")

		originalSizes = make(map[string]int)
		sum := 0
		for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			size, err := framework.GroupSize(mig)
			framework.ExpectNoError(err)
			By(fmt.Sprintf("Initial size of %s: %d", mig, size))
			originalSizes[mig] = size
			sum += size
		}
		// Give instances time to spin up
		framework.ExpectNoError(framework.WaitForReadyNodes(c, sum, scaleUpTimeout))

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		nodeCount = len(nodes.Items)
		By(fmt.Sprintf("Initial number of schedulable nodes: %v", nodeCount))
		Expect(nodeCount).NotTo(BeZero())
		mem := nodes.Items[0].Status.Allocatable[v1.ResourceMemory]
		memAllocatableMb = int((&mem).Value() / 1024 / 1024)

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
		expectedNodes := 0
		for _, size := range originalSizes {
			expectedNodes += size
		}
		framework.ExpectNoError(framework.WaitForReadyNodes(c, expectedNodes, scaleDownTimeout))
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

	It("shouldn't increase cluster size if pending pod is too large [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		By("Creating unschedulable pod")
		ReserveMemory(f, "memory-reservation", 1, int(1.1*float64(memAllocatableMb)), false, defaultTimeout)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		By("Waiting for scale up hoping it won't happen")
		// Verfiy, that the appropreate event was generated.
		eventFound := false
	EventsLoop:
		for start := time.Now(); time.Since(start) < scaleUpTimeout; time.Sleep(20 * time.Second) {
			By("Waiting for NotTriggerScaleUp event")
			events, err := f.ClientSet.Core().Events(f.Namespace.Name).List(metav1.ListOptions{})
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
		// Verify that cluster size is not changed
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size <= nodeCount }, time.Second))
	})

	simpleScaleUpTest := func(unready int) {
		ReserveMemory(f, "memory-reservation", 100, nodeCount*memAllocatableMb, false, 1*time.Second)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Verify that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout, unready))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	}

	It("should increase cluster size if pending pods are small [Feature:ClusterSizeAutoscalingScaleUp]",
		func() { simpleScaleUpTest(0) })

	It("should increase cluster size if pending pods are small and one node is broken [Feature:ClusterSizeAutoscalingScaleUp]",
		func() {
			framework.TestUnderTemporaryNetworkFailure(c, "default", getAnyNode(c), func() { simpleScaleUpTest(1) })
		})

	It("shouldn't trigger additional scale-ups during processing scale-up [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		// Wait for the situation to stabilize - CA should be running and have up-to-date node readiness info.
		status, err := waitForScaleUpStatus(c, func(s *scaleUpStatus) bool {
			return s.ready == s.target && s.ready <= nodeCount
		}, scaleUpTriggerTimeout)
		framework.ExpectNoError(err)

		unmanagedNodes := nodeCount - status.ready

		By("Schedule more pods than can fit and wait for cluster to scale-up")
		ReserveMemory(f, "memory-reservation", 100, nodeCount*memAllocatableMb, false, 1*time.Second)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		status, err = waitForScaleUpStatus(c, func(s *scaleUpStatus) bool {
			return s.status == caOngoingScaleUpStatus
		}, scaleUpTriggerTimeout)
		framework.ExpectNoError(err)
		target := status.target
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		By("Expect no more scale-up to be happening after all pods are scheduled")
		status, err = getScaleUpStatus(c)
		framework.ExpectNoError(err)
		if status.target != target {
			glog.Warningf("Final number of nodes (%v) does not match initial scale-up target (%v).", status.target, target)
		}
		Expect(status.timestamp.Add(freshStatusLimit).Before(time.Now())).Should(Equal(false))
		Expect(status.status).Should(Equal(caNoScaleUpStatus))
		Expect(status.ready).Should(Equal(status.target))
		Expect(len(framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items)).Should(Equal(status.target + unmanagedNodes))
	})

	It("should increase cluster size if pending pods are small and there is another node pool that is not autoscaled [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Creating new node-pool with one n1-standard-4 machine")
		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-4", 1)
		defer deleteNodePool(extraPoolName)
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+1, resizeTimeout))
		glog.Infof("Not enabling cluster autoscaler for the node pool (on purpose).")

		By("Get memory available on new node, so we can account for it when creating RC")
		nodes := getPoolNodes(f, extraPoolName)
		Expect(len(nodes)).Should(Equal(1))
		extraMem := nodes[0].Status.Capacity[v1.ResourceMemory]
		extraMemMb := int((&extraMem).Value() / 1024 / 1024)

		ReserveMemory(f, "memory-reservation", 100, nodeCount*memAllocatableMb+extraMemMb, false, defaultTimeout)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Verify, that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+2 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("should disable node pool autoscaling [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gke")

		By("Creating new node-pool with one n1-standard-4 machine")
		const extraPoolName = "extra-pool"
		addNodePool(extraPoolName, "n1-standard-4", 1)
		defer deleteNodePool(extraPoolName)
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+1, resizeTimeout))
		framework.ExpectNoError(enableAutoscaler(extraPoolName, 1, 2))
		framework.ExpectNoError(disableAutoscaler(extraPoolName, 1, 2))
	})

	It("should increase cluster size if pods are pending due to host port conflict [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		scheduling.CreateHostPortPods(f, "host-port", nodeCount+2, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "host-port")

		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+2 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("should increase cluster size if pods are pending due to pod anti-affinity [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		pods := nodeCount
		newPods := 2
		labels := map[string]string{
			"anti-affinity": "yes",
		}
		By("starting a pod with anti-affinity on each node")
		framework.ExpectNoError(runAntiAffinityPods(f, f.Namespace.Name, pods, "some-pod", labels, labels))
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "some-pod")
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		By("scheduling extra pods with anti-affinity to existing ones")
		framework.ExpectNoError(runAntiAffinityPods(f, f.Namespace.Name, newPods, "extra-pod", labels, labels))
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "extra-pod")

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+newPods, scaleUpTimeout))
	})

	It("should increase cluster size if pod requesting EmptyDir volume is pending [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		By("creating pods")
		pods := nodeCount
		newPods := 1
		labels := map[string]string{
			"anti-affinity": "yes",
		}
		framework.ExpectNoError(runAntiAffinityPods(f, f.Namespace.Name, pods, "some-pod", labels, labels))
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "some-pod")

		By("waiting for all pods before triggering scale up")
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		By("creating a pod requesting EmptyDir")
		framework.ExpectNoError(runVolumeAntiAffinityPods(f, f.Namespace.Name, newPods, "extra-pod", labels, labels, emptyDirVolumes))
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "extra-pod")

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+newPods, scaleUpTimeout))
	})

	It("should increase cluster size if pod requesting volume is pending [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessProviderIs("gce", "gke")

		volumeLabels := labels.Set{
			framework.VolumeSelectorKey: f.Namespace.Name,
		}
		selector := metav1.SetAsLabelSelector(volumeLabels)

		By("creating volume & pvc")
		diskName, err := framework.CreatePDWithRetry()
		framework.ExpectNoError(err)
		pvConfig := framework.PersistentVolumeConfig{
			NamePrefix: "gce-",
			Labels:     volumeLabels,
			PVSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   diskName,
					FSType:   "ext3",
					ReadOnly: false,
				},
			},
			Prebind: nil,
		}
		pvcConfig := framework.PersistentVolumeClaimConfig{
			Annotations: map[string]string{
				v1.BetaStorageClassAnnotation: "",
			},
			Selector: selector,
		}

		pv, pvc, err := framework.CreatePVPVC(c, pvConfig, pvcConfig, f.Namespace.Name, false)
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitOnPVandPVC(c, f.Namespace.Name, pv, pvc))

		defer func() {
			errs := framework.PVPVCCleanup(c, f.Namespace.Name, pv, pvc)
			if len(errs) > 0 {
				framework.Failf("failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pv, pvc = nil, nil
			if diskName != "" {
				framework.ExpectNoError(framework.DeletePDWithRetry(diskName))
			}
		}()

		By("creating pods")
		pods := nodeCount
		labels := map[string]string{
			"anti-affinity": "yes",
		}
		framework.ExpectNoError(runAntiAffinityPods(f, f.Namespace.Name, pods, "some-pod", labels, labels))
		defer func() {
			framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "some-pod")
			glog.Infof("RC and pods not using volume deleted")
		}()

		By("waiting for all pods before triggering scale up")
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))

		By("creating a pod requesting PVC")
		pvcPodName := "pvc-pod"
		newPods := 1
		volumes := buildVolumes(pv, pvc)
		framework.ExpectNoError(runVolumeAntiAffinityPods(f, f.Namespace.Name, newPods, pvcPodName, labels, labels, volumes))
		defer func() {
			framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, pvcPodName)
			framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
		}()

		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+newPods, scaleUpTimeout))
	})

	It("should add node to the particular mig [Feature:ClusterSizeAutoscalingScaleUp]", func() {
		labelKey := "cluster-autoscaling-test.special-node"
		labelValue := "true"

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
			for node := range nodesToClean {
				framework.RemoveLabelOffNode(c, node, labelKey)
			}
		}

		nodes, err := framework.GetGroupNodes(minMig)
		framework.ExpectNoError(err)
		nodesSet := sets.NewString(nodes...)
		defer removeLabels(nodesSet)
		By(fmt.Sprintf("Annotating nodes of the smallest MIG(%s): %v", minMig, nodes))

		for node := range nodesSet {
			framework.AddOrUpdateLabelOnNode(c, node, labelKey, labelValue)
		}

		scheduling.CreateNodeSelectorPods(f, "node-selector", minSize+1, map[string]string{labelKey: labelValue}, false)

		By("Waiting for new node to appear and annotating it")
		framework.WaitForGroupSize(minMig, int32(minSize+1))
		// Verify, that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= nodeCount+1 }, scaleUpTimeout))

		newNodes, err := framework.GetGroupNodes(minMig)
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
				if output, err := execCmd("gcloud", "compute", "instances", "describe",
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
		for node := range registeredNodes {
			framework.AddOrUpdateLabelOnNode(c, node, labelKey, labelValue)
		}

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
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+1, resizeTimeout))
		framework.ExpectNoError(enableAutoscaler(extraPoolName, 1, 2))

		By("Creating rc with 2 pods too big to fit default-pool but fitting extra-pool")
		ReserveMemory(f, "memory-reservation", 2, int(2.5*float64(memAllocatableMb)), false, defaultTimeout)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Apparently GKE master is restarted couple minutes after the node pool is added
		// reseting all the timers in scale down code. Adding 5 extra minutes to workaround
		// this issue.
		// TODO: Remove the extra time when GKE restart is fixed.
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount+2, scaleUpTimeout+5*time.Minute))
	})

	simpleScaleDownTest := func(unready int) {
		cleanup, err := addKubeSystemPdbs(f)
		defer cleanup()
		framework.ExpectNoError(err)

		By("Manually increase cluster size")
		increasedSize := 0
		newSizes := make(map[string]int)
		for key, val := range originalSizes {
			newSizes[key] = val + 2 + unready
			increasedSize += val + 2 + unready
		}
		setMigSizes(newSizes)
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(f.ClientSet,
			func(size int) bool { return size >= increasedSize }, scaleUpTimeout, unready))

		By("Some node should be removed")
		framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(f.ClientSet,
			func(size int) bool { return size < increasedSize }, scaleDownTimeout, unready))
	}

	It("should correctly scale down after a node is not needed [Feature:ClusterSizeAutoscalingScaleDown]",
		func() { simpleScaleDownTest(0) })

	It("should correctly scale down after a node is not needed and one node is broken [Feature:ClusterSizeAutoscalingScaleDown]",
		func() {
			framework.TestUnderTemporaryNetworkFailure(c, "default", getAnyNode(c), func() { simpleScaleDownTest(1) })
		})

	It("should correctly scale down after a node is not needed when there is non autoscaled pool[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		framework.SkipUnlessProviderIs("gke")

		increasedSize := manuallyIncreaseClusterSize(f, originalSizes)

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

	It("should be able to scale down when rescheduling a pod is required and pdb allows for it[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		runDrainTest(f, originalSizes, f.Namespace.Name, 1, 1, func(increasedSize int) {
			By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	It("shouldn't be able to scale down when rescheduling a pod is required, but pdb doesn't allow drain[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		runDrainTest(f, originalSizes, f.Namespace.Name, 1, 0, func(increasedSize int) {
			By("No nodes should be removed")
			time.Sleep(scaleDownTimeout)
			nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			Expect(len(nodes.Items)).Should(Equal(increasedSize))
		})
	})

	It("should be able to scale down by draining multiple pods one by one as dictated by pdb[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		runDrainTest(f, originalSizes, f.Namespace.Name, 2, 1, func(increasedSize int) {
			By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	It("should be able to scale down by draining system pods with pdb[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		runDrainTest(f, originalSizes, "kube-system", 2, 1, func(increasedSize int) {
			By("Some node should be removed")
			framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
				func(size int) bool { return size < increasedSize }, scaleDownTimeout))
		})
	})

	It("Should be able to scale a node group up from 0[Feature:ClusterSizeAutoscalingScaleUp]", func() {
		framework.SkipUnlessAtLeast(len(originalSizes), 2, "At least 2 node groups are needed for scale-to-0 tests")
		By("Manually scale smallest node group to 0")
		minMig := ""
		minSize := nodeCount
		for mig, size := range originalSizes {
			if size <= minSize {
				minMig = mig
				minSize = size
			}
		}
		err := framework.ResizeGroup(minMig, int32(0))
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount-minSize, resizeTimeout))

		By("Make remaining nodes unschedulable")
		nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		framework.ExpectNoError(err)

		for _, node := range nodes.Items {
			err = makeNodeUnschedulable(f.ClientSet, &node)

			defer func(n v1.Node) {
				makeNodeSchedulable(f.ClientSet, &n, false)
			}(node)
			framework.ExpectNoError(err)
		}

		By("Run a scale-up test")
		ReserveMemory(f, "memory-reservation", 1, 100, false, 1*time.Second)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")

		// Verify that cluster size is increased
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size >= len(nodes.Items)+1 }, scaleUpTimeout))
		framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(f, c))
	})

	It("Should be able to scale a node group down to 0[Feature:ClusterSizeAutoscalingScaleDown]", func() {
		framework.SkipUnlessAtLeast(len(originalSizes), 2, "At least 2 node groups are needed for scale-to-0 tests")
		By("Find smallest node group and manually scale it to a single node")
		minMig := ""
		minSize := nodeCount
		for mig, size := range originalSizes {
			if size <= minSize {
				minMig = mig
				minSize = size
			}
		}
		err := framework.ResizeGroup(minMig, int32(1))
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForReadyNodes(c, nodeCount-minSize+1, resizeTimeout))

		By("Make the single node unschedulable")
		allNodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		framework.ExpectNoError(err)
		ngNodes, err := framework.GetGroupNodes(minMig)
		framework.ExpectNoError(err)
		By(fmt.Sprintf("Target nodes for scale-down: %s", ngNodes))
		Expect(len(ngNodes) == 1).To(BeTrue())
		node, err := f.ClientSet.Core().Nodes().Get(ngNodes[0], metav1.GetOptions{})
		framework.ExpectNoError(err)
		makeNodeUnschedulable(f.ClientSet, node)

		By("Manually drain the single node")
		podOpts := metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
		pods, err := c.Core().Pods(metav1.NamespaceAll).List(podOpts)
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			err = f.ClientSet.Core().Pods(pod.Namespace).Delete(pod.Name, metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)
		}

		By("The node should be removed")
		framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet,
			func(size int) bool { return size < len(allNodes.Items) }, scaleDownTimeout))
		minSize, err = framework.GroupSize(minMig)
		framework.ExpectNoError(err)
		Expect(minSize).Should(Equal(0))
	})

	It("Shouldn't perform scale up operation and should list unhealthy status if most of the cluster is broken[Feature:ClusterSizeAutoscalingScaleUp]", func() {
		clusterSize := nodeCount
		for clusterSize < unhealthyClusterThreshold+1 {
			clusterSize = manuallyIncreaseClusterSize(f, originalSizes)
		}

		By("Block network connectivity to some nodes to simulate unhealthy cluster")
		nodesToBreakCount := int(math.Floor(math.Max(float64(unhealthyClusterThreshold), 0.5*float64(clusterSize))))
		nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		framework.ExpectNoError(err)
		Expect(nodesToBreakCount <= len(nodes.Items)).To(BeTrue())
		nodesToBreak := nodes.Items[:nodesToBreakCount]

		// TestUnderTemporaryNetworkFailure only removes connectivity to a single node,
		// and accepts func() callback. This is expanding the loop to recursive call
		// to avoid duplicating TestUnderTemporaryNetworkFailure
		var testFunction func()
		testFunction = func() {
			if len(nodesToBreak) > 0 {
				ntb := &nodesToBreak[0]
				nodesToBreak = nodesToBreak[1:]
				framework.TestUnderTemporaryNetworkFailure(c, "default", ntb, testFunction)
			} else {
				ReserveMemory(f, "memory-reservation", 100, nodeCount*memAllocatableMb, false, defaultTimeout)
				defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, "memory-reservation")
				time.Sleep(scaleUpTimeout)
				currentNodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
				framework.Logf("Currently available nodes: %v, nodes available at the start of test: %v, disabled nodes: %v", len(currentNodes.Items), len(nodes.Items), nodesToBreakCount)
				Expect(len(currentNodes.Items)).Should(Equal(len(nodes.Items) - nodesToBreakCount))
				status, err := getClusterwideStatus(c)
				framework.Logf("Clusterwide status: %v", status)
				framework.ExpectNoError(err)
				Expect(status).Should(Equal("Unhealthy"))
			}
		}
		testFunction()
		// Give nodes time to recover from network failure
		framework.ExpectNoError(framework.WaitForReadyNodes(c, len(nodes.Items), nodesRecoverTimeout))
	})

})

func execCmd(args ...string) *exec.Cmd {
	glog.Infof("Executing: %s", strings.Join(args, " "))
	return exec.Command(args[0], args[1:]...)
}

func runDrainTest(f *framework.Framework, migSizes map[string]int, namespace string, podsPerNode, pdbSize int, verifyFunction func(int)) {
	increasedSize := manuallyIncreaseClusterSize(f, migSizes)

	nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
		"spec.unschedulable": "false",
	}.AsSelector().String()})
	framework.ExpectNoError(err)
	numPods := len(nodes.Items) * podsPerNode
	testId := string(uuid.NewUUID()) // So that we can label and find pods
	labelMap := map[string]string{"test_id": testId}
	framework.ExpectNoError(runReplicatedPodOnEachNode(f, nodes.Items, namespace, podsPerNode, "reschedulable-pods", labelMap, 0))

	defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, namespace, "reschedulable-pods")

	By("Create a PodDisruptionBudget")
	minAvailable := intstr.FromInt(numPods - pdbSize)
	pdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test_pdb",
			Namespace: namespace,
		},
		Spec: policy.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: labelMap},
			MinAvailable: &minAvailable,
		},
	}
	_, err = f.ClientSet.Policy().PodDisruptionBudgets(namespace).Create(pdb)

	defer func() {
		f.ClientSet.Policy().PodDisruptionBudgets(namespace).Delete(pdb.Name, &metav1.DeleteOptions{})
	}()

	framework.ExpectNoError(err)
	verifyFunction(increasedSize)
}

func getGKEClusterUrl() string {
	out, err := execCmd("gcloud", "auth", "print-access-token").Output()
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
	if strings.Contains(strBody, "\"minNodeCount\": "+strconv.Itoa(expectedMinNodeCountInTargetPool)) {
		return true, nil
	}
	return false, nil
}

func enableAutoscaler(nodePool string, minCount, maxCount int) error {

	if nodePool == "default-pool" {
		glog.Infof("Using gcloud to enable autoscaling for pool %s", nodePool)

		output, err := execCmd("gcloud", "alpha", "container", "clusters", "update", framework.TestContext.CloudConfig.Cluster,
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

		output, err := execCmd("gcloud", "alpha", "container", "clusters", "update", framework.TestContext.CloudConfig.Cluster,
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
	output, err := execCmd("gcloud", "alpha", "container", "node-pools", "create", name, "--quiet",
		"--machine-type="+machineType,
		"--num-nodes="+strconv.Itoa(numNodes),
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	glog.Infof("Creating node-pool %s: %s", name, output)
	framework.ExpectNoError(err)
}

func deleteNodePool(name string) {
	glog.Infof("Deleting node pool %s", name)
	output, err := execCmd("gcloud", "alpha", "container", "node-pools", "delete", name, "--quiet",
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	if err != nil {
		glog.Infof("Error: %v", err)
	}
	glog.Infof("Node-pool deletion output: %s", output)
}

func getPoolNodes(f *framework.Framework, poolName string) []*v1.Node {
	nodes := make([]*v1.Node, 0, 1)
	nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	for _, node := range nodeList.Items {
		if poolLabel := node.Labels["cloud.google.com/gke-nodepool"]; poolLabel == poolName {
			nodes = append(nodes, &node)
		}
	}
	return nodes
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

func ReserveMemory(f *framework.Framework, id string, replicas, megabytes int, expectRunning bool, timeout time.Duration) func() error {
	By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        timeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		MemRequest:     request,
	}
	for start := time.Now(); time.Since(start) < rcCreationRetryTimeout; time.Sleep(rcCreationRetryDelay) {
		err := framework.RunRC(*config)
		if err != nil && strings.Contains(err.Error(), "Error creating replication controller") {
			glog.Warningf("Failed to create memory reservation: %v", err)
			continue
		}
		if expectRunning {
			framework.ExpectNoError(err)
		}
		return func() error {
			return framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, id)
		}
	}
	framework.Failf("Failed to reserve memory within timeout")
	return nil
}

// WaitForClusterSizeFunc waits until the cluster size matches the given function.
func WaitForClusterSizeFunc(c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration) error {
	return WaitForClusterSizeFuncWithUnready(c, sizeFunc, timeout, 0)
}

// WaitForClusterSizeWithUnready waits until the cluster size matches the given function and assumes some unready nodes.
func WaitForClusterSizeFuncWithUnready(c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration, expectedUnready int) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
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

		if numNodes == numReady+expectedUnready && sizeFunc(numNodes) {
			glog.Infof("Cluster has reached the desired size")
			return nil
		}
		glog.Infof("Waiting for cluster with func, current size %d, not ready nodes %d", numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for appropriate cluster size", timeout)
}

func waitForCaPodsReadyInNamespace(f *framework.Framework, c clientset.Interface, tolerateUnreadyCount int) error {
	var notready []string
	for start := time.Now(); time.Now().Before(start.Add(scaleUpTimeout)); time.Sleep(20 * time.Second) {
		pods, err := c.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
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
		if len(notready) <= tolerateUnreadyCount {
			glog.Infof("sufficient number of pods ready. Tolerating %d unready", tolerateUnreadyCount)
			return nil
		}
		glog.Infof("Too many pods are not ready yet: %v", notready)
	}
	glog.Info("Timeout on waiting for pods being ready")
	glog.Info(framework.RunKubectlOrDie("get", "pods", "-o", "json", "--all-namespaces"))
	glog.Info(framework.RunKubectlOrDie("get", "nodes", "-o", "json"))

	// Some pods are still not running.
	return fmt.Errorf("Too many pods are still not running: %v", notready)
}

func waitForAllCaPodsReadyInNamespace(f *framework.Framework, c clientset.Interface) error {
	return waitForCaPodsReadyInNamespace(f, c, 0)
}

func getAnyNode(c clientset.Interface) *v1.Node {
	nodes, err := c.Core().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
		"spec.unschedulable": "false",
	}.AsSelector().String()})
	if err != nil {
		glog.Errorf("Failed to get node list: %v", err)
		return nil
	}
	if len(nodes.Items) == 0 {
		glog.Errorf("No nodes")
		return nil
	}
	return &nodes.Items[0]
}

func setMigSizes(sizes map[string]int) bool {
	madeChanges := false
	for mig, desiredSize := range sizes {
		currentSize, err := framework.GroupSize(mig)
		framework.ExpectNoError(err)
		if desiredSize != currentSize {
			By(fmt.Sprintf("Setting size of %s to %d", mig, desiredSize))
			err = framework.ResizeGroup(mig, int32(desiredSize))
			framework.ExpectNoError(err)
			madeChanges = true
		}
	}
	return madeChanges
}

func makeNodeUnschedulable(c clientset.Interface, node *v1.Node) error {
	By(fmt.Sprintf("Taint node %s", node.Name))
	for j := 0; j < 3; j++ {
		freshNode, err := c.Core().Nodes().Get(node.Name, metav1.GetOptions{})
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
		_, err = c.Core().Nodes().Update(freshNode)
		if err == nil {
			return nil
		}
		if !errors.IsConflict(err) {
			return err
		}
		glog.Warningf("Got 409 conflict when trying to taint node, retries left: %v", 3-j)
	}
	return fmt.Errorf("Failed to taint node in allowed number of retries")
}

type CriticalAddonsOnlyError struct{}

func (_ CriticalAddonsOnlyError) Error() string {
	return fmt.Sprintf("CriticalAddonsOnly taint found on node")
}

func makeNodeSchedulable(c clientset.Interface, node *v1.Node, failOnCriticalAddonsOnly bool) error {
	By(fmt.Sprintf("Remove taint from node %s", node.Name))
	for j := 0; j < 3; j++ {
		freshNode, err := c.Core().Nodes().Get(node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		newTaints := make([]v1.Taint, 0)
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
		_, err = c.Core().Nodes().Update(freshNode)
		if err == nil {
			return nil
		}
		if !errors.IsConflict(err) {
			return err
		}
		glog.Warningf("Got 409 conflict when trying to taint node, retries left: %v", 3-j)
	}
	return fmt.Errorf("Failed to remove taint from node in allowed number of retries")
}

// Create an RC running a given number of pods with anti-affinity
func runAntiAffinityPods(f *framework.Framework, namespace string, pods int, id string, podLabels, antiAffinityLabels map[string]string) error {
	config := &testutils.RCConfig{
		Affinity:       buildAntiAffinity(antiAffinityLabels),
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      namespace,
		Timeout:        scaleUpTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       pods,
		Labels:         podLabels,
	}
	err := framework.RunRC(*config)
	if err != nil {
		return err
	}
	_, err = f.ClientSet.Core().ReplicationControllers(namespace).Get(id, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func runVolumeAntiAffinityPods(f *framework.Framework, namespace string, pods int, id string, podLabels, antiAffinityLabels map[string]string, volumes []v1.Volume) error {
	config := &testutils.RCConfig{
		Affinity:       buildAntiAffinity(antiAffinityLabels),
		Volumes:        volumes,
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      namespace,
		Timeout:        scaleUpTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       pods,
		Labels:         podLabels,
	}
	err := framework.RunRC(*config)
	if err != nil {
		return err
	}
	_, err = f.ClientSet.Core().ReplicationControllers(namespace).Get(id, metav1.GetOptions{})
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

func buildVolumes(pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) []v1.Volume {
	return []v1.Volume{
		{
			Name: pv.Name,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvc.Name,
					ReadOnly:  false,
				},
			},
		},
	}
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
func runReplicatedPodOnEachNode(f *framework.Framework, nodes []v1.Node, namespace string, podsPerNode int, id string, labels map[string]string, memRequest int64) error {
	By("Run a pod on each node")
	for _, node := range nodes {
		err := makeNodeUnschedulable(f.ClientSet, &node)

		defer func(n v1.Node) {
			makeNodeSchedulable(f.ClientSet, &n, false)
		}(node)

		if err != nil {
			return err
		}
	}
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      namespace,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       0,
		Labels:         labels,
		MemRequest:     memRequest,
	}
	err := framework.RunRC(*config)
	if err != nil {
		return err
	}
	rc, err := f.ClientSet.Core().ReplicationControllers(namespace).Get(id, metav1.GetOptions{})
	if err != nil {
		return err
	}
	for i, node := range nodes {
		err = makeNodeSchedulable(f.ClientSet, &node, false)
		if err != nil {
			return err
		}

		// Update replicas count, to create new pods that will be allocated on node
		// (we retry 409 errors in case rc reference got out of sync)
		for j := 0; j < 3; j++ {
			*rc.Spec.Replicas = int32((i + 1) * podsPerNode)
			rc, err = f.ClientSet.Core().ReplicationControllers(namespace).Update(rc)
			if err == nil {
				break
			}
			if !errors.IsConflict(err) {
				return err
			}
			glog.Warningf("Got 409 conflict when trying to scale RC, retries left: %v", 3-j)
			rc, err = f.ClientSet.Core().ReplicationControllers(namespace).Get(id, metav1.GetOptions{})
			if err != nil {
				return err
			}
		}

		err = wait.PollImmediate(5*time.Second, podTimeout, func() (bool, error) {
			rc, err = f.ClientSet.Core().ReplicationControllers(namespace).Get(id, metav1.GetOptions{})
			if err != nil || rc.Status.ReadyReplicas < int32((i+1)*podsPerNode) {
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			return fmt.Errorf("failed to coerce RC into spawning a pod on node %s within timeout", node.Name)
		}
		err = makeNodeUnschedulable(f.ClientSet, &node)
		if err != nil {
			return err
		}
	}
	return nil
}

// wrap runReplicatedPodOnEachNode to return cleanup
func runReplicatedPodOnEachNodeWithCleanup(f *framework.Framework, nodes []v1.Node, namespace string, podsPerNode int, id string, labels map[string]string, memRequest int64) (func(), error) {
	err := runReplicatedPodOnEachNode(f, nodes, namespace, podsPerNode, id, labels, memRequest)
	return func() {
		framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, namespace, id)
	}, err
}

// Increase cluster size by newNodesForScaledownTests to create some unused nodes
// that can be later removed by cluster autoscaler.
func manuallyIncreaseClusterSize(f *framework.Framework, originalSizes map[string]int) int {
	By("Manually increase cluster size")
	increasedSize := 0
	newSizes := make(map[string]int)
	for key, val := range originalSizes {
		newSizes[key] = val + newNodesForScaledownTests
		increasedSize += val + newNodesForScaledownTests
	}
	setMigSizes(newSizes)

	checkClusterSize := func(size int) bool {
		if size >= increasedSize {
			return true
		}
		resized := setMigSizes(newSizes)
		if resized {
			glog.Warning("Unexpected node group size while waiting for cluster resize. Setting size to target again.")
		}
		return false
	}

	framework.ExpectNoError(WaitForClusterSizeFunc(f.ClientSet, checkClusterSize, scaleUpTimeout))
	return increasedSize
}

// Try to get clusterwide health from CA status configmap.
// Status configmap is not parsing-friendly, so evil regexpery follows.
func getClusterwideStatus(c clientset.Interface) (string, error) {
	configMap, err := c.CoreV1().ConfigMaps("kube-system").Get("cluster-autoscaler-status", metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	status, ok := configMap.Data["status"]
	if !ok {
		return "", fmt.Errorf("Status information not found in configmap")
	}
	matcher, err := regexp.Compile("Cluster-wide:\\s*\n\\s*Health:\\s*([A-Za-z]+)")
	if err != nil {
		return "", err
	}
	result := matcher.FindStringSubmatch(status)
	if len(result) < 2 {
		return "", fmt.Errorf("Failed to parse CA status configmap, raw status: %v", status)
	}
	return result[1], nil
}

type scaleUpStatus struct {
	status    string
	ready     int
	target    int
	timestamp time.Time
}

// Try to get timestamp from status.
// Status configmap is not parsing-friendly, so evil regexpery follows.
func getStatusTimestamp(status string) (time.Time, error) {
	timestampMatcher, err := regexp.Compile("Cluster-autoscaler status at \\s*([0-9\\-]+ [0-9]+:[0-9]+:[0-9]+\\.[0-9]+ \\+[0-9]+ [A-Za-z]+):")
	if err != nil {
		return time.Time{}, err
	}

	timestampMatches := timestampMatcher.FindAllStringSubmatch(status, -1)
	if len(timestampMatches) < 1 {
		return time.Time{}, fmt.Errorf("Failed to parse CA status timestamp, raw status: %v", status)
	}

	timestamp, err := time.Parse(timestampFormat, timestampMatches[0][1])
	if err != nil {
		return time.Time{}, err
	}
	return timestamp, nil
}

// Try to get scaleup statuses of all node groups.
// Status configmap is not parsing-friendly, so evil regexpery follows.
func getScaleUpStatus(c clientset.Interface) (*scaleUpStatus, error) {
	configMap, err := c.CoreV1().ConfigMaps("kube-system").Get("cluster-autoscaler-status", metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	status, ok := configMap.Data["status"]
	if !ok {
		return nil, fmt.Errorf("Status information not found in configmap")
	}

	timestamp, err := getStatusTimestamp(status)
	if err != nil {
		return nil, err
	}

	matcher, err := regexp.Compile("s*ScaleUp:\\s*([A-Za-z]+)\\s*\\(ready=([0-9]+)\\s*cloudProviderTarget=([0-9]+)\\s*\\)")
	if err != nil {
		return nil, err
	}
	matches := matcher.FindAllStringSubmatch(status, -1)
	if len(matches) < 1 {
		return nil, fmt.Errorf("Failed to parse CA status configmap, raw status: %v", status)
	}

	result := scaleUpStatus{
		status:    caNoScaleUpStatus,
		ready:     0,
		target:    0,
		timestamp: timestamp,
	}
	for _, match := range matches {
		if match[1] == caOngoingScaleUpStatus {
			result.status = caOngoingScaleUpStatus
		}
		newReady, err := strconv.Atoi(match[2])
		if err != nil {
			return nil, err
		}
		result.ready += newReady
		newTarget, err := strconv.Atoi(match[3])
		if err != nil {
			return nil, err
		}
		result.target += newTarget
	}
	glog.Infof("Cluster-Autoscaler scale-up status: %v (%v, %v)", result.status, result.ready, result.target)
	return &result, nil
}

func waitForScaleUpStatus(c clientset.Interface, cond func(s *scaleUpStatus) bool, timeout time.Duration) (*scaleUpStatus, error) {
	var finalErr error
	var status *scaleUpStatus
	err := wait.PollImmediate(5*time.Second, timeout, func() (bool, error) {
		status, finalErr = getScaleUpStatus(c)
		if finalErr != nil {
			return false, nil
		}
		if status.timestamp.Add(freshStatusLimit).Before(time.Now()) {
			// stale status
			finalErr = fmt.Errorf("Status too old")
			return false, nil
		}
		return cond(status), nil
	})
	if err != nil {
		err = fmt.Errorf("Failed to find expected scale up status: %v, last status: %v, final err: %v", err, status, finalErr)
	}
	return status, err
}

// This is a temporary fix to allow CA to migrate some kube-system pods
// TODO: Remove this when the PDB is added for some of those components
func addKubeSystemPdbs(f *framework.Framework) (func(), error) {
	By("Create PodDisruptionBudgets for kube-system components, so they can be migrated if required")

	newPdbs := make([]string, 0)
	cleanup := func() {
		var finalErr error
		for _, newPdbName := range newPdbs {
			By(fmt.Sprintf("Delete PodDisruptionBudget %v", newPdbName))
			err := f.ClientSet.Policy().PodDisruptionBudgets("kube-system").Delete(newPdbName, &metav1.DeleteOptions{})
			if err != nil {
				// log error, but attempt to remove other pdbs
				glog.Errorf("Failed to delete PodDisruptionBudget %v, err: %v", newPdbName, err)
				finalErr = err
			}
		}
		if finalErr != nil {
			framework.Failf("Error during PodDisruptionBudget cleanup: %v", finalErr)
		}
	}

	type pdbInfo struct {
		label         string
		min_available int
	}
	pdbsToAdd := []pdbInfo{
		{label: "kube-dns", min_available: 1},
		{label: "kube-dns-autoscaler", min_available: 0},
		{label: "metrics-server", min_available: 0},
		{label: "kubernetes-dashboard", min_available: 0},
		{label: "glbc", min_available: 0},
	}
	for _, pdbData := range pdbsToAdd {
		By(fmt.Sprintf("Create PodDisruptionBudget for %v", pdbData.label))
		labelMap := map[string]string{"k8s-app": pdbData.label}
		pdbName := fmt.Sprintf("test-pdb-for-%v", pdbData.label)
		minAvailable := intstr.FromInt(pdbData.min_available)
		pdb := &policy.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pdbName,
				Namespace: "kube-system",
			},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector:     &metav1.LabelSelector{MatchLabels: labelMap},
				MinAvailable: &minAvailable,
			},
		}
		_, err := f.ClientSet.Policy().PodDisruptionBudgets("kube-system").Create(pdb)
		newPdbs = append(newPdbs, pdbName)

		if err != nil {
			return cleanup, err
		}
	}
	return cleanup, nil
}
