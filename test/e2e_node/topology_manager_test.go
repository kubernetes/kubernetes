/*
Copyright 2019 The Kubernetes Authors.

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
	"io/ioutil"
	testutils "k8s.io/kubernetes/test/utils"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	numalignCmd = `export CPULIST_ALLOWED=$( awk -F":\t*" '/Cpus_allowed_list/ { print $2 }' /proc/self/status); env; sleep 1d`

	minNumaNodes     = 2
	minCoreCount     = 4
	minSriovResource = 7 // This is the min number of SRIOV VFs needed on the system under test.
)

// Helper for makeTopologyManagerPod().
type tmCtnAttribute struct {
	ctnName       string
	cpuRequest    string
	cpuLimit      string
	deviceName    string
	deviceRequest string
	deviceLimit   string
}

func detectNUMANodes() int {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"NUMA node(s):\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	numaNodes, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return numaNodes
}

func detectCoresPerSocket() int {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"Core(s) per socket:\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	coreCount, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return coreCount
}

func detectSRIOVDevices() int {
	outData, err := exec.Command("/bin/sh", "-c", "ls /sys/bus/pci/devices/*/physfn | wc -w").Output()
	framework.ExpectNoError(err)

	devCount, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return devCount
}

func makeTopologyManagerTestPod(podName, podCmd string, tmCtnAttributes []tmCtnAttribute) *v1.Pod {
	var containers []v1.Container
	for _, ctnAttr := range tmCtnAttributes {
		ctn := v1.Container{
			Name:  ctnAttr.ctnName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(ctnAttr.cpuRequest),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(ctnAttr.cpuLimit),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
				},
			},
			Command: []string{"sh", "-c", podCmd},
		}
		if ctnAttr.deviceName != "" {
			ctn.Resources.Requests[v1.ResourceName(ctnAttr.deviceName)] = resource.MustParse(ctnAttr.deviceRequest)
			ctn.Resources.Limits[v1.ResourceName(ctnAttr.deviceName)] = resource.MustParse(ctnAttr.deviceLimit)
		}
		containers = append(containers, ctn)
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers:    containers,
		},
	}
}

func findNUMANodeWithoutSRIOVDevicesFromConfigMap(configMap *v1.ConfigMap, numaNodes int) (int, bool) {
	for nodeNum := 0; nodeNum < numaNodes; nodeNum++ {
		value, ok := configMap.Annotations[fmt.Sprintf("pcidevice_node%d", nodeNum)]
		if !ok {
			framework.Logf("missing pcidevice annotation for NUMA node %d", nodeNum)
			return -1, false
		}
		v, err := strconv.Atoi(value)
		if err != nil {
			framework.Failf("error getting the PCI device count on NUMA node %d: %v", nodeNum, err)
		}
		if v == 0 {
			framework.Logf("NUMA node %d has no SRIOV devices attached", nodeNum)
			return nodeNum, true
		}
		framework.Logf("NUMA node %d has %d SRIOV devices attached", nodeNum, v)
	}
	return -1, false
}

func findNUMANodeWithoutSRIOVDevicesFromSysfs(numaNodes int) (int, bool) {
	pciDevs, err := getPCIDeviceInfo("/sys/bus/pci/devices")
	if err != nil {
		framework.Failf("error detecting the PCI device NUMA node: %v", err)
	}

	pciPerNuma := make(map[int]int)
	for _, pciDev := range pciDevs {
		if pciDev.IsVFn {
			pciPerNuma[pciDev.NUMANode]++
		}
	}

	if len(pciPerNuma) == 0 {
		framework.Logf("failed to find any VF device from %v", pciDevs)
		return -1, false
	}

	for nodeNum := 0; nodeNum < numaNodes; nodeNum++ {
		v := pciPerNuma[nodeNum]
		if v == 0 {
			framework.Logf("NUMA node %d has no SRIOV devices attached", nodeNum)
			return nodeNum, true
		}
		framework.Logf("NUMA node %d has %d SRIOV devices attached", nodeNum, v)
	}
	return -1, false
}

func findNUMANodeWithoutSRIOVDevices(configMap *v1.ConfigMap, numaNodes int) (int, bool) {
	// if someone annotated the configMap, let's use this information
	if nodeNum, found := findNUMANodeWithoutSRIOVDevicesFromConfigMap(configMap, numaNodes); found {
		return nodeNum, found
	}
	// no annotations, try to autodetect
	// NOTE: this assumes all the VFs in the box can be used for the tests.
	return findNUMANodeWithoutSRIOVDevicesFromSysfs(numaNodes)
}

func configureTopologyManagerInKubelet(f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration, policy string, configMap *v1.ConfigMap, numaNodes int) string {
	// Configure Topology Manager in Kubelet with policy.
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}

	newCfg.FeatureGates["CPUManager"] = true
	newCfg.FeatureGates["TopologyManager"] = true

	deleteStateFile()

	// Set the Topology Manager policy
	newCfg.TopologyManagerPolicy = policy

	// Set the CPU Manager policy to static.
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)

	// Set the CPU Manager reconcile period to 1 second.
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	if nodeNum, ok := findNUMANodeWithoutSRIOVDevices(configMap, numaNodes); ok {
		cpus, err := getCPUsPerNUMANode(nodeNum)
		framework.Logf("NUMA Node %d doesn't seem to have attached SRIOV devices and has cpus=%v", nodeNum, cpus)
		framework.ExpectNoError(err)
		newCfg.ReservedSystemCPUs = fmt.Sprintf("%d", cpus[len(cpus)-1])
	} else {
		// The Kubelet panics if either kube-reserved or system-reserved is not set
		// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
		// kubelet doesn't panic.
		if newCfg.KubeReserved == nil {
			newCfg.KubeReserved = map[string]string{}
		}

		if _, ok := newCfg.KubeReserved["cpu"]; !ok {
			newCfg.KubeReserved["cpu"] = "200m"
		}
	}
	// Dump the config -- debug
	framework.Logf("New kubelet config is %s", *newCfg)

	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return newCfg.ReservedSystemCPUs
}

// getSRIOVDevicePluginPod returns the Device Plugin pod for sriov resources in e2e tests.
func getSRIOVDevicePluginPod() *v1.Pod {
	ds := readDaemonSetV1OrDie(e2etestfiles.ReadOrDie(SRIOVDevicePluginDSYAML))
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      SRIOVDevicePluginName,
			Namespace: metav1.NamespaceSystem,
		},

		Spec: ds.Spec.Template.Spec,
	}

	return p
}

func readConfigMapV1OrDie(objBytes []byte) *v1.ConfigMap {
	v1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(v1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*v1.ConfigMap)
}

func readServiceAccountV1OrDie(objBytes []byte) *v1.ServiceAccount {
	v1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(v1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*v1.ServiceAccount)
}

func findSRIOVResource(node *v1.Node) (string, int64) {
	framework.Logf("Node status allocatable: %v", node.Status.Allocatable)
	re := regexp.MustCompile(`^intel.com/.*sriov.*`)
	for key, val := range node.Status.Allocatable {
		resource := string(key)
		if re.MatchString(resource) {
			v := val.Value()
			if v > 0 {
				return resource, v
			}
		}
	}
	return "", 0
}

func validatePodAlignment(f *framework.Framework, pod *v1.Pod, envInfo *testEnvInfo) {
	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.Logf("got pod logs: %v", logs)
		numaRes, err := checkNUMAAlignment(f, pod, &cnt, logs, envInfo)
		framework.ExpectNoError(err, "NUMA Alignment check failed for [%s] of pod [%s]", cnt.Name, pod.Name)
		if numaRes != nil {
			framework.Logf("NUMA resources for %s/%s: %s", pod.Name, cnt.Name, numaRes.String())
		}
	}
}

func runTopologyManagerPolicySuiteTests(f *framework.Framework) {
	var cpuCap, cpuAlloc int64

	cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(f)

	ginkgo.By("running a non-Gu pod")
	runNonGuPodTest(f, cpuCap)

	ginkgo.By("running a Gu pod")
	runGuPodTest(f)

	ginkgo.By("running multiple Gu and non-Gu pods")
	runMultipleGuNonGuPods(f, cpuCap, cpuAlloc)

	// Skip rest of the tests if CPU capacity < 3.
	if cpuCap < 3 {
		e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
	}

	ginkgo.By("running a Gu pod requesting multiple CPUs")
	runMultipleCPUGuPod(f)

	ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
	runMultipleCPUContainersGuPod(f)

	ginkgo.By("running multiple Gu pods")
	runMultipleGuPods(f)
}

func waitForAllContainerRemoval(podName, podNS string) {
	rs, _, err := getCRIClient()
	framework.ExpectNoError(err)
	gomega.Eventually(func() bool {
		containers, err := rs.ListContainers(&runtimeapi.ContainerFilter{
			LabelSelector: map[string]string{
				types.KubernetesPodNameLabel:      podName,
				types.KubernetesPodNamespaceLabel: podNS,
			},
		})
		if err != nil {
			return false
		}
		return len(containers) == 0
	}, 2*time.Minute, 1*time.Second).Should(gomega.BeTrue())
}

func runTopologyManagerPositiveTest(f *framework.Framework, numPods int, ctnAttrs []tmCtnAttribute, envInfo *testEnvInfo) {
	var pods []*v1.Pod

	for podID := 0; podID < numPods; podID++ {
		podName := fmt.Sprintf("gu-pod-%d", podID)
		framework.Logf("creating pod %s attrs %v", podName, ctnAttrs)
		pod := makeTopologyManagerTestPod(podName, numalignCmd, ctnAttrs)
		pod = f.PodClient().CreateSync(pod)
		framework.Logf("created pod %s", podName)
		pods = append(pods, pod)
	}

	// per https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/0035-20190130-topology-manager.md#multi-numa-systems-tests
	// we can do a menaingful validation only when using the single-numa node policy
	if envInfo.policy == topologymanager.PolicySingleNumaNode {
		for podID := 0; podID < numPods; podID++ {
			validatePodAlignment(f, pods[podID], envInfo)
		}
	}

	for podID := 0; podID < numPods; podID++ {
		pod := pods[podID]
		framework.Logf("deleting the pod %s/%s and waiting for container removal",
			pod.Namespace, pod.Name)
		deletePods(f, []string{pod.Name})
		waitForAllContainerRemoval(pod.Name, pod.Namespace)
	}
}

func runTopologyManagerNegativeTest(f *framework.Framework, numPods int, ctnAttrs []tmCtnAttribute, envInfo *testEnvInfo) {
	podName := "gu-pod"
	framework.Logf("creating pod %s attrs %v", podName, ctnAttrs)
	pod := makeTopologyManagerTestPod(podName, numalignCmd, ctnAttrs)

	pod = f.PodClient().Create(pod)
	err := e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
		if pod.Status.Phase != v1.PodPending {
			return true, nil
		}
		return false, nil
	})
	framework.ExpectNoError(err)
	pod, err = f.PodClient().Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if pod.Status.Phase != v1.PodFailed {
		framework.Failf("pod %s not failed: %v", pod.Name, pod.Status)
	}
	if !isTopologyAffinityError(pod) {
		framework.Failf("pod %s failed for wrong reason: %q", pod.Name, pod.Status.Reason)
	}

	deletePods(f, []string{pod.Name})
}

func isTopologyAffinityError(pod *v1.Pod) bool {
	re := regexp.MustCompile(`Topology.*Affinity.*Error`)
	return re.MatchString(pod.Status.Reason)
}

func getSRIOVDevicePluginConfigMap(cmFile string) *v1.ConfigMap {
	cmData := e2etestfiles.ReadOrDie(SRIOVDevicePluginCMYAML)
	var err error

	// the SRIOVDP configuration is hw-dependent, so we allow per-test-host customization.
	framework.Logf("host-local SRIOV Device Plugin Config Map %q", cmFile)
	if cmFile != "" {
		cmData, err = ioutil.ReadFile(cmFile)
		if err != nil {
			framework.Failf("unable to load the SRIOV Device Plugin ConfigMap: %v", err)
		}
	} else {
		framework.Logf("Using built-in SRIOV Device Plugin Config Map")
	}

	return readConfigMapV1OrDie(cmData)
}

type sriovData struct {
	configMap      *v1.ConfigMap
	serviceAccount *v1.ServiceAccount
	pod            *v1.Pod

	resourceName   string
	resourceAmount int64
}

func setupSRIOVConfigOrFail(f *framework.Framework, configMap *v1.ConfigMap) *sriovData {
	var err error

	ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", metav1.NamespaceSystem, configMap.Name))
	if _, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	serviceAccount := readServiceAccountV1OrDie(e2etestfiles.ReadOrDie(SRIOVDevicePluginSAYAML))
	ginkgo.By(fmt.Sprintf("Creating serviceAccount %v/%v", metav1.NamespaceSystem, serviceAccount.Name))
	if _, err = f.ClientSet.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(context.TODO(), serviceAccount, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test serviceAccount %s: %v", serviceAccount.Name, err)
	}

	e2enode.WaitForNodeToBeReady(f.ClientSet, framework.TestContext.NodeName, 5*time.Minute)

	dp := getSRIOVDevicePluginPod()
	dp.Spec.NodeName = framework.TestContext.NodeName

	ginkgo.By("Create SRIOV device plugin pod")
	dpPod, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if err = e2epod.WaitForPodCondition(f.ClientSet, metav1.NamespaceSystem, dp.Name, "Ready", 120*time.Second, testutils.PodRunningReady); err != nil {
		framework.Logf("SRIOV Pod %v took too long to enter running/ready: %v", dp.Name, err)
	}
	framework.ExpectNoError(err)

	sriovResourceName := ""
	var sriovResourceAmount int64
	ginkgo.By("Waiting for devices to become available on the local node")
	gomega.Eventually(func() bool {
		node := getLocalNode(f)
		sriovResourceName, sriovResourceAmount = findSRIOVResource(node)
		return sriovResourceAmount > minSriovResource
	}, 2*time.Minute, framework.Poll).Should(gomega.BeTrue())
	framework.Logf("Successfully created device plugin pod, detected %d SRIOV allocatable devices %q", sriovResourceAmount, sriovResourceName)

	return &sriovData{
		configMap:      configMap,
		serviceAccount: serviceAccount,
		pod:            dpPod,
		resourceName:   sriovResourceName,
		resourceAmount: sriovResourceAmount,
	}
}

func teardownSRIOVConfigOrFail(f *framework.Framework, sd *sriovData) {
	var err error
	gp := int64(0)
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}

	ginkgo.By("Delete SRIOV device plugin pod %s/%s")
	err = f.ClientSet.CoreV1().Pods(sd.pod.Namespace).Delete(context.TODO(), sd.pod.Name, deleteOptions)
	framework.ExpectNoError(err)
	waitForContainerRemoval(sd.pod.Spec.Containers[0].Name, sd.pod.Name, sd.pod.Namespace)

	ginkgo.By(fmt.Sprintf("Deleting configMap %v/%v", metav1.NamespaceSystem, sd.configMap.Name))
	err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Delete(context.TODO(), sd.configMap.Name, deleteOptions)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Deleting serviceAccount %v/%v", metav1.NamespaceSystem, sd.serviceAccount.Name))
	err = f.ClientSet.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Delete(context.TODO(), sd.serviceAccount.Name, deleteOptions)
	framework.ExpectNoError(err)
}

func runTopologyManagerNodeAlignmentSuiteTests(f *framework.Framework, configMap *v1.ConfigMap, reservedSystemCPUs string, numaNodes, coreCount int, policy string) {
	threadsPerCore := 1
	if isHTEnabled() {
		threadsPerCore = 2
	}

	sd := setupSRIOVConfigOrFail(f, configMap)
	envInfo := &testEnvInfo{
		numaNodes:         numaNodes,
		sriovResourceName: sd.resourceName,
		policy:            policy,
	}

	// could have been a loop, we unroll it to explain the testcases
	var ctnAttrs []tmCtnAttribute

	// simplest case
	ginkgo.By(fmt.Sprintf("Successfully admit one guaranteed pod with 1 core, 1 %s device", sd.resourceName))
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:       "gu-container",
			cpuRequest:    "1000m",
			cpuLimit:      "1000m",
			deviceName:    sd.resourceName,
			deviceRequest: "1",
			deviceLimit:   "1",
		},
	}
	runTopologyManagerPositiveTest(f, 1, ctnAttrs, envInfo)

	ginkgo.By(fmt.Sprintf("Successfully admit one guaranteed pod with 2 cores, 1 %s device", sd.resourceName))
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:       "gu-container",
			cpuRequest:    "2000m",
			cpuLimit:      "2000m",
			deviceName:    sd.resourceName,
			deviceRequest: "1",
			deviceLimit:   "1",
		},
	}
	runTopologyManagerPositiveTest(f, 1, ctnAttrs, envInfo)

	if reservedSystemCPUs != "" {
		// to avoid false negatives, we have put reserved CPUs in such a way there is at least a NUMA node
		// with 1+ SRIOV devices and not reserved CPUs.
		numCores := threadsPerCore * coreCount
		allCoresReq := fmt.Sprintf("%dm", numCores*1000)
		ginkgo.By(fmt.Sprintf("Successfully admit an entire socket (%d cores), 1 %s device", numCores, sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container",
				cpuRequest:    allCoresReq,
				cpuLimit:      allCoresReq,
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerPositiveTest(f, 1, ctnAttrs, envInfo)
	}

	if sd.resourceAmount > 1 {
		// no matter how busses are connected to NUMA nodes and SRIOV devices are installed, this function
		// preconditions must ensure the following can be fulfilled
		ginkgo.By(fmt.Sprintf("Successfully admit two guaranteed pods, each with 1 core, 1 %s device", sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container",
				cpuRequest:    "1000m",
				cpuLimit:      "1000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerPositiveTest(f, 2, ctnAttrs, envInfo)

		ginkgo.By(fmt.Sprintf("Successfully admit two guaranteed pods, each with 2 cores, 1 %s device", sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container",
				cpuRequest:    "2000m",
				cpuLimit:      "2000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerPositiveTest(f, 2, ctnAttrs, envInfo)

		// testing more complex conditions require knowledge about the system cpu+bus topology
	}

	// multi-container tests
	if sd.resourceAmount >= 4 {
		ginkgo.By(fmt.Sprintf("Successfully admit one guaranteed pods, each with two containers, each with 2 cores, 1 %s device", sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container-0",
				cpuRequest:    "2000m",
				cpuLimit:      "2000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
			{
				ctnName:       "gu-container-1",
				cpuRequest:    "2000m",
				cpuLimit:      "2000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerPositiveTest(f, 1, ctnAttrs, envInfo)

		ginkgo.By(fmt.Sprintf("Successfully admit two guaranteed pods, each with two containers, each with 1 core, 1 %s device", sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container-0",
				cpuRequest:    "1000m",
				cpuLimit:      "1000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
			{
				ctnName:       "gu-container-1",
				cpuRequest:    "1000m",
				cpuLimit:      "1000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerPositiveTest(f, 2, ctnAttrs, envInfo)

		ginkgo.By(fmt.Sprintf("Successfully admit two guaranteed pods, each with two containers, both with with 2 cores, one with 1 %s device", sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container-dev",
				cpuRequest:    "2000m",
				cpuLimit:      "2000m",
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
			{
				ctnName:    "gu-container-nodev",
				cpuRequest: "2000m",
				cpuLimit:   "2000m",
			},
		}
		runTopologyManagerPositiveTest(f, 2, ctnAttrs, envInfo)
	}

	// this is the only policy that can guarantee reliable rejects
	if policy == topologymanager.PolicySingleNumaNode {
		// overflow NUMA node capacity: cores
		numCores := 1 + (threadsPerCore * coreCount)
		excessCoresReq := fmt.Sprintf("%dm", numCores*1000)
		ginkgo.By(fmt.Sprintf("Trying to admit a guaranteed pods, with %d cores, 1 %s device - and it should be rejected", numCores, sd.resourceName))
		ctnAttrs = []tmCtnAttribute{
			{
				ctnName:       "gu-container",
				cpuRequest:    excessCoresReq,
				cpuLimit:      excessCoresReq,
				deviceName:    sd.resourceName,
				deviceRequest: "1",
				deviceLimit:   "1",
			},
		}
		runTopologyManagerNegativeTest(f, 1, ctnAttrs, envInfo)
	}
	teardownSRIOVConfigOrFail(f, sd)
}

func runTopologyManagerTests(f *framework.Framework) {
	var oldCfg *kubeletconfig.KubeletConfiguration
	var err error

	ginkgo.It("run Topology Manager policy test suite", func() {
		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)

		var policies = []string{topologymanager.PolicySingleNumaNode, topologymanager.PolicyRestricted,
			topologymanager.PolicyBestEffort, topologymanager.PolicyNone}

		for _, policy := range policies {
			// Configure Topology Manager
			ginkgo.By(fmt.Sprintf("by configuring Topology Manager policy to %s", policy))
			framework.Logf("Configuring topology Manager policy to %s", policy)

			configureTopologyManagerInKubelet(f, oldCfg, policy, nil, 0)
			// Run the tests
			runTopologyManagerPolicySuiteTests(f)
		}
		// restore kubelet config
		setOldKubeletConfig(f, oldCfg)

		// Delete state file to allow repeated runs
		deleteStateFile()
	})

	ginkgo.It("run Topology Manager node alignment test suite", func() {
		// this is a very rough check. We just want to rule out system that does NOT have
		// any SRIOV device. A more proper check will be done in runTopologyManagerPositiveTest
		sriovdevCount := detectSRIOVDevices()
		numaNodes := detectNUMANodes()
		coreCount := detectCoresPerSocket()

		if numaNodes < minNumaNodes {
			e2eskipper.Skipf("this test is meant to run on a multi-node NUMA system")
		}
		if coreCount < minCoreCount {
			e2eskipper.Skipf("this test is meant to run on a system with at least 4 cores per socket")
		}
		if sriovdevCount == 0 {
			e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
		}

		configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)

		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)

		var policies = []string{topologymanager.PolicySingleNumaNode, topologymanager.PolicyRestricted,
			topologymanager.PolicyBestEffort, topologymanager.PolicyNone}

		for _, policy := range policies {
			// Configure Topology Manager
			ginkgo.By(fmt.Sprintf("by configuring Topology Manager policy to %s", policy))
			framework.Logf("Configuring topology Manager policy to %s", policy)

			reservedSystemCPUs := configureTopologyManagerInKubelet(f, oldCfg, policy, configMap, numaNodes)

			runTopologyManagerNodeAlignmentSuiteTests(f, configMap, reservedSystemCPUs, numaNodes, coreCount, policy)
		}

		// restore kubelet config
		setOldKubeletConfig(f, oldCfg)

		// Delete state file to allow repeated runs
		deleteStateFile()
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Topology Manager [Serial] [Feature:TopologyManager][NodeFeature:TopologyManager]", func() {
	f := framework.NewDefaultFramework("topology-manager-test")

	ginkgo.Context("With kubeconfig updated to static CPU Manager policy run the Topology Manager tests", func() {
		runTopologyManagerTests(f)
	})

})
