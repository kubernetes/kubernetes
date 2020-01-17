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
	"os/exec"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// Helper for makeTopologyManagerPod().
type tmCtnAttribute struct {
	ctnName     string
	cpuRequest  string
	cpuLimit    string
	devResource string
}

func detectNUMANodes() int {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"NUMA node(s):\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	numaNodes, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return numaNodes
}

// TODO: what about HT?
func detectCoresPerSocket() int {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"Core(s) per socket:\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	coreCount, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return coreCount
}

func detectSRIOVDevices() int {
	outData, err := exec.Command("/bin/sh", "-c", "ls /sys/bus/pci/devices/*/sriov_totalvfs | wc -w").Output()
	framework.ExpectNoError(err)

	devCount, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return devCount
}

// makeTopologyMangerPod returns a pod with the provided tmCtnAttributes.
func makeTopologyManagerPod(podName string, tmCtnAttributes []tmCtnAttribute) *v1.Pod {
	cpusetCmd := "grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d"
	return makeTopologyManagerTestPod(podName, cpusetCmd, tmCtnAttributes)
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
		if ctnAttr.devResource != "" {
			ctn.Resources.Requests[v1.ResourceName(ctnAttr.devResource)] = resource.MustParse("1")
			ctn.Resources.Limits[v1.ResourceName(ctnAttr.devResource)] = resource.MustParse("1")
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

func configureTopologyManagerInKubelet(f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration, policy string) {
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

	// The Kubelet panics if either kube-reserved or system-reserved is not set
	// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
	// kubelet doesn't panic.
	if newCfg.KubeReserved == nil {
		newCfg.KubeReserved = map[string]string{}
	}

	if _, ok := newCfg.KubeReserved["cpu"]; !ok {
		newCfg.KubeReserved["cpu"] = "200m"
	}
	// Dump the config -- debug
	framework.Logf("New kublet config is %s", *newCfg)

	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())
}

// getSRIOVDevicePluginPod returns the Device Plugin pod for sriov resources in e2e tests.
func getSRIOVDevicePluginPod() *v1.Pod {
	ds := readDaemonSetV1OrDie(testfiles.ReadOrDie(SRIOVDevicePluginDSYAML))
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

// numberOfResources returns the number of resources advertised by a node.
func numberOfResources(node *v1.Node, resourceKey string) int64 {
	val, ok := node.Status.Capacity[v1.ResourceName(resourceKey)]

	if !ok {
		return 0
	}

	return val.Value()
}

func deletePodInNamespace(f *framework.Framework, namespace, name string) {
	gp := int64(0)
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}
	err := f.ClientSet.CoreV1().Pods(namespace).Delete(context.TODO(), name, &deleteOptions)
	framework.ExpectNoError(err)
}

func runTopologyManagerPolicySuiteTests(f *framework.Framework) {
	var cpuCap, cpuAlloc int64
	var cpuListString, expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1, cpu2 int
	var cset cpuset.CPUSet
	var err error
	var ctnAttrs []tmCtnAttribute
	var pod, pod1, pod2 *v1.Pod

	cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(f)

	ginkgo.By("running a non-Gu pod")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "100m",
			cpuLimit:   "200m",
		},
	}
	pod = makeTopologyManagerPod("non-gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	expAllowedCPUsListRegex = fmt.Sprintf("^0-%d\n$", cpuCap-1)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

	ginkgo.By("running a Gu pod")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod = makeTopologyManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1 = 1
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		cpu1 = cpuList[1]
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

	ginkgo.By("running multiple Gu and non-Gu pods")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod1 = makeTopologyManagerPod("gu-pod", ctnAttrs)
	pod1 = f.PodClient().CreateSync(pod1)

	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "200m",
			cpuLimit:   "300m",
		},
	}
	pod2 = makeTopologyManagerPod("non-gu-pod", ctnAttrs)
	pod2 = f.PodClient().CreateSync(pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1 = 1
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		cpu1 = cpuList[1]
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = f.PodClient().MatchContainerOutput(pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	cpuListString = "0"
	if cpuAlloc > 2 {
		cset = cpuset.MustParse(fmt.Sprintf("0-%d", cpuCap-1))
		cpuListString = fmt.Sprintf("%s", cset.Difference(cpuset.NewCPUSet(cpu1)))
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = f.PodClient().MatchContainerOutput(pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)

	// Skip rest of the tests if CPU capacity < 3.
	if cpuCap < 3 {
		e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
	}

	ginkgo.By("running a Gu pod requesting multiple CPUs")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "2000m",
			cpuLimit:   "2000m",
		},
	}
	pod = makeTopologyManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpuListString = "1-2"
	if isHTEnabled() {
		cpuListString = "2-3"
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cset = cpuset.MustParse(getCPUSiblingList(1))
			cpuListString = fmt.Sprintf("%s", cset)
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

	ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container1",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
		{
			ctnName:    "gu-container2",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod = makeTopologyManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
	}

	expAllowedCPUsListRegex = fmt.Sprintf("^%d|%d\n$", cpu1, cpu2)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[1].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
	waitForContainerRemoval(pod.Spec.Containers[1].Name, pod.Name, pod.Namespace)

	ginkgo.By("running multiple Gu pods")
	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container1",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod1 = makeTopologyManagerPod("gu-pod1", ctnAttrs)
	pod1 = f.PodClient().CreateSync(pod1)

	ctnAttrs = []tmCtnAttribute{
		{
			ctnName:    "gu-container2",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod2 = makeTopologyManagerPod("gu-pod2", ctnAttrs)
	pod2 = f.PodClient().CreateSync(pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
	}

	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = f.PodClient().MatchContainerOutput(pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu2)
	err = f.PodClient().MatchContainerOutput(pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)
}

func runTopologyManagerNodeAlignmentSuiteTests(f *framework.Framework) {
	var err error

	configMap := readConfigMapV1OrDie(testfiles.ReadOrDie(SRIOVDevicePluginCMYAML))
	ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", metav1.NamespaceSystem, configMap.Name))
	if _, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	serviceAccount := readServiceAccountV1OrDie(testfiles.ReadOrDie(SRIOVDevicePluginSAYAML))
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

	ginkgo.By("Waiting for devices to become available on the local node")
	gomega.Eventually(func() bool {
		node := getLocalNode(f)
		framework.Logf("Node status: %v", node.Status.Capacity)
		return numberOfResources(node, SRIOVResourceName) > 0
	}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
	framework.Logf("Successfully created device plugin pod")

	ginkgo.By("running a Gu pod")
	ctnAttrs := []tmCtnAttribute{
		{
			ctnName:     "gu-container",
			cpuRequest:  "1000m",
			cpuLimit:    "1000m",
			devResource: SRIOVResourceName,
		},
	}

	pod := makeTopologyManagerTestPod("gu-pod", "env && sleep 1d", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("validating the Gu pod")
	_, err = e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

	framework.Logf("deleting the SRIOV device plugin pod %s/%s and waiting for container %s removal",
		dpPod.Namespace, dpPod.Name, dpPod.Spec.Containers[0].Name)
	deletePodInNamespace(f, dpPod.Namespace, dpPod.Name)
	waitForContainerRemoval(dpPod.Spec.Containers[0].Name, dpPod.Name, dpPod.Namespace)
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

			configureTopologyManagerInKubelet(f, oldCfg, policy)
			// Run the tests
			runTopologyManagerPolicySuiteTests(f)
		}
		// restore kubelet config
		setOldKubeletConfig(f, oldCfg)

		// Delete state file to allow repeated runs
		deleteStateFile()
	})

	ginkgo.It("run Topology Manager node alignment test suite", func() {
		numaNodes := detectNUMANodes()
		coreCount := detectCoresPerSocket()
		sriovdevCount := detectSRIOVDevices()

		if numaNodes < 2 {
			e2eskipper.Skipf("this test is meant to run on a multi-node NUMA system")
		}
		if coreCount < 4 {
			e2eskipper.Skipf("this test is meant to run on a system with at least 4 cores per socket")
		}
		if sriovdevCount == 0 {
			e2eskipper.Skipf("this test is meant to run on a system with at least one SRIOV device")
		}

		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)

		policy := topologymanager.PolicySingleNumaNode

		// Configure Topology Manager
		ginkgo.By(fmt.Sprintf("by configuring Topology Manager policy to %s", policy))
		framework.Logf("Configuring topology Manager policy to %s", policy)

		configureTopologyManagerInKubelet(f, oldCfg, policy)

		runTopologyManagerNodeAlignmentSuiteTests(f)

		// restore kubelet config
		setOldKubeletConfig(f, oldCfg)

		// Delete state file to allow repeated runs
		deleteStateFile()
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Topology Manager [Serial] [Feature:TopologyManager][NodeAlphaFeature:TopologyManager]", func() {
	f := framework.NewDefaultFramework("topology-manager-test")

	ginkgo.Context("With kubeconfig updated to static CPU Manager policy run the Topology Manager tests", func() {
		runTopologyManagerTests(f)
	})

})
