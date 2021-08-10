/*
Copyright 2020 The Kubernetes Authors.

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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/util"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

type podDesc struct {
	podName        string
	cntName        string
	resourceName   string
	resourceAmount int
	cpuCount       int
}

func makePodResourcesTestPod(desc podDesc) *v1.Pod {
	cnt := v1.Container{
		Name:  desc.cntName,
		Image: busyboxImage,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{},
			Limits:   v1.ResourceList{},
		},
		Command: []string{"sh", "-c", "sleep 1d"},
	}
	if desc.cpuCount > 0 {
		cnt.Resources.Requests[v1.ResourceCPU] = resource.MustParse(fmt.Sprintf("%d", desc.cpuCount))
		cnt.Resources.Limits[v1.ResourceCPU] = resource.MustParse(fmt.Sprintf("%d", desc.cpuCount))
		// we don't really care, we only need to be in guaranteed QoS
		cnt.Resources.Requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		cnt.Resources.Limits[v1.ResourceMemory] = resource.MustParse("100Mi")
	}
	if desc.resourceName != "" && desc.resourceAmount > 0 {
		cnt.Resources.Requests[v1.ResourceName(desc.resourceName)] = resource.MustParse(fmt.Sprintf("%d", desc.resourceAmount))
		cnt.Resources.Limits[v1.ResourceName(desc.resourceName)] = resource.MustParse(fmt.Sprintf("%d", desc.resourceAmount))
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: desc.podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				cnt,
			},
		},
	}
}

func logPodResources(podIdx int, pr *kubeletpodresourcesv1.PodResources) {
	ns := pr.GetNamespace()
	cnts := pr.GetContainers()
	if len(cnts) == 0 {
		framework.Logf("#%02d/%02d/%02d - %s/%s/%s   No containers", podIdx, 0, 0, ns, pr.GetName(), "_")
		return
	}

	for cntIdx, cnt := range cnts {
		if len(cnt.Devices) == 0 {
			framework.Logf("#%02d/%02d/%02d - %s/%s/%s   cpus -> %v   resources -> none", podIdx, cntIdx, 0, ns, pr.GetName(), cnt.Name, cnt.CpuIds)
			continue
		}

		for devIdx, dev := range cnt.Devices {
			framework.Logf("#%02d/%02d/%02d - %s/%s/%s   cpus -> %v   %s -> %s", podIdx, cntIdx, devIdx, ns, pr.GetName(), cnt.Name, cnt.CpuIds, dev.ResourceName, strings.Join(dev.DeviceIds, ", "))
		}
	}
}

type podResMap map[string]map[string]kubeletpodresourcesv1.ContainerResources

func getPodResources(cli kubeletpodresourcesv1.PodResourcesListerClient) podResMap {
	resp, err := cli.List(context.TODO(), &kubeletpodresourcesv1.ListPodResourcesRequest{})
	framework.ExpectNoError(err)

	res := make(map[string]map[string]kubeletpodresourcesv1.ContainerResources)
	for idx, podResource := range resp.GetPodResources() {
		// to make troubleshooting easier
		logPodResources(idx, podResource)

		cnts := make(map[string]kubeletpodresourcesv1.ContainerResources)
		for _, cnt := range podResource.GetContainers() {
			cnts[cnt.GetName()] = *cnt
		}
		res[podResource.GetName()] = cnts
	}
	return res
}

type testPodData struct {
	PodMap map[string]*v1.Pod
}

func newTestPodData() *testPodData {
	return &testPodData{
		PodMap: make(map[string]*v1.Pod),
	}
}

func (tpd *testPodData) createPodsForTest(f *framework.Framework, podReqs []podDesc) {
	for _, podReq := range podReqs {
		pod := makePodResourcesTestPod(podReq)
		pod = f.PodClient().CreateSync(pod)

		framework.Logf("created pod %s", podReq.podName)
		tpd.PodMap[podReq.podName] = pod
	}
}

/* deletePodsForTest clean up all the pods run for a testcase. Must ensure proper cleanup */
func (tpd *testPodData) deletePodsForTest(f *framework.Framework) {
	deletePodsAsync(f, tpd.PodMap)
}

/* deletePod removes pod during a test. Should do a best-effort clean up */
func (tpd *testPodData) deletePod(f *framework.Framework, podName string) {
	_, ok := tpd.PodMap[podName]
	if !ok {
		return
	}
	deletePodSyncByName(f, podName)
	delete(tpd.PodMap, podName)
}

func findContainerDeviceByName(devs []*kubeletpodresourcesv1.ContainerDevices, resourceName string) *kubeletpodresourcesv1.ContainerDevices {
	for _, dev := range devs {
		if dev.ResourceName == resourceName {
			return dev
		}
	}
	return nil
}

func matchPodDescWithResources(expected []podDesc, found podResMap) error {
	for _, podReq := range expected {
		framework.Logf("matching: %#v", podReq)

		podInfo, ok := found[podReq.podName]
		if !ok {
			return fmt.Errorf("no pod resources for pod %q", podReq.podName)
		}
		cntInfo, ok := podInfo[podReq.cntName]
		if !ok {
			return fmt.Errorf("no container resources for pod %q container %q", podReq.podName, podReq.cntName)
		}

		if podReq.cpuCount > 0 {
			if len(cntInfo.CpuIds) != podReq.cpuCount {
				return fmt.Errorf("pod %q container %q expected %d cpus got %v", podReq.podName, podReq.cntName, podReq.cpuCount, cntInfo.CpuIds)
			}
		}

		if podReq.resourceName != "" && podReq.resourceAmount > 0 {
			dev := findContainerDeviceByName(cntInfo.GetDevices(), podReq.resourceName)
			if dev == nil {
				return fmt.Errorf("pod %q container %q expected data for resource %q not found", podReq.podName, podReq.cntName, podReq.resourceName)
			}
			if len(dev.DeviceIds) != podReq.resourceAmount {
				return fmt.Errorf("pod %q container %q resource %q expected %d items got %v", podReq.podName, podReq.cntName, podReq.resourceName, podReq.resourceAmount, dev.DeviceIds)
			}
		} else {
			devs := cntInfo.GetDevices()
			if len(devs) > 0 {
				return fmt.Errorf("pod %q container %q expected no resources, got %v", podReq.podName, podReq.cntName, devs)
			}
		}
	}
	return nil
}

func expectPodResources(offset int, cli kubeletpodresourcesv1.PodResourcesListerClient, expected []podDesc) {
	gomega.EventuallyWithOffset(1+offset, func() error {
		found := getPodResources(cli)
		return matchPodDescWithResources(expected, found)
	}, time.Minute, 10*time.Second).Should(gomega.BeNil())
}

func filterOutDesc(descs []podDesc, name string) []podDesc {
	var ret []podDesc
	for _, desc := range descs {
		if desc.podName == name {
			continue
		}
		ret = append(ret, desc)
	}
	return ret
}

func podresourcesListTests(f *framework.Framework, cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData) {
	var tpd *testPodData

	var found podResMap
	var expected []podDesc
	var extra podDesc

	expectedBasePods := 0 /* nothing but pods we create */
	if sd != nil {
		expectedBasePods = 1 // sriovdp
	}

	ginkgo.By("checking the output when no pods are present")
	found = getPodResources(cli)
	gomega.ExpectWithOffset(1, found).To(gomega.HaveLen(expectedBasePods), "base pod expectation mismatch")

	tpd = newTestPodData()
	ginkgo.By("checking the output when only pods which don't require resources are present")
	expected = []podDesc{
		{
			podName: "pod-00",
			cntName: "cnt-00",
		},
		{
			podName: "pod-01",
			cntName: "cnt-00",
		},
	}
	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when only a subset of pods require resources")
	if sd != nil {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:        "pod-01",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuCount:       2,
			},
			{
				podName:  "pod-02",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
			{
				podName:        "pod-03",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuCount:       1,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:  "pod-01",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
			{
				podName:  "pod-02",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
			{
				podName:  "pod-03",
				cntName:  "cnt-00",
				cpuCount: 1,
			},
		}

	}
	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when creating pods which require resources between calls")
	if sd != nil {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:        "pod-01",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuCount:       2,
			},
			{
				podName:  "pod-02",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:  "pod-01",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
			{
				podName:  "pod-02",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
		}
	}

	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)

	if sd != nil {
		extra = podDesc{
			podName:        "pod-03",
			cntName:        "cnt-00",
			resourceName:   sd.resourceName,
			resourceAmount: 1,
			cpuCount:       1,
		}
	} else {
		extra = podDesc{
			podName:  "pod-03",
			cntName:  "cnt-00",
			cpuCount: 1,
		}

	}

	tpd.createPodsForTest(f, []podDesc{
		extra,
	})

	expected = append(expected, extra)
	expectPodResources(1, cli, expected)
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when deleting pods which require resources between calls")

	if sd != nil {
		expected = []podDesc{
			{
				podName:  "pod-00",
				cntName:  "cnt-00",
				cpuCount: 1,
			},
			{
				podName:        "pod-01",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuCount:       2,
			},
			{
				podName: "pod-02",
				cntName: "cnt-00",
			},
			{
				podName:        "pod-03",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuCount:       1,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName:  "pod-00",
				cntName:  "cnt-00",
				cpuCount: 1,
			},
			{
				podName:  "pod-01",
				cntName:  "cnt-00",
				cpuCount: 2,
			},
			{
				podName: "pod-02",
				cntName: "cnt-00",
			},
			{
				podName:  "pod-03",
				cntName:  "cnt-00",
				cpuCount: 1,
			},
		}
	}
	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)

	tpd.deletePod(f, "pod-01")
	expectedPostDelete := filterOutDesc(expected, "pod-01")
	expectPodResources(1, cli, expectedPostDelete)
	tpd.deletePodsForTest(f)
}

func podresourcesGetAllocatableResourcesTests(f *framework.Framework, cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData, onlineCPUs, reservedSystemCPUs cpuset.CPUSet) {
	ginkgo.By("checking the devices known to the kubelet")
	resp, err := cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
	framework.ExpectNoErrorWithOffset(1, err)
	devs := resp.GetDevices()
	allocatableCPUs := cpuset.NewCPUSetInt64(resp.GetCpuIds()...)

	if onlineCPUs.Size() == 0 {
		ginkgo.By("expecting no CPUs reported")
		gomega.ExpectWithOffset(1, onlineCPUs.Size()).To(gomega.Equal(reservedSystemCPUs.Size()), "with no online CPUs, no CPUs should be reserved")
	} else {
		ginkgo.By(fmt.Sprintf("expecting online CPUs reported - online=%v (%d) reserved=%v (%d)", onlineCPUs, onlineCPUs.Size(), reservedSystemCPUs, reservedSystemCPUs.Size()))
		if reservedSystemCPUs.Size() > onlineCPUs.Size() {
			ginkgo.Fail("more reserved CPUs than online")
		}
		expectedCPUs := onlineCPUs.Difference(reservedSystemCPUs)

		ginkgo.By(fmt.Sprintf("expecting CPUs '%v'='%v'", allocatableCPUs, expectedCPUs))
		gomega.ExpectWithOffset(1, allocatableCPUs.Equals(expectedCPUs)).To(gomega.BeTrue(), "mismatch expecting CPUs")
	}

	if sd == nil { // no devices in the environment, so expect no devices
		ginkgo.By("expecting no devices reported")
		gomega.ExpectWithOffset(1, devs).To(gomega.BeEmpty(), fmt.Sprintf("got unexpected devices %#v", devs))
		return
	}

	ginkgo.By(fmt.Sprintf("expecting some %q devices reported", sd.resourceName))
	gomega.ExpectWithOffset(1, devs).ToNot(gomega.BeEmpty())
	for _, dev := range devs {
		framework.ExpectEqual(dev.ResourceName, sd.resourceName)
		gomega.ExpectWithOffset(1, dev.DeviceIds).ToNot(gomega.BeEmpty())
	}
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("POD Resources [Serial] [Feature:PodResources][NodeFeature:PodResources]", func() {
	f := framework.NewDefaultFramework("podresources-test")

	reservedSystemCPUs := cpuset.MustParse("1")

	ginkgo.Context("With SRIOV devices in the system", func() {
		ginkgo.It("should return the expected responses with cpumanager static policy enabled", func() {
			// this is a very rough check. We just want to rule out system that does NOT have enough resources
			_, cpuAlloc, _ := getLocalNodeCPUDetails(f)

			if cpuAlloc < minCoreCount {
				e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
			}
			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount == 0 {
				e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
			}

			onlineCPUs, err := getOnlineCPUs()
			framework.ExpectNoError(err)

			// Make sure all the feature gates and the right settings are in place.
			oldCfg := configurePodResourcesInKubelet(f, true, reservedSystemCPUs)
			defer func() {
				// restore kubelet config
				setOldKubeletConfig(f, oldCfg)

				// Delete state file to allow repeated runs
				deleteStateFile()
			}()

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)
			defer teardownSRIOVConfigOrFail(f, sd)

			waitForSRIOVResources(f, sd)

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			waitForSRIOVResources(f, sd)

			ginkgo.By("checking List()")
			podresourcesListTests(f, cli, sd)
			ginkgo.By("checking GetAllocatableResources()")
			podresourcesGetAllocatableResourcesTests(f, cli, sd, onlineCPUs, reservedSystemCPUs)

		})

		ginkgo.It("should return the expected responses with cpumanager none policy", func() {
			// current default is "none" policy - no need to restart the kubelet

			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount == 0 {
				e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
			}

			oldCfg := enablePodResourcesFeatureGateInKubelet(f)
			defer func() {
				// restore kubelet config
				setOldKubeletConfig(f, oldCfg)

				// Delete state file to allow repeated runs
				deleteStateFile()
			}()

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)
			defer teardownSRIOVConfigOrFail(f, sd)

			waitForSRIOVResources(f, sd)

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			waitForSRIOVResources(f, sd)

			// intentionally passing empty cpuset instead of onlineCPUs because with none policy
			// we should get no allocatable cpus - no exclusively allocatable CPUs, depends on policy static
			podresourcesGetAllocatableResourcesTests(f, cli, sd, cpuset.CPUSet{}, cpuset.CPUSet{})
		})

	})

	ginkgo.Context("Without SRIOV devices in the system", func() {
		ginkgo.It("should return the expected responses with cpumanager static policy enabled", func() {
			// this is a very rough check. We just want to rule out system that does NOT have enough resources
			_, cpuAlloc, _ := getLocalNodeCPUDetails(f)

			if cpuAlloc < minCoreCount {
				e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
			}
			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount > 0 {
				e2eskipper.Skipf("this test is meant to run on a system with no configured VF from SRIOV device")
			}

			onlineCPUs, err := getOnlineCPUs()
			framework.ExpectNoError(err)

			// Make sure all the feature gates and the right settings are in place.
			oldCfg := configurePodResourcesInKubelet(f, true, reservedSystemCPUs)
			defer func() {
				// restore kubelet config
				setOldKubeletConfig(f, oldCfg)

				// Delete state file to allow repeated runs
				deleteStateFile()
			}()

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			podresourcesListTests(f, cli, nil)
			podresourcesGetAllocatableResourcesTests(f, cli, nil, onlineCPUs, reservedSystemCPUs)
		})

		ginkgo.It("should return the expected responses with cpumanager none policy", func() {
			// current default is "none" policy - no need to restart the kubelet

			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount > 0 {
				e2eskipper.Skipf("this test is meant to run on a system with no configured VF from SRIOV device")
			}

			oldCfg := enablePodResourcesFeatureGateInKubelet(f)
			defer func() {
				// restore kubelet config
				setOldKubeletConfig(f, oldCfg)

				// Delete state file to allow repeated runs
				deleteStateFile()
			}()

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			// intentionally passing empty cpuset instead of onlineCPUs because with none policy
			// we should get no allocatable cpus - no exclusively allocatable CPUs, depends on policy static
			podresourcesGetAllocatableResourcesTests(f, cli, nil, cpuset.CPUSet{}, cpuset.CPUSet{})
		})

		ginkgo.It("should return the expected error with the feature gate disabled", func() {
			if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.KubeletPodResourcesGetAllocatable) {
				e2eskipper.Skipf("this test is meant to run with the POD Resources Extensions feature gate disabled")
			}

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			ginkgo.By("checking GetAllocatableResources fail if the feature gate is not enabled")
			_, err = cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectError(err, "With feature gate disabled, the call must fail")
		})

	})
})

func getOnlineCPUs() (cpuset.CPUSet, error) {
	onlineCPUList, err := ioutil.ReadFile("/sys/devices/system/cpu/online")
	if err != nil {
		return cpuset.CPUSet{}, err
	}
	return cpuset.Parse(strings.TrimSpace(string(onlineCPUList)))
}

func configurePodResourcesInKubelet(f *framework.Framework, cleanStateFile bool, reservedSystemCPUs cpuset.CPUSet) (oldCfg *kubeletconfig.KubeletConfiguration) {
	// we also need CPUManager with static policy to be able to do meaningful testing
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}
	newCfg.FeatureGates["CPUManager"] = true
	newCfg.FeatureGates["KubeletPodResourcesGetAllocatable"] = true

	// After graduation of the CPU Manager feature to Beta, the CPU Manager
	// "none" policy is ON by default. But when we set the CPU Manager policy to
	// "static" in this test and the Kubelet is restarted so that "static"
	// policy can take effect, there will always be a conflict with the state
	// checkpointed in the disk (i.e., the policy checkpointed in the disk will
	// be "none" whereas we are trying to restart Kubelet with "static"
	// policy). Therefore, we delete the state file so that we can proceed
	// with the tests.
	// Only delete the state file at the begin of the tests.
	if cleanStateFile {
		deleteStateFile()
	}

	// Set the CPU Manager policy to static.
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)

	// Set the CPU Manager reconcile period to 1 second.
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	if reservedSystemCPUs.Size() > 0 {
		cpus := reservedSystemCPUs.String()
		framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
		newCfg.ReservedSystemCPUs = cpus
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
	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return oldCfg
}

func enablePodResourcesFeatureGateInKubelet(f *framework.Framework) (oldCfg *kubeletconfig.KubeletConfiguration) {
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}
	newCfg.FeatureGates["KubeletPodResourcesGetAllocatable"] = true

	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return oldCfg
}
