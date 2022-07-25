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
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/util"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
)

type podDesc struct {
	podName        string
	cntName        string
	resourceName   string
	resourceAmount int
	cpuRequest     int // cpuRequest is in millicores
}

func (desc podDesc) CpuRequestQty() resource.Quantity {
	qty := resource.NewMilliQuantity(int64(desc.cpuRequest), resource.DecimalSI)
	return *qty
}

func (desc podDesc) CpuRequestExclusive() int {
	if (desc.cpuRequest % 1000) != 0 {
		// exclusive cpus are request only if the quantity is integral;
		// hence, explicitly rule out non-integral requests
		return 0
	}
	return desc.cpuRequest / 1000
}

func (desc podDesc) RequiresCPU() bool {
	return desc.cpuRequest > 0
}

func (desc podDesc) RequiresDevices() bool {
	return desc.resourceName != "" && desc.resourceAmount > 0
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
	if desc.RequiresCPU() {
		cpuRequestQty := desc.CpuRequestQty()
		cnt.Resources.Requests[v1.ResourceCPU] = cpuRequestQty
		cnt.Resources.Limits[v1.ResourceCPU] = cpuRequestQty
		// we don't really care, we only need to be in guaranteed QoS
		cnt.Resources.Requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		cnt.Resources.Limits[v1.ResourceMemory] = resource.MustParse("100Mi")
	}
	if desc.RequiresDevices() {
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
		if podReq.RequiresCPU() {
			if exclusiveCpus := podReq.CpuRequestExclusive(); exclusiveCpus != len(cntInfo.CpuIds) {
				if exclusiveCpus == 0 {
					return fmt.Errorf("pod %q container %q requested %d expected to be allocated CPUs from shared pool %v", podReq.podName, podReq.cntName, podReq.cpuRequest, cntInfo.CpuIds)
				}
				return fmt.Errorf("pod %q container %q expected %d cpus got %v", podReq.podName, podReq.cntName, exclusiveCpus, cntInfo.CpuIds)
			}
		}
		if podReq.RequiresDevices() {
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
		if cnts, ok := found[KubeVirtResourceName]; ok {
			for _, cnt := range cnts {
				for _, cd := range cnt.GetDevices() {
					if cd.ResourceName != KubeVirtResourceName {
						continue
					}
					if cd.Topology != nil {
						//we expect nil topology
						return fmt.Errorf("Nil topology is expected")
					}
				}

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
				cpuRequest:     2000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 2000,
			},
			{
				podName:        "pod-03",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuRequest:     1000,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:    "pod-01",
				cntName:    "cnt-00",
				cpuRequest: 2000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 2000,
			},
			{
				podName:    "pod-03",
				cntName:    "cnt-00",
				cpuRequest: 1000,
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
				cpuRequest:     2000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 2000,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName: "pod-00",
				cntName: "cnt-00",
			},
			{
				podName:    "pod-01",
				cntName:    "cnt-00",
				cpuRequest: 2000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 2000,
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
			cpuRequest:     1000,
		}
	} else {
		extra = podDesc{
			podName:    "pod-03",
			cntName:    "cnt-00",
			cpuRequest: 1000,
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
				podName:    "pod-00",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
			{
				podName:        "pod-01",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuRequest:     2000,
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
				cpuRequest:     1000,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName:    "pod-00",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
			{
				podName:    "pod-01",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
			{
				podName: "pod-02",
				cntName: "cnt-00",
			},
			{
				podName:    "pod-03",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
		}
	}
	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)

	tpd.deletePod(f, "pod-01")
	expectedPostDelete := filterOutDesc(expected, "pod-01")
	expectPodResources(1, cli, expectedPostDelete)
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when pods request non integral CPUs")
	if sd != nil {
		expected = []podDesc{
			{
				podName:    "pod-00",
				cntName:    "cnt-00",
				cpuRequest: 1500,
			},
			{
				podName:        "pod-01",
				cntName:        "cnt-00",
				resourceName:   sd.resourceName,
				resourceAmount: 1,
				cpuRequest:     1500,
			},
		}
	} else {
		expected = []podDesc{
			{
				podName:    "pod-00",
				cntName:    "cnt-00",
				cpuRequest: 1500,
			},
		}

	}
	tpd.createPodsForTest(f, expected)
	expectPodResources(1, cli, expected)
	tpd.deletePodsForTest(f)

}

func podresourcesGetAllocatableResourcesTests(cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData, onlineCPUs, reservedSystemCPUs cpuset.CPUSet) {
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
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	reservedSystemCPUs := cpuset.MustParse("1")

	ginkgo.Context("with SRIOV devices in the system", func() {
		ginkgo.BeforeEach(func() {
			requireSRIOVDevices()
		})

		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func() {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
				})

				ginkgo.It("should return the expected responses", func() {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err)

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
					podresourcesGetAllocatableResourcesTests(cli, sd, onlineCPUs, reservedSystemCPUs)
				})
			})
		})

		ginkgo.Context("with CPU manager None policy", func() {
			ginkgo.It("should return the expected responses", func() {
				// current default is "none" policy - no need to restart the kubelet

				requireSRIOVDevices()

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
				podresourcesGetAllocatableResourcesTests(cli, sd, cpuset.CPUSet{}, cpuset.CPUSet{})
			})
		})
	})

	ginkgo.Context("without SRIOV devices in the system", func() {
		ginkgo.BeforeEach(func() {
			requireLackOfSRIOVDevices()
		})

		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func() {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
				})

				ginkgo.It("should return the expected responses", func() {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err)
					defer conn.Close()

					podresourcesListTests(f, cli, nil)
					podresourcesGetAllocatableResourcesTests(cli, nil, onlineCPUs, reservedSystemCPUs)
				})
			})
		})

		ginkgo.Context("with CPU manager None policy", func() {
			ginkgo.It("should return the expected responses", func() {
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)
				defer conn.Close()

				// intentionally passing empty cpuset instead of onlineCPUs because with none policy
				// we should get no allocatable cpus - no exclusively allocatable CPUs, depends on policy static
				podresourcesGetAllocatableResourcesTests(cli, nil, cpuset.CPUSet{}, cpuset.CPUSet{})
			})
		})

		ginkgo.Context("with disabled KubeletPodResourcesGetAllocatable feature gate", func() {
			tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
				if initialConfig.FeatureGates == nil {
					initialConfig.FeatureGates = make(map[string]bool)
				}
				initialConfig.FeatureGates[string(kubefeatures.KubeletPodResourcesGetAllocatable)] = false
			})

			ginkgo.It("should return the expected error with the feature gate disabled", func() {
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)
				defer conn.Close()

				ginkgo.By("checking GetAllocatableResources fail if the feature gate is not enabled")
				allocatableRes, err := cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
				framework.Logf("GetAllocatableResources result: %v, err: %v", allocatableRes, err)
				framework.ExpectError(err, "With feature gate disabled, the call must fail")
			})
		})
	})

	ginkgo.Context("with KubeVirt device plugin, which reports resources w/o hardware topology", func() {
		ginkgo.BeforeEach(func() {
			_, err := os.Stat("/dev/kvm")
			if errors.Is(err, os.ErrNotExist) {
				e2eskipper.Skipf("KubeVirt device plugin could work only in kvm based environment")
			}
		})

		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func() {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
				})

				ginkgo.It("should return proper podresources the same as before the restart of kubelet", func() {
					dpPod := setupKubeVirtDevicePluginOrFail(f)
					defer teardownKubeVirtDevicePluginOrFail(f, dpPod)

					waitForKubeVirtResources(f, dpPod)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err)
					defer conn.Close()

					ginkgo.By("checking List and resources kubevirt resource should be without topology")

					allocatableResponse, _ := cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
					for _, dev := range allocatableResponse.GetDevices() {
						if dev.ResourceName != KubeVirtResourceName {
							continue
						}

						framework.ExpectEqual(dev.Topology == nil, true, "Topology is expected to be empty for kubevirt resources")
					}

					// Run pod which requires KubeVirtResourceName
					desc := podDesc{
						podName:        "pod-01",
						cntName:        "cnt-01",
						resourceName:   KubeVirtResourceName,
						resourceAmount: 1,
						cpuRequest:     1000,
					}

					tpd := newTestPodData()
					tpd.createPodsForTest(f, []podDesc{
						desc,
					})

					expectPodResources(1, cli, []podDesc{desc})

					restartTime := time.Now()
					ginkgo.By("Restarting Kubelet")
					restartKubelet(true)

					// we need to wait for the node to be reported ready before we can safely query
					// the podresources endpoint again. Otherwise we will have false negatives.
					ginkgo.By("Wait for node to be ready")
					gomega.Eventually(func() bool {
						node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
						framework.ExpectNoError(err)
						for _, cond := range node.Status.Conditions {
							if cond.Type == v1.NodeReady && cond.Status == v1.ConditionTrue && cond.LastHeartbeatTime.After(restartTime) {
								return true
							}
						}
						return false
					}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

					expectPodResources(1, cli, []podDesc{desc})
					tpd.deletePodsForTest(f)
				})
			})
		})
	})
})

func requireLackOfSRIOVDevices() {
	if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount > 0 {
		e2eskipper.Skipf("this test is meant to run on a system with no configured VF from SRIOV device")
	}
}

func getOnlineCPUs() (cpuset.CPUSet, error) {
	onlineCPUList, err := os.ReadFile("/sys/devices/system/cpu/online")
	if err != nil {
		return cpuset.CPUSet{}, err
	}
	return cpuset.Parse(strings.TrimSpace(string(onlineCPUList)))
}

func setupKubeVirtDevicePluginOrFail(f *framework.Framework) *v1.Pod {
	e2enode.WaitForNodeToBeReady(f.ClientSet, framework.TestContext.NodeName, 5*time.Minute)

	dp := getKubeVirtDevicePluginPod()
	dp.Spec.NodeName = framework.TestContext.NodeName

	ginkgo.By("Create KubeVirt device plugin pod")

	dpPod, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if err = e2epod.WaitForPodCondition(f.ClientSet, metav1.NamespaceSystem, dp.Name, "Ready", 120*time.Second, testutils.PodRunningReady); err != nil {
		framework.Logf("KubeVirt Pod %v took too long to enter running/ready: %v", dp.Name, err)
	}
	framework.ExpectNoError(err)

	return dpPod
}

func teardownKubeVirtDevicePluginOrFail(f *framework.Framework, pod *v1.Pod) {
	gp := int64(0)
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}
	ginkgo.By(fmt.Sprintf("Delete KubeVirt device plugin pod %s/%s", pod.Namespace, pod.Name))
	err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, deleteOptions)

	framework.ExpectNoError(err)
	waitForAllContainerRemoval(pod.Name, pod.Namespace)
}

func findKubeVirtResource(node *v1.Node) int64 {
	framework.Logf("Node status allocatable: %v", node.Status.Allocatable)
	for key, val := range node.Status.Allocatable {
		if string(key) == KubeVirtResourceName {
			v := val.Value()
			if v > 0 {
				return v
			}
		}
	}
	return 0
}

func waitForKubeVirtResources(f *framework.Framework, pod *v1.Pod) {
	ginkgo.By("Waiting for kubevirt resources to become available on the local node")

	gomega.Eventually(func() bool {
		node := getLocalNode(f)
		kubeVirtResourceAmount := findKubeVirtResource(node)
		return kubeVirtResourceAmount != 0
	}, 2*time.Minute, framework.Poll).Should(gomega.BeTrue())
}

// getKubeVirtDevicePluginPod returns the Device Plugin pod for kube resources in e2e tests.
func getKubeVirtDevicePluginPod() *v1.Pod {
	data, err := e2etestfiles.Read(KubeVirtDevicePluginDSYAML)
	if err != nil {
		framework.Fail(err.Error())
	}

	ds := readDaemonSetV1OrDie(data)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      KubeVirtDevicePluginName,
			Namespace: metav1.NamespaceSystem,
		},

		Spec: ds.Spec.Template.Spec,
	}

	return p
}
