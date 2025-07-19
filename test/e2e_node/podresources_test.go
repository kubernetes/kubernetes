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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	"google.golang.org/grpc"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletdevicepluginv1beta1 "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	apisgrpc "k8s.io/kubernetes/pkg/kubelet/apis/grpc"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/util"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

const (
	defaultTopologyUnawareResourceName = "example.com/resource"
)

type podDesc struct {
	podName        string
	cntName        string
	resourceName   string
	resourceAmount int
	cpuRequest     int // cpuRequest is in millicores
	memRequest     int // memRequest is in megabytes
	cpuLimit       int // cpuLimit is in millicores
	memLimit       int // memLimit is in megabytes
	initContainers []initContainerDesc
	mainCntCommand []string
	restartPolicy  *v1.RestartPolicy
}

func (desc podDesc) CPURequestQty() resource.Quantity {
	qty := resource.NewMilliQuantity(int64(desc.cpuRequest), resource.DecimalSI)
	return *qty
}

func (desc podDesc) MemRequestQty() resource.Quantity {
	qty := resource.NewQuantity(int64(desc.memRequest)*1024*1024, resource.DecimalSI)
	return *qty
}

func (desc podDesc) CPULimitQty() resource.Quantity {
	qty := resource.NewMilliQuantity(int64(desc.cpuLimit), resource.DecimalSI)
	return *qty
}

func (desc podDesc) MemLimitQty() resource.Quantity {
	qty := resource.NewQuantity(int64(desc.memLimit)*1024*1024, resource.DecimalSI)
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

type initContainerDesc struct {
	cntName        string
	resourceName   string
	resourceAmount int
	cpuRequest     int // cpuRequest is in millicores
	restartPolicy  *v1.ContainerRestartPolicy
}

func (desc initContainerDesc) CPURequestQty() resource.Quantity {
	qty := resource.NewMilliQuantity(int64(desc.cpuRequest), resource.DecimalSI)
	return *qty
}

func (desc initContainerDesc) CPURequestExclusive() int {
	if (desc.cpuRequest % 1000) != 0 {
		// exclusive cpus are request only if the quantity is integral;
		// hence, explicitly rule out non-integral requests
		return 0
	}
	return desc.cpuRequest / 1000
}

func (desc initContainerDesc) RequiresCPU() bool {
	return desc.cpuRequest > 0
}

func (desc initContainerDesc) RequiresDevices() bool {
	return desc.resourceName != "" && desc.resourceAmount > 0
}

// TODO: replace makePodResourcesTestPod reimplementing in terms of this function
func makeFullPodResourcesTestPod(desc podDesc) *v1.Pod {
	cnt := v1.Container{
		Name:  desc.cntName,
		Image: busyboxImage,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{},
			Limits:   v1.ResourceList{},
		},
		Command: []string{"sh", "-c", "sleep 1d"},
	}
	if desc.cpuLimit > 0 {
		cnt.Resources.Limits[v1.ResourceCPU] = desc.CPULimitQty()
	}
	if desc.cpuRequest > 0 {
		cnt.Resources.Requests[v1.ResourceCPU] = desc.CPURequestQty()
	}
	if desc.memLimit > 0 {
		cnt.Resources.Limits[v1.ResourceMemory] = desc.MemLimitQty()
	}
	if desc.memRequest > 0 {
		cnt.Resources.Requests[v1.ResourceMemory] = desc.MemRequestQty()
	}
	if desc.RequiresDevices() {
		devQty := resource.NewQuantity(int64(desc.resourceAmount), resource.DecimalSI)
		cnt.Resources.Requests[v1.ResourceName(desc.resourceName)] = *devQty
		cnt.Resources.Limits[v1.ResourceName(desc.resourceName)] = *devQty
	}
	if len(desc.mainCntCommand) > 0 {
		cnt.Command = desc.mainCntCommand
	}
	restartPolicy := v1.RestartPolicyNever
	if desc.restartPolicy != nil {
		restartPolicy = *desc.restartPolicy
	}

	// TODO: support for init containers - currently unneeded
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: desc.podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: restartPolicy,
			Containers: []v1.Container{
				cnt,
			},
		},
	}
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
		cpuRequestQty := desc.CPURequestQty()
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
	if len(desc.mainCntCommand) > 0 {
		cnt.Command = desc.mainCntCommand
	}
	restartPolicy := v1.RestartPolicyNever
	if desc.restartPolicy != nil {
		restartPolicy = *desc.restartPolicy
	}

	var initCnts []v1.Container
	for _, cntDesc := range desc.initContainers {
		initCnt := v1.Container{
			Name:  cntDesc.cntName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{},
				Limits:   v1.ResourceList{},
			},
			Command:       []string{"sh", "-c", "sleep 5s"},
			RestartPolicy: cntDesc.restartPolicy,
		}
		if cntDesc.restartPolicy != nil && *cntDesc.restartPolicy == v1.ContainerRestartPolicyAlways {
			initCnt.Command = []string{"sh", "-c", "sleep 1d"}
		}
		if cntDesc.RequiresCPU() {
			cpuRequestQty := cntDesc.CPURequestQty()
			initCnt.Resources.Requests[v1.ResourceCPU] = cpuRequestQty
			initCnt.Resources.Limits[v1.ResourceCPU] = cpuRequestQty
			// we don't really care, we only need to be in guaranteed QoS
			initCnt.Resources.Requests[v1.ResourceMemory] = resource.MustParse("100Mi")
			initCnt.Resources.Limits[v1.ResourceMemory] = resource.MustParse("100Mi")
		}
		if cntDesc.RequiresDevices() {
			initCnt.Resources.Requests[v1.ResourceName(cntDesc.resourceName)] = resource.MustParse(fmt.Sprintf("%d", cntDesc.resourceAmount))
			initCnt.Resources.Limits[v1.ResourceName(cntDesc.resourceName)] = resource.MustParse(fmt.Sprintf("%d", cntDesc.resourceAmount))
		}
		initCnts = append(initCnts, initCnt)
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: desc.podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy:  restartPolicy,
			InitContainers: initCnts,
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

type podResMap map[string]map[string]*kubeletpodresourcesv1.ContainerResources

func convertToMap(podsResources []*kubeletpodresourcesv1.PodResources) podResMap {
	res := make(map[string]map[string]*kubeletpodresourcesv1.ContainerResources)
	for idx, podResource := range podsResources {
		// to make troubleshooting easier
		logPodResources(idx, podResource)

		cnts := make(map[string]*kubeletpodresourcesv1.ContainerResources)
		for _, cnt := range podResource.GetContainers() {
			cnts[cnt.GetName()] = cnt
		}
		res[podResource.GetName()] = cnts
	}
	return res
}

func getPodResourcesValues(ctx context.Context, cli kubeletpodresourcesv1.PodResourcesListerClient) (podResMap, error) {
	resp, err := cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}
	return convertToMap(resp.GetPodResources()), nil
}

type testPodData struct {
	PodMap map[string]*v1.Pod
}

func newTestPodData() *testPodData {
	return &testPodData{
		PodMap: make(map[string]*v1.Pod),
	}
}

func (tpd *testPodData) createPodsForTest(ctx context.Context, f *framework.Framework, podReqs []podDesc) {
	for _, podReq := range podReqs {
		pod := makePodResourcesTestPod(podReq)
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		framework.Logf("created pod %s", podReq.podName)
		tpd.PodMap[podReq.podName] = pod
	}
}

/* deletePodsForTest clean up all the pods run for a testcase. Must ensure proper cleanup */
func (tpd *testPodData) deletePodsForTest(ctx context.Context, f *framework.Framework) {
	deletePodsAsync(ctx, f, tpd.PodMap)
}

/* deletePod removes pod during a test. Should do a best-effort clean up */
func (tpd *testPodData) deletePod(ctx context.Context, f *framework.Framework, podName string) {
	_, ok := tpd.PodMap[podName]
	if !ok {
		return
	}
	deletePodSyncByName(ctx, f, podName)
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

func matchPodDescWithResourcesNamesOnly(expected []podDesc, found podResMap) error {
	framework.Logf("got %d pods expected %d", len(found), len(expected))
	if len(found) != len(expected) {
		return fmt.Errorf("found %d items expected %d", len(found), len(expected))
	}
	for _, podReq := range expected {
		framework.Logf("matching: %#v", podReq)

		if _, ok := found[podReq.podName]; !ok {
			return fmt.Errorf("no pod resources for pod %q", podReq.podName)
		}

	}
	return nil
}

func matchPodDescAndCountWithResources(expected []podDesc, found podResMap) error {
	framework.Logf("got %d pods expected %d", len(found), len(expected))
	if len(found) != len(expected) {
		return fmt.Errorf("found %d items expected %d", len(found), len(expected))
	}
	return matchPodDescWithResources(expected, found)
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
		if cnts, ok := found[defaultTopologyUnawareResourceName]; ok {
			for _, cnt := range cnts {
				for _, cd := range cnt.GetDevices() {
					if cd.ResourceName != defaultTopologyUnawareResourceName {
						continue
					}
					if cd.Topology != nil {
						//we expect nil topology
						return fmt.Errorf("Nil topology is expected")
					}
				}

			}
		}

		// check init containers
		for _, initCntDesc := range podReq.initContainers {
			if initCntDesc.restartPolicy == nil || *initCntDesc.restartPolicy != v1.ContainerRestartPolicyAlways {
				// If the init container is not restartable, we don't expect it
				// to be reported.
				_, ok := podInfo[initCntDesc.cntName]
				if ok {
					return fmt.Errorf("pod %q regular init container %q should not be reported", podReq.podName, initCntDesc.cntName)
				}
				continue
			}

			cntInfo, ok := podInfo[initCntDesc.cntName]
			if !ok {
				return fmt.Errorf("no container resources for pod %q container %q", podReq.podName, initCntDesc.cntName)
			}
			if initCntDesc.RequiresCPU() {
				if exclusiveCpus := initCntDesc.CPURequestExclusive(); exclusiveCpus != len(cntInfo.CpuIds) {
					if exclusiveCpus == 0 {
						return fmt.Errorf("pod %q container %q requested %d expected to be allocated CPUs from shared pool %v", podReq.podName, initCntDesc.cntName, initCntDesc.cpuRequest, cntInfo.CpuIds)
					}
					return fmt.Errorf("pod %q container %q expected %d cpus got %v", podReq.podName, initCntDesc.cntName, exclusiveCpus, cntInfo.CpuIds)
				}
			}
			if initCntDesc.RequiresDevices() {
				dev := findContainerDeviceByName(cntInfo.GetDevices(), initCntDesc.resourceName)
				if dev == nil {
					return fmt.Errorf("pod %q container %q expected data for resource %q not found", podReq.podName, initCntDesc.cntName, initCntDesc.resourceName)
				}
				if len(dev.DeviceIds) != initCntDesc.resourceAmount {
					return fmt.Errorf("pod %q container %q resource %q expected %d items got %v", podReq.podName, initCntDesc.cntName, initCntDesc.resourceName, initCntDesc.resourceAmount, dev.DeviceIds)
				}
			} else {
				devs := cntInfo.GetDevices()
				if len(devs) > 0 {
					return fmt.Errorf("pod %q container %q expected no resources, got %v", podReq.podName, initCntDesc.cntName, devs)
				}
			}
			if cnts, ok := found[defaultTopologyUnawareResourceName]; ok {
				for _, cnt := range cnts {
					for _, cd := range cnt.GetDevices() {
						if cd.ResourceName != defaultTopologyUnawareResourceName {
							continue
						}
						if cd.Topology != nil {
							// we expect nil topology
							return fmt.Errorf("Nil topology is expected")
						}
					}
				}
			}
		}
	}
	return nil
}

func expectPodResources(ctx context.Context, offset int, cli kubeletpodresourcesv1.PodResourcesListerClient, expected []podDesc) {
	gomega.EventuallyWithOffset(1+offset, ctx, func(ctx context.Context) error {
		found, err := getPodResourcesValues(ctx, cli)
		if err != nil {
			return err
		}
		return matchPodDescWithResources(expected, found)
	}, time.Minute, 10*time.Second).Should(gomega.Succeed())
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

func podresourcesListTests(ctx context.Context, f *framework.Framework, cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData, sidecarContainersEnabled bool) {
	var tpd *testPodData

	var found podResMap
	var expected []podDesc
	var extra podDesc

	expectedBasePods := 0 /* nothing but pods we create */
	if sd != nil {
		expectedBasePods = 1 // sriovdp
	}

	var err error
	ginkgo.By("checking the output when no pods are present")
	found, err = getPodResourcesValues(ctx, cli)
	framework.ExpectNoError(err, "getPodResourcesValues() failed err: %v", err)
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

	tpd.createPodsForTest(ctx, f, expected)
	expectPodResources(ctx, 1, cli, expected)
	tpd.deletePodsForTest(ctx, f)

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
				cpuRequest:     1000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 1000,
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
				cpuRequest: 1000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
			{
				podName:    "pod-03",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
		}

	}
	tpd.createPodsForTest(ctx, f, expected)
	expectPodResources(ctx, 1, cli, expected)
	tpd.deletePodsForTest(ctx, f)

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
				cpuRequest:     1000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 1000,
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
				cpuRequest: 1000,
			},
			{
				podName:    "pod-02",
				cntName:    "cnt-00",
				cpuRequest: 1000,
			},
		}
	}

	tpd.createPodsForTest(ctx, f, expected)
	expectPodResources(ctx, 1, cli, expected)

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

	tpd.createPodsForTest(ctx, f, []podDesc{
		extra,
	})

	expected = append(expected, extra)
	expectPodResources(ctx, 1, cli, expected)
	tpd.deletePodsForTest(ctx, f)

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
	tpd.createPodsForTest(ctx, f, expected)
	expectPodResources(ctx, 1, cli, expected)

	tpd.deletePod(ctx, f, "pod-01")
	expectedPostDelete := filterOutDesc(expected, "pod-01")
	expectPodResources(ctx, 1, cli, expectedPostDelete)
	tpd.deletePodsForTest(ctx, f)

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
	tpd.createPodsForTest(ctx, f, expected)
	expectPodResources(ctx, 1, cli, expected)
	tpd.deletePodsForTest(ctx, f)

	if sidecarContainersEnabled {
		containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

		tpd = newTestPodData()
		ginkgo.By("checking the output when pods have init containers")
		if sd != nil {
			expected = []podDesc{
				{
					podName:    "pod-00",
					cntName:    "regular-00",
					cpuRequest: 1000,
					initContainers: []initContainerDesc{
						{
							cntName:        "init-00",
							resourceName:   sd.resourceName,
							resourceAmount: 1,
							cpuRequest:     1000,
						},
					},
				},
				{
					podName:    "pod-01",
					cntName:    "regular-00",
					cpuRequest: 1000,
					initContainers: []initContainerDesc{
						{
							cntName:        "restartable-init-00",
							resourceName:   sd.resourceName,
							resourceAmount: 1,
							cpuRequest:     1000,
							restartPolicy:  &containerRestartPolicyAlways,
						},
					},
				},
			}
		} else {
			expected = []podDesc{
				{
					podName:    "pod-00",
					cntName:    "regular-00",
					cpuRequest: 1000,
					initContainers: []initContainerDesc{
						{
							cntName:    "init-00",
							cpuRequest: 1000,
						},
					},
				},
				{
					podName:    "pod-01",
					cntName:    "regular-00",
					cpuRequest: 1000,
					initContainers: []initContainerDesc{
						{
							cntName:       "restartable-init-00",
							cpuRequest:    1000,
							restartPolicy: &containerRestartPolicyAlways,
						},
					},
				},
			}
		}

		tpd.createPodsForTest(ctx, f, expected)
		expectPodResources(ctx, 1, cli, expected)
		tpd.deletePodsForTest(ctx, f)
	}
}

func podresourcesGetAllocatableResourcesTests(ctx context.Context, cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData, onlineCPUs, reservedSystemCPUs cpuset.CPUSet) {
	ginkgo.GinkgoHelper()

	ginkgo.By("checking the devices known to the kubelet")
	resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
	framework.ExpectNoError(err, "cannot get allocatable CPUs from podresources")
	allocatableCPUs, devs := demuxCPUsAndDevicesFromGetAllocatableResources(resp)

	if onlineCPUs.Size() == 0 {
		ginkgo.By("expecting no CPUs reported")
		gomega.Expect(onlineCPUs.Size()).To(gomega.Equal(reservedSystemCPUs.Size()), "with no online CPUs, no CPUs should be reserved")
	} else {
		ginkgo.By(fmt.Sprintf("expecting online CPUs reported - online=%v (%d) reserved=%v (%d)", onlineCPUs, onlineCPUs.Size(), reservedSystemCPUs, reservedSystemCPUs.Size()))
		if reservedSystemCPUs.Size() > onlineCPUs.Size() {
			ginkgo.Fail("more reserved CPUs than online")
		}
		expectedCPUs := onlineCPUs.Difference(reservedSystemCPUs)

		ginkgo.By(fmt.Sprintf("expecting CPUs '%v'='%v'", allocatableCPUs, expectedCPUs))
		gomega.Expect(allocatableCPUs.Equals(expectedCPUs)).To(gomega.BeTrueBecause("mismatch expecting CPUs"))
	}

	if sd == nil { // no devices in the environment, so expect no devices
		ginkgo.By("expecting no devices reported")
		gomega.Expect(devs).To(gomega.BeEmpty(), fmt.Sprintf("got unexpected devices %#v", devs))
		return
	}

	ginkgo.By(fmt.Sprintf("expecting some %q devices reported", sd.resourceName))
	gomega.Expect(devs).ToNot(gomega.BeEmpty())
	for _, dev := range devs {
		gomega.Expect(dev.ResourceName).To(gomega.Equal(sd.resourceName))
		gomega.Expect(dev.DeviceIds).ToNot(gomega.BeEmpty())
	}
}

func demuxCPUsAndDevicesFromGetAllocatableResources(resp *kubeletpodresourcesv1.AllocatableResourcesResponse) (cpuset.CPUSet, []*kubeletpodresourcesv1.ContainerDevices) {
	devs := resp.GetDevices()
	var cpus []int
	for _, cpuid := range resp.GetCpuIds() {
		cpus = append(cpus, int(cpuid))
	}
	return cpuset.New(cpus...), devs
}

func podresourcesGetTests(ctx context.Context, f *framework.Framework, cli kubeletpodresourcesv1.PodResourcesListerClient, sidecarContainersEnabled bool) {
	//var err error
	ginkgo.By("checking the output when no pods are present")
	expected := []podDesc{}
	resp, err := cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "test", PodNamespace: f.Namespace.Name})
	podResourceList := []*kubeletpodresourcesv1.PodResources{resp.GetPodResources()}
	gomega.Expect(err).To(gomega.HaveOccurred(), "pod not found")
	res := convertToMap(podResourceList)
	err = matchPodDescWithResources(expected, res)
	framework.ExpectNoError(err, "matchPodDescWithResources() failed err %v", err)

	tpd := newTestPodData()
	ginkgo.By("checking the output when only pods which don't require resources are present")
	expected = []podDesc{
		{
			podName: "pod-00",
			cntName: "cnt-00",
		},
	}
	tpd.createPodsForTest(ctx, f, expected)
	resp, err = cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "pod-00", PodNamespace: f.Namespace.Name})
	framework.ExpectNoError(err, "Get() call failed for pod %s/%s", f.Namespace.Name, "pod-00")
	podResourceList = []*kubeletpodresourcesv1.PodResources{resp.GetPodResources()}
	res = convertToMap(podResourceList)
	err = matchPodDescWithResources(expected, res)
	framework.ExpectNoError(err, "matchPodDescWithResources() failed err %v", err)
	tpd.deletePodsForTest(ctx, f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when only pod require CPU")
	expected = []podDesc{
		{
			podName:    "pod-01",
			cntName:    "cnt-00",
			cpuRequest: 1000,
		},
	}
	tpd.createPodsForTest(ctx, f, expected)
	resp, err = cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "pod-01", PodNamespace: f.Namespace.Name})
	framework.ExpectNoError(err, "Get() call failed for pod %s/%s", f.Namespace.Name, "pod-01")
	podResourceList = []*kubeletpodresourcesv1.PodResources{resp.GetPodResources()}
	res = convertToMap(podResourceList)
	err = matchPodDescWithResources(expected, res)
	framework.ExpectNoError(err, "matchPodDescWithResources() failed err %v", err)
	tpd.deletePodsForTest(ctx, f)

	if sidecarContainersEnabled {
		containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

		tpd = newTestPodData()
		ginkgo.By("checking the output when only pod with init containers require CPU")
		expected = []podDesc{
			{
				podName:    "pod-01",
				cntName:    "cnt-00",
				cpuRequest: 1000,
				initContainers: []initContainerDesc{
					{
						cntName:    "init-00",
						cpuRequest: 1000,
					},
					{
						cntName:       "restartable-init-01",
						cpuRequest:    1000,
						restartPolicy: &containerRestartPolicyAlways,
					},
				},
			},
		}
		tpd.createPodsForTest(ctx, f, expected)
		resp, err = cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "pod-01", PodNamespace: f.Namespace.Name})
		framework.ExpectNoError(err, "Get() call failed for pod %s/%s", f.Namespace.Name, "pod-01")
		podResourceList = []*kubeletpodresourcesv1.PodResources{resp.GetPodResources()}
		res = convertToMap(podResourceList)
		err = matchPodDescWithResources(expected, res)
		framework.ExpectNoError(err, "matchPodDescWithResources() failed err %v", err)
		tpd.deletePodsForTest(ctx, f)
	}
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("POD Resources API", framework.WithSerial(), feature.PodResourcesAPI, func() {
	f := framework.NewDefaultFramework("podresources-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var reservedSystemCPUs cpuset.CPUSet
	var memoryQuantity resource.Quantity
	var defaultKubeParams *memoryManagerKubeletParams

	ginkgo.BeforeEach(func() {
		reservedSystemCPUs = cpuset.New(1)
		memoryQuantity = resource.MustParse("1100Mi")
		defaultKubeParams = &memoryManagerKubeletParams{
			systemReservedMemory: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						resourceMemory: memoryQuantity,
					},
				},
			},
			systemReserved: map[string]string{resourceMemory: "500Mi"},
			kubeReserved:   map[string]string{resourceMemory: "500Mi"},
			evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
		}
	})

	ginkgo.Context("with SRIOV devices in the system", func() {
		ginkgo.BeforeEach(func() {
			requireSRIOVDevices()
		})

		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func(ctx context.Context) {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
				})

				ginkgo.It("should return the expected responses", func(ctx context.Context) {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err, "getOnlineCPUs() failed err: %v", err)

					configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
					sd := setupSRIOVConfigOrFail(ctx, f, configMap)
					ginkgo.DeferCleanup(teardownSRIOVConfigOrFail, f, sd)

					waitForSRIOVResources(ctx, f, sd)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
					defer conn.Close()

					waitForSRIOVResources(ctx, f, sd)

					ginkgo.By("checking List()")
					podresourcesListTests(ctx, f, cli, sd, false)
					ginkgo.By("checking GetAllocatableResources()")
					podresourcesGetAllocatableResourcesTests(ctx, cli, sd, onlineCPUs, reservedSystemCPUs)
				})

				framework.It("should return the expected responses", feature.SidecarContainers, func(ctx context.Context) {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err, "getOnlineCPUs() failed err: %v", err)

					configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
					sd := setupSRIOVConfigOrFail(ctx, f, configMap)
					ginkgo.DeferCleanup(teardownSRIOVConfigOrFail, f, sd)

					waitForSRIOVResources(ctx, f, sd)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
					defer framework.ExpectNoError(conn.Close())

					waitForSRIOVResources(ctx, f, sd)

					ginkgo.By("checking List()")
					podresourcesListTests(ctx, f, cli, sd, true)
					ginkgo.By("checking GetAllocatableResources()")
					podresourcesGetAllocatableResourcesTests(ctx, cli, sd, onlineCPUs, reservedSystemCPUs)
				})
			})
		})

		ginkgo.Context("with CPU manager None policy", func() {
			ginkgo.It("should return the expected responses", func(ctx context.Context) {
				// current default is "none" policy - no need to restart the kubelet

				requireSRIOVDevices()

				configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
				sd := setupSRIOVConfigOrFail(ctx, f, configMap)
				ginkgo.DeferCleanup(teardownSRIOVConfigOrFail, f, sd)

				waitForSRIOVResources(ctx, f, sd)

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
				defer conn.Close()

				waitForSRIOVResources(ctx, f, sd)

				// intentionally passing empty cpuset instead of onlineCPUs because with none policy
				// we should get no allocatable cpus - no exclusively allocatable CPUs, depends on policy static
				podresourcesGetAllocatableResourcesTests(ctx, cli, sd, cpuset.CPUSet{}, cpuset.CPUSet{})
			})
		})
	})

	framework.Context("without SRIOV devices in the system", framework.WithFlaky(), func() {
		ginkgo.BeforeEach(func() {
			requireLackOfSRIOVDevices()
		})

		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func(ctx context.Context) {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
					if initialConfig.FeatureGates == nil {
						initialConfig.FeatureGates = make(map[string]bool)
					}
					initialConfig.FeatureGates[string(kubefeatures.KubeletPodResourcesGet)] = true
				})

				ginkgo.Context("with KubeletPodResourcesGet feature gate enabled", func() {
					ginkgo.BeforeEach(func() {
						e2eskipper.SkipUnlessFeatureGateEnabled("KubeletPodResourcesGet")
					})

					ginkgo.It("should succeed when calling Get for a valid pod", func(ctx context.Context) {
						endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
						framework.ExpectNoError(err, "LocalEndpoint() faild err: %v", err)

						cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
						framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
						defer framework.ExpectNoError(conn.Close())

						ginkgo.By("checking Get succeeds when the feature gate is enabled")
						pd := podDesc{
							podName:    "fg-enabled-pod",
							cntName:    "fg-enabled-cnt",
							cpuRequest: 1000,
						}
						pod := makePodResourcesTestPod(pd)
						pod = e2epod.NewPodClient(f).Create(ctx, pod)
						defer e2epod.NewPodClient(f).DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
						err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Ready", 2*time.Minute, testutils.PodRunningReady)
						framework.ExpectNoError(err)

						res, err := cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{
							PodName:      pod.Name,
							PodNamespace: pod.Namespace,
						})

						framework.Logf("Get result: %v, err: %v", res, err)
						framework.ExpectNoError(err, "Expected Get to succeed with the feature gate enabled")
						gomega.Expect(res.PodResources.Name).To(gomega.Equal(pod.Name))
						gomega.Expect(res.PodResources.Containers).To(gomega.HaveLen(1), "expected one container")
						container := res.PodResources.Containers[0]
						gomega.Expect(container.Name).To(gomega.Equal(pd.cntName), "expected container name match")
						gomega.Expect(container.CpuIds).ToNot(gomega.BeEmpty(), "expected CPU IDs to be reported")
						gomega.Expect(container.CpuIds).To(gomega.HaveLen(pd.CpuRequestExclusive()), "expected one exclusive CPU")
						gomega.Expect(container.Devices).To(gomega.BeEmpty(), "expected no devices")
					})
				})

				ginkgo.It("should return the expected responses", func(ctx context.Context) {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err, "getOnlineCPUs() failed err: %v", err)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
					defer conn.Close()

					podresourcesListTests(ctx, f, cli, nil, false)
					podresourcesGetAllocatableResourcesTests(ctx, cli, nil, onlineCPUs, reservedSystemCPUs)
					podresourcesGetTests(ctx, f, cli, false)
				})

				framework.It("should return the expected responses", feature.SidecarContainers, func(ctx context.Context) {
					onlineCPUs, err := getOnlineCPUs()
					framework.ExpectNoError(err, "getOnlineCPUs() failed err: %v", err)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
					defer func() {
						framework.ExpectNoError(conn.Close())
					}()

					podresourcesListTests(ctx, f, cli, nil, true)
					podresourcesGetAllocatableResourcesTests(ctx, cli, nil, onlineCPUs, reservedSystemCPUs)
					podresourcesGetTests(ctx, f, cli, true)
				})
			})
		})

		ginkgo.Context("with CPU manager None policy", func() {
			ginkgo.It("should return the expected responses", func(ctx context.Context) {
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
				defer conn.Close()

				// intentionally passing empty cpuset instead of onlineCPUs because with none policy
				// we should get no allocatable cpus - no exclusively allocatable CPUs, depends on policy static
				podresourcesGetAllocatableResourcesTests(ctx, cli, nil, cpuset.CPUSet{}, cpuset.CPUSet{})
			})
		})

		ginkgo.Context("with disabled KubeletPodResourcesGet feature gate", func() {

			ginkgo.It("should return the expected error with the feature gate disabled", func(ctx context.Context) {
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err, "LocalEndpoint() faild err %v", err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err, "GetV1Client() failed err %v", err)
				defer conn.Close()

				ginkgo.By("checking Get fail if the feature gate is not enabled")
				getRes, err := cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "test", PodNamespace: f.Namespace.Name})
				framework.Logf("Get result: %v, err: %v", getRes, err)
				gomega.Expect(err).To(gomega.HaveOccurred(), "With feature gate disabled, the call must fail")
			})
		})
	})

	ginkgo.When("checking core resource managers assignments", func() {
		var podresConn *grpc.ClientConn
		var podMap map[string]*v1.Pod
		var cpuAlloc int64

		ginkgo.BeforeEach(func(ctx context.Context) {
			podMap = make(map[string]*v1.Pod)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if podresConn != nil {
				framework.ExpectNoError(podresConn.Close())
			}
			deletePodsAsync(ctx, f, podMap)
		})

		ginkgo.JustBeforeEach(func(ctx context.Context) {
			// this is a very rough check. We just want to rule out system that does NOT have enough resources
			_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
			if cpuAlloc < minCoreCount {
				e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
			}
		})

		ginkgo.When("listing with restricted list output enabled", func() {

			tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
				kubeParams := *defaultKubeParams
				kubeParams.policy = staticPolicy
				updateKubeletConfigWithMemoryManagerParams(initialConfig, &kubeParams)
				initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)
				initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 10 * time.Minute} // set it long enough it is practically disabled
				cpus := reservedSystemCPUs.String()
				framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
				initialConfig.ReservedSystemCPUs = cpus
			})

			ginkgo.It("should report only the running pods regardless of the QoS", func(ctx context.Context) {
				// TODO: compute automatically
				if cpuAlloc < 8 {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, 8)
				}

				restartNever := v1.RestartPolicyNever
				restartAlways := v1.RestartPolicyAlways
				descs := []podDesc{
					{
						cntName:        "e2e-test-cnt-gu-term-1",
						podName:        "e2e-test-pod-gu-term-1",
						cpuLimit:       1000,
						memLimit:       256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:        "e2e-test-cnt-gu-term-2",
						podName:        "e2e-test-pod-gu-term-2",
						cpuLimit:       200,
						memLimit:       256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-gu-run-1",
						podName:       "e2e-test-pod-gu-run-1",
						cpuLimit:      200,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-gu-run-2",
						podName:       "e2e-test-pod-gu-run-2",
						cpuLimit:      1000,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-bu-term-1",
						podName:        "e2e-test-pod-bu-term-1",
						cpuRequest:     400,
						memRequest:     256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-bu-run-1",
						podName:       "e2e-test-pod-bu-run-1",
						cpuRequest:    400,
						memRequest:    256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-be-term-1",
						podName:        "e2e-test-pod-be-term-1",
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-be-run-1",
						podName:       "e2e-test-pod-be-run-1",
						restartPolicy: &restartAlways,
					},
				}
				expected := []podDesc{
					{
						cntName:       "e2e-test-cnt-gu-run-1",
						podName:       "e2e-test-pod-gu-run-1",
						cpuLimit:      200,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-gu-run-2",
						podName:       "e2e-test-pod-gu-run-2",
						cpuLimit:      1000,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-bu-run-1",
						podName:       "e2e-test-pod-bu-run-1",
						cpuRequest:    400,
						memRequest:    256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-be-run-1",
						podName:       "e2e-test-pod-be-run-1",
						restartPolicy: &restartAlways,
					},
				}

				for _, desc := range descs {
					pod := makeFullPodResourcesTestPod(desc)
					if desc.restartPolicy == &restartAlways {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
						podMap[string(pod.UID)] = pod
					} else {
						pod = e2epod.NewPodClient(f).Create(ctx, pod)
						pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err)
						podMap[string(pod.UID)] = pod
						err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
						framework.ExpectNoError(err)
					}
				}

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					return matchPodDescAndCountWithResources(expected, found)
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
			})

			ginkgo.It("should report only the running pods among many", func(ctx context.Context) {
				// TODO: compute automatically
				if cpuAlloc < 8 {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, 8)
				}
				restartNever := v1.RestartPolicyNever
				restartAlways := v1.RestartPolicyAlways
				descs := []podDesc{
					{
						cntName:       "e2e-test-cnt-payload-gu-1",
						podName:       "e2e-test-pod-payload-gu-1",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-mixed-cpu-mem",
						podName:        "e2e-test-pod-mixed-cpu-mem",
						cpuRequest:     1000,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-2",
						podName:       "e2e-test-pod-payload-gu-2",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-3",
						podName:       "e2e-test-pod-payload-gu-3",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-mixed-mem",
						podName:        "e2e-test-pod-mixed-mem",
						cpuRequest:     1000,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					},
				}
				expected := []podDesc{
					{
						cntName:       "e2e-test-cnt-payload-gu-1",
						podName:       "e2e-test-pod-payload-gu-1",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-2",
						podName:       "e2e-test-pod-payload-gu-2",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-3",
						podName:       "e2e-test-pod-payload-gu-3",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
				}

				for _, desc := range descs {
					pod := makePodResourcesTestPod(desc)
					if desc.restartPolicy == &restartAlways {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
						podMap[string(pod.UID)] = pod
					} else {
						pod = e2epod.NewPodClient(f).Create(ctx, pod)
						pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err)
						podMap[string(pod.UID)] = pod
						err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
						framework.ExpectNoError(err)
					}
				}

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					return matchPodDescAndCountWithResources(expected, found)
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
			})

			ginkgo.DescribeTable("all pods terminated and requiring", func(ctx context.Context, cpuReqMilli, podCount int) {
				cpuReqTot := int64((cpuReqMilli * podCount) / 1000)
				ginkgo.By(fmt.Sprintf("needed: %d cores", cpuReqTot))
				if cpuAlloc < cpuReqTot {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, cpuReqTot)
				}

				restartNever := v1.RestartPolicyNever

				for idx := range podCount {
					pd := podDesc{
						cntName:        fmt.Sprintf("e2e-test-cnt-%d", idx),
						podName:        fmt.Sprintf("e2e-test-pod-%d", idx),
						cpuRequest:     cpuReqMilli,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					}
					pod := makePodResourcesTestPod(pd)
					pod = e2epod.NewPodClient(f).Create(ctx, pod)
					pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
					podMap[string(pod.UID)] = pod
					err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
					framework.ExpectNoError(err)
				}
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					if len(found) > 0 {
						return fmt.Errorf("returned unexpected pods: %v", found)
					}
					return nil
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())

			},
				ginkgo.Entry("cpu and mem single", context.TODO(), 1000, 1),
				ginkgo.Entry("cpu and mem multi", context.TODO(), 1000, 3),
				ginkgo.Entry("mem only single", context.TODO(), 200, 1),
				ginkgo.Entry("mem only multi", context.TODO(), 200, 3),
			)
		})

		ginkgo.When("listing with restricted list output disabled for backward compatible defaults", func() {

			tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
				kubeParams := *defaultKubeParams
				kubeParams.policy = staticPolicy
				updateKubeletConfigWithMemoryManagerParams(initialConfig, &kubeParams)
				initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)
				initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 10 * time.Minute} // set it long enough it is practically disabled
				cpus := reservedSystemCPUs.String()
				framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
				initialConfig.ReservedSystemCPUs = cpus
				initialConfig.FeatureGates["KubeletPodResourcesListUseActivePods"] = false
			})

			ginkgo.It("should report all the known pods regardless of the QoS", func(ctx context.Context) {
				// TODO: compute automatically
				if cpuAlloc < 8 {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, 8)
				}

				restartNever := v1.RestartPolicyNever
				restartAlways := v1.RestartPolicyAlways
				descs := []podDesc{
					{
						cntName:        "e2e-test-cnt-gu-term-1",
						podName:        "e2e-test-pod-gu-term-1",
						cpuLimit:       1000,
						memLimit:       256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:        "e2e-test-cnt-gu-term-2",
						podName:        "e2e-test-pod-gu-term-2",
						cpuLimit:       200,
						memLimit:       256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-gu-run-1",
						podName:       "e2e-test-pod-gu-run-1",
						cpuLimit:      200,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-gu-run-2",
						podName:       "e2e-test-pod-gu-run-2",
						cpuLimit:      1000,
						memLimit:      256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-bu-term-1",
						podName:        "e2e-test-pod-bu-term-1",
						cpuRequest:     400,
						memRequest:     256,
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-bu-run-1",
						podName:       "e2e-test-pod-bu-run-1",
						cpuRequest:    400,
						memRequest:    256,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-be-term-1",
						podName:        "e2e-test-pod-be-term-1",
						restartPolicy:  &restartNever,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
					},
					{
						cntName:       "e2e-test-cnt-be-run-1",
						podName:       "e2e-test-pod-be-run-1",
						restartPolicy: &restartAlways,
					},
				}
				expected := descs

				for _, desc := range descs {
					pod := makeFullPodResourcesTestPod(desc)
					if desc.restartPolicy == &restartAlways {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
						podMap[string(pod.UID)] = pod
					} else {
						pod = e2epod.NewPodClient(f).Create(ctx, pod)
						pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err)
						podMap[string(pod.UID)] = pod
						err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
						framework.ExpectNoError(err)
					}
				}

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					return matchPodDescAndCountWithResources(expected, found)
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
			})

			ginkgo.It("should report all the known pods", func(ctx context.Context) {
				// TODO: compute automatically
				if cpuAlloc < 8 {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, 8)
				}
				restartNever := v1.RestartPolicyNever
				restartAlways := v1.RestartPolicyAlways
				descs := []podDesc{
					{
						cntName:       "e2e-test-cnt-payload-gu-1",
						podName:       "e2e-test-pod-payload-gu-1",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-mixed-cpu-mem",
						podName:        "e2e-test-pod-mixed-cpu-mem",
						cpuRequest:     1000,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-2",
						podName:       "e2e-test-pod-payload-gu-2",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:       "e2e-test-cnt-payload-gu-3",
						podName:       "e2e-test-pod-payload-gu-3",
						cpuRequest:    1000,
						restartPolicy: &restartAlways,
					},
					{
						cntName:        "e2e-test-cnt-mixed-mem",
						podName:        "e2e-test-pod-mixed-mem",
						cpuRequest:     1000,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					},
				}
				expected := descs

				for _, desc := range descs {
					pod := makePodResourcesTestPod(desc)
					if desc.restartPolicy == &restartAlways {
						pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
						podMap[string(pod.UID)] = pod
					} else {
						pod = e2epod.NewPodClient(f).Create(ctx, pod)
						pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err)
						podMap[string(pod.UID)] = pod
						err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
						framework.ExpectNoError(err)
					}
				}

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					// all but the last pod will have inconsistent resource reporting.
					// See: https://github.com/kubernetes/kubernetes/issues/132020#issuecomment-2921478368
					// this is why we introduced the FeatureGate  flag in the first place
					return matchPodDescWithResourcesNamesOnly(expected, found)
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
			})

			ginkgo.DescribeTable("all pods terminated and requiring", func(ctx context.Context, cpuReqMilli, podCount int) {
				cpuReqTot := int64((cpuReqMilli * podCount) / 1000)
				ginkgo.By(fmt.Sprintf("needed: %d cores", cpuReqTot))
				if cpuAlloc < cpuReqTot {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU %d allocatable < %d", cpuAlloc, cpuReqTot)
				}

				restartNever := v1.RestartPolicyNever

				var expected []podDesc
				for idx := range podCount {
					pd := podDesc{
						cntName:        fmt.Sprintf("e2e-test-cnt-%d", idx),
						podName:        fmt.Sprintf("e2e-test-pod-%d", idx),
						cpuRequest:     cpuReqMilli,
						mainCntCommand: []string{"sh", "-c", "/bin/true"},
						restartPolicy:  &restartNever,
					}
					expected = append(expected, pd)
					pod := makePodResourcesTestPod(pd)
					pod = e2epod.NewPodClient(f).Create(ctx, pod)
					pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
					podMap[string(pod.UID)] = pod
					err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "Pod Succeeded", time.Minute*2, testutils.PodSucceeded)
					framework.ExpectNoError(err)
				}
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				var cli kubeletpodresourcesv1.PodResourcesListerClient
				cli, podresConn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)

				gomega.Consistently(func(ctx context.Context) error {
					found, err := getPodResourcesValues(ctx, cli)
					if err != nil {
						return err
					}
					// all but the last pod will have inconsistent resource reporting.
					// See: https://github.com/kubernetes/kubernetes/issues/132020#issuecomment-2921478368
					// this is why we introduced the FeatureGate in the first place
					return matchPodDescWithResourcesNamesOnly(expected, found)
				}).WithContext(ctx).WithTimeout(30 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
			},
				ginkgo.Entry("cpu and mem single", context.TODO(), 1000, 1),
				ginkgo.Entry("cpu and mem multi", context.TODO(), 1000, 3),
				ginkgo.Entry("mem only single", context.TODO(), 200, 1),
				ginkgo.Entry("mem only multi", context.TODO(), 200, 3),
			)
		})
	})

	ginkgo.Context("with a topology-unaware device plugin, which reports resources w/o hardware topology", func() {
		ginkgo.Context("with CPU manager Static policy", func() {
			ginkgo.BeforeEach(func(ctx context.Context) {
				// this is a very rough check. We just want to rule out system that does NOT have enough resources
				_, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)

				if cpuAlloc < minCoreCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < %d", minCoreCount)
				}
			})

			// empty context to apply kubelet config changes
			ginkgo.Context("", func() {
				tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
					// Set the CPU Manager policy to static.
					initialConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)

					// Set the CPU Manager reconcile period to 1 second.
					initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

					cpus := reservedSystemCPUs.String()
					framework.Logf("configurePodResourcesInKubelet: using reservedSystemCPUs=%q", cpus)
					initialConfig.ReservedSystemCPUs = cpus
				})

				ginkgo.It("should return proper podresources the same as before the restart of kubelet", func(ctx context.Context) {
					dpPod := setupSampleDevicePluginOrFail(ctx, f)
					ginkgo.DeferCleanup(teardownSampleDevicePluginOrFail, f, dpPod)

					waitForTopologyUnawareResources(ctx, f)

					endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
					framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

					cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
					framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
					defer conn.Close()

					ginkgo.By("checking List and resources topology unaware resource should be without topology")

					allocatableResponse, _ := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
					for _, dev := range allocatableResponse.GetDevices() {
						if dev.ResourceName != defaultTopologyUnawareResourceName {
							continue
						}
						gomega.Expect(dev.Topology).To(gomega.BeNil(), "Topology is expected to be empty for topology unaware resources")
					}

					desc := podDesc{
						podName:        "pod-01",
						cntName:        "cnt-01",
						resourceName:   defaultTopologyUnawareResourceName,
						resourceAmount: 1,
						cpuRequest:     1000,
					}

					tpd := newTestPodData()
					tpd.createPodsForTest(ctx, f, []podDesc{
						desc,
					})

					expectPodResources(ctx, 0, cli, []podDesc{desc})

					ginkgo.By("Restarting Kubelet")
					restartKubelet(ctx, true)

					// we need to wait for the node to be reported ready before we can safely query
					// the podresources endpoint again. Otherwise we will have false negatives.
					ginkgo.By("Wait for node to be ready")
					waitForTopologyUnawareResources(ctx, f)

					expectPodResources(ctx, 0, cli, []podDesc{desc})
					tpd.deletePodsForTest(ctx, f)
				})
			})
		})
	})

	f.Context("when querying /metrics", f.WithNodeConformance(), func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(kubefeatures.KubeletPodResourcesGet)] = true
		})
		ginkgo.BeforeEach(func(ctx context.Context) {
			// ensure APIs have been called at least once
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed err %v", err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed err %v", err)
			defer conn.Close()

			_, err = cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
			framework.ExpectNoError(err, "List() failed err %v", err)

			_, err = cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err, "GetAllocatableResources() failed err %v", err)

			desc := podDesc{
				podName: "pod-01",
				cntName: "cnt-01",
			}
			tpd := newTestPodData()
			tpd.createPodsForTest(ctx, f, []podDesc{
				desc,
			})
			expectPodResources(ctx, 1, cli, []podDesc{desc})

			expected := []podDesc{}
			resp, err := cli.Get(ctx, &kubeletpodresourcesv1.GetPodResourcesRequest{PodName: "pod-01", PodNamespace: f.Namespace.Name})
			framework.ExpectNoError(err, "Get() call failed for pod %s/%s", f.Namespace.Name, "pod-01")
			podResourceList := []*kubeletpodresourcesv1.PodResources{resp.GetPodResources()}
			res := convertToMap(podResourceList)
			err = matchPodDescWithResources(expected, res)
			framework.ExpectNoError(err, "matchPodDescWithResources() failed err %v", err)
			tpd.deletePodsForTest(ctx, f)
		})

		ginkgo.It("should report the values for the podresources metrics", func(ctx context.Context) {
			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the value of the podresources metrics")

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_pod_resources_endpoint_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSampleAtLeast(1),
				}),
				"kubelet_pod_resources_endpoint_requests_list": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSampleAtLeast(1),
				}),
				"kubelet_pod_resources_endpoint_requests_get_allocatable": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSampleAtLeast(1),
				}),
				"kubelet_pod_resources_endpoint_requests_get": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSampleAtLeast(1),
				}),
				// not checking errors: the calls don't have non-catastrophic (e.g. out of memory) error conditions yet.
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getPodResourcesMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getPodResourcesMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})
	})

	framework.Context("with the builtin rate limit values", framework.WithFlaky(), func() {
		ginkgo.It("should hit throttling when calling podresources List in a tight loop", func(ctx context.Context) {
			// ensure APIs have been called at least once
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed err %v", err)

			ginkgo.By("Connecting to the kubelet endpoint")
			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed err %v", err)
			defer conn.Close()

			tries := podresources.DefaultQPS * 2 // This should also be greater than DefaultBurstTokens
			errs := []error{}

			ginkgo.By(fmt.Sprintf("Issuing %d List() calls in a tight loop", tries))
			startTime := time.Now()
			for try := 0; try < tries; try++ {
				_, err = cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
				errs = append(errs, err)
			}
			elapsed := time.Since(startTime)

			ginkgo.By(fmt.Sprintf("Checking return codes for %d List() calls in %v", tries, elapsed))

			framework.ExpectNoError(errs[0], "the first List() call unexpectedly failed with %v", errs[0])
			// we would expect (burst) successes and then (tries-burst) errors on a clean test environment running with
			// enough CPU power. CI is usually harsher. So we relax constraints, expecting at least _a_ failure, while
			// we are likely to get much more. But we can't predict yet how more we should expect, so we prefer to relax
			// constraints than to risk flakes at this stage.
			errLimitExceededCount := 0
			for _, err := range errs[1:] {
				if errors.Is(err, apisgrpc.ErrorLimitExceeded) {
					errLimitExceededCount++
				}
			}
			gomega.Expect(errLimitExceededCount).ToNot(gomega.BeZero(), "never hit the rate limit trying %d calls in %v", tries, elapsed)

			framework.Logf("got %d/%d rate limit errors, at least one needed, the more the better", errLimitExceededCount, tries)

			// this is not needed for this test. We're done. But we need to play nice with *other* tests which may run just after,
			// and which need to query the API. If they run "too fast", they can still be throttled because the throttling period
			// is not exhausted yet, yielding false negatives, leading to flakes.
			// We can't reset the period for the rate limit, we just wait "long enough" to make sure we absorb the burst
			// and other queries are not rejected because happening to soon
			ginkgo.By("Cooling down to reset the podresources API rate limit")
			time.Sleep(5 * time.Second)
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

func setupSampleDevicePluginOrFail(ctx context.Context, f *framework.Framework) *v1.Pod {
	e2enode.WaitForNodeToBeReady(ctx, f.ClientSet, framework.TestContext.NodeName, 5*time.Minute)

	dp := getSampleDevicePluginPod(kubeletdevicepluginv1beta1.DevicePluginPath)
	dp.Spec.NodeName = framework.TestContext.NodeName

	ginkgo.By("Create the sample device plugin pod")

	dpPod := e2epod.NewPodClient(f).CreateSync(ctx, dp)

	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, dpPod.Namespace, dpPod.Name, "Ready", 120*time.Second, testutils.PodRunningReady)
	if err != nil {
		framework.Logf("Sample Device Pod %v took too long to enter running/ready: %v", dp.Name, err)
	}
	framework.ExpectNoError(err, "WaitForPodCondition() failed err: %v", err)

	return dpPod
}

func teardownSampleDevicePluginOrFail(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	gp := int64(0)
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}
	ginkgo.By(fmt.Sprintf("Delete sample device plugin pod %s/%s", pod.Namespace, pod.Name))
	err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, deleteOptions)

	framework.ExpectNoError(err, "Failed to delete Pod %v in Namespace %v", pod.Name, pod.Namespace)
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func waitForTopologyUnawareResources(ctx context.Context, f *framework.Framework) {
	ginkgo.By(fmt.Sprintf("Waiting for %q resources to become available on the local node", defaultTopologyUnawareResourceName))

	gomega.Eventually(ctx, func(ctx context.Context) bool {
		node := getLocalNode(ctx, f)
		resourceAmount := CountSampleDeviceAllocatable(node)
		return resourceAmount > 0
	}, 2*time.Minute, framework.Poll).Should(gomega.BeTrueBecause("expected %q resources to be available, got no resources", defaultTopologyUnawareResourceName))
}

func getPodResourcesMetrics(ctx context.Context) (e2emetrics.KubeletMetrics, error) {
	// we are running out of good names, so we need to be unnecessarily specific to avoid clashes
	ginkgo.By("getting Pod Resources metrics from the metrics API")
	return e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, nodeNameOrIP()+":10255", "/metrics")
}

func timelessSampleAtLeast(lower interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		// We already check Metric when matching the Id
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically(">=", lower),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}
