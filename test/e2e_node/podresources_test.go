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

	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	podresources "k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/util"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

func podresourcesGetAllocatableResourcesTests(f *framework.Framework, cli podresourcesapi.PodResourcesListerClient, sd *sriovData) {
	ginkgo.By("checking the devices known to the kubelet")
	resp, err := cli.GetAllocatableResources(context.TODO(), &podresourcesapi.AllocatableResourcesRequest{})
	framework.ExpectNoError(err)
	devs := resp.GetDevices()
	cpuIds := resp.GetCpuIds()

	onlineCpuIds, err := getOnlineCPUs()
	framework.ExpectNoError(err)
	framework.ExpectEqual(cpuIds, onlineCpuIds)

	if sd == nil { // no devices in the environment, so expect no devices
		ginkgo.By("expecting no devices reported")
		framework.ExpectEqual(len(devs), 0, fmt.Sprintf("got unexpected devices %#v", devs))
		return
	}

	ginkgo.By(fmt.Sprintf("expecting some %q devices reported", sd.resourceName))
	gomega.Expect(len(devs)).To(gomega.BeNumerically(">", 0))
	for _, dev := range devs {
		framework.ExpectEqual(dev.ResourceName, sd.resourceName)
		gomega.Expect(len(dev.DeviceIds)).To(gomega.BeNumerically(">", 0))
	}
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("POD Resources [Serial] [Feature:PODResources][NodeFeature:PODResources]", func() {
	f := framework.NewDefaultFramework("podresources-test")

	ginkgo.Context("With SRIOV devices in the system", func() {
		// this is a very rough check. We just want to rule out system that does NOT have any SRIOV device.
		if sriovdevCount, err := countSRIOVDevices(); false && (err != nil || sriovdevCount == 0) {
			e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
		}

		ginkgo.It("should return the expected device count", func() {
			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)
			defer teardownSRIOVConfigOrFail(f, sd)

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)
			cli, conn, err := podresources.GetClient(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			podresourcesGetAllocatableResourcesTests(f, cli, sd)
		})
	})

	ginkgo.Context("Without SRIOV devices in the system", func() {
		if sriovdevCount, err := countSRIOVDevices(); false && (err != nil || sriovdevCount > 0) {
			e2eskipper.Skipf("this test is meant to run on a system without any configured VF from SRIOV device")
		}

		ginkgo.It("should return empty devices", func() {
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)
			cli, conn, err := podresources.GetClient(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			podresourcesGetAllocatableResourcesTests(f, cli, nil)
		})
	})
})

func getOnlineCPUs() ([]int64, error) {
	onlineCPUList, err := ioutil.ReadFile("/sys/devices/system/cpu/online")
	if err != nil {
		return nil, err
	}
	cpus, err := cpuset.Parse(strings.TrimSpace(string(onlineCPUList)))
	if err != nil {
		return nil, err
	}
	cpuSlice := cpus.ToSlice()
	cpuIds := make([]int64, len(cpuSlice))
	for idx, cpuId := range cpuSlice {
		cpuIds[idx] = int64(cpuId)
	}
	return cpuIds, nil
}
