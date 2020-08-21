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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/util"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

func makePodResourcesTestPod(podName, cntName, devName, devCount string) *v1.Pod {
	cnt := v1.Container{
		Name:  cntName,
		Image: busyboxImage,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{},
			Limits:   v1.ResourceList{},
		},
		Command: []string{"sh", "-c", "sleep 1d"},
	}
	if devName != "" && devCount != "" {
		cnt.Resources.Requests[v1.ResourceName(devName)] = resource.MustParse(devCount)
		cnt.Resources.Limits[v1.ResourceName(devName)] = resource.MustParse(devCount)
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				cnt,
			},
		},
	}
}

func countPodResources(podIdx int, pr *kubeletpodresourcesv1.PodResources) int {
	ns := pr.GetNamespace()
	devCount := 0
	for cntIdx, cnt := range pr.GetContainers() {
		if len(cnt.Devices) > 0 {
			for devIdx, dev := range cnt.Devices {
				framework.Logf("#%02d/%02d/%02d - %s/%s/%s   %s -> %s", podIdx, cntIdx, devIdx, ns, pr.GetName(), cnt.Name, dev.ResourceName, strings.Join(dev.DeviceIds, ", "))
				devCount++
			}
		} else {
			framework.Logf("#%02d/%02d/%02d - %s/%s/%s   No resources", podIdx, cntIdx, 0, ns, pr.GetName(), cnt.Name)
		}
	}
	return devCount
}

func getPodResources(cli kubeletpodresourcesv1.PodResourcesListerClient) ([]*kubeletpodresourcesv1.PodResources, []*kubeletpodresourcesv1.PodResources) {
	resp, err := cli.List(context.TODO(), &kubeletpodresourcesv1.ListPodResourcesRequest{})
	framework.ExpectNoError(err)

	res := []*kubeletpodresourcesv1.PodResources{}
	noRes := []*kubeletpodresourcesv1.PodResources{}
	for idx, podResource := range resp.GetPodResources() {
		if countPodResources(idx, podResource) > 0 {
			res = append(res, podResource)
		} else {
			noRes = append(noRes, podResource)
		}
	}
	return res, noRes
}

type podDesc struct {
	podName        string
	resourceName   string
	resourceAmount string
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
		pod := makePodResourcesTestPod(podReq.podName, "cnt-0", podReq.resourceName, podReq.resourceAmount)
		pod = f.PodClient().CreateSync(pod)

		framework.Logf("created pod %s", podReq.podName)
		tpd.PodMap[podReq.podName] = pod
	}
}

func (tpd *testPodData) deletePodsForTest(f *framework.Framework) {
	for podName := range tpd.PodMap {
		deletePodSyncByName(f, podName)
	}
}

func (tpd *testPodData) deletePod(f *framework.Framework, podName string) {
	_, ok := tpd.PodMap[podName]
	if !ok {
		return
	}
	deletePodSyncByName(f, podName)
	delete(tpd.PodMap, podName)
}

func expectPodResources(cli kubeletpodresourcesv1.PodResourcesListerClient, expectedPodsWithResources, expectedPodsWithoutResources int) {
	gomega.EventuallyWithOffset(1, func() error {
		podResources, noResources := getPodResources(cli)
		if len(podResources) != expectedPodsWithResources {
			return fmt.Errorf("pod with resources: expected %d found %d", expectedPodsWithResources, len(podResources))
		}
		if len(noResources) != expectedPodsWithoutResources {
			return fmt.Errorf("pod WITHOUT resources: expected %d found %d", expectedPodsWithoutResources, len(noResources))
		}
		return nil
	}, time.Minute, 10*time.Second).Should(gomega.BeNil())
}

func podresourcesListTests(f *framework.Framework, cli kubeletpodresourcesv1.PodResourcesListerClient, sd *sriovData) {
	var podResources []*kubeletpodresourcesv1.PodResources
	var noResources []*kubeletpodresourcesv1.PodResources
	var tpd *testPodData

	ginkgo.By("checking the output when no pods are present")
	expectPodResources(cli, 0, 1) // sriovdp

	tpd = newTestPodData()
	ginkgo.By("checking the output when only pods which don't require resources are present")
	tpd.createPodsForTest(f, []podDesc{
		{
			podName: "pod-00",
		},
		{
			podName: "pod-01",
		},
	})
	expectPodResources(cli, 0, 2+1) // test pods + sriovdp
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when only a subset of pods require resources")
	tpd.createPodsForTest(f, []podDesc{
		{
			podName: "pod-00",
		},
		{
			podName:        "pod-01",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
		{
			podName: "pod-02",
		},
		{
			podName:        "pod-03",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
	})
	expectPodResources(cli, 2, 2+1) // test pods + sriovdp
	// TODO check for specific pods
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when creating pods which require resources between calls")
	tpd.createPodsForTest(f, []podDesc{
		{
			podName: "pod-00",
		},
		{
			podName:        "pod-01",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
		{
			podName: "pod-02",
		},
	})
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 1)
	framework.ExpectEqual(len(noResources), 2+1) // test pods + sriovdp
	// TODO check for specific pods

	tpd.createPodsForTest(f, []podDesc{
		{
			podName:        "pod-03",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
	})
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 2)
	framework.ExpectEqual(len(noResources), 2+1) // test pods + sriovdp
	// TODO check for specific pods
	tpd.deletePodsForTest(f)

	tpd = newTestPodData()
	ginkgo.By("checking the output when deleting pods which require resources between calls")
	tpd.createPodsForTest(f, []podDesc{
		{
			podName: "pod-00",
		},
		{
			podName:        "pod-01",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
		{
			podName: "pod-02",
		},
		{
			podName:        "pod-03",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
	})
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 2)
	framework.ExpectEqual(len(noResources), 2+1) // test pods + sriovdp
	// TODO check for specific pods

	tpd.deletePod(f, "pod-01")
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 1)
	framework.ExpectEqual(len(noResources), 2+1) // test pods + sriovdp
	// TODO check for specific pods
	tpd.deletePodsForTest(f)
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("POD Resources [Serial] [Feature:PODResources][NodeFeature:PODResources]", func() {
	f := framework.NewDefaultFramework("podresources-test")

	ginkgo.Context("With SRIOV devices in the system", func() {
		ginkgo.It("should return the expected responses from List()", func() {
			// this is a very rough check. We just want to rule out system that does NOT have any SRIOV device.
			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount == 0 {
				e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
			}

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)
			defer teardownSRIOVConfigOrFail(f, sd)

			waitForSRIOVResources(f, sd)

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			defer conn.Close()

			podresourcesListTests(f, cli, sd)
		})
	})
})
