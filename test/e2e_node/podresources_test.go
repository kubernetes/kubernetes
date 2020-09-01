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
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podresources "k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	podresourcesapi "k8s.io/kubernetes/pkg/kubelet/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/util"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
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

func podActionToString(action podresourcesapi.WatchPodAction) string {
	switch action {
	case podresourcesapi.WatchPodAction_ADDED:
		return "added"
	case podresourcesapi.WatchPodAction_MODIFIED:
		return "modified"
	case podresourcesapi.WatchPodAction_DELETED:
		return "deleted"
	default:
		return "unknown"
	}
}

func scanPodResources(podIdx int, action podresourcesapi.WatchPodAction, pr *podresourcesapi.PodResources) map[string]int {
	resources := make(map[string]int)

	ns := pr.GetNamespace()
	cnts := pr.GetContainers()
	framework.Logf("#%02d %s (%s) - %d containers", podIdx, pr.GetName(), podActionToString(action), len(cnts))
	for cntIdx, cnt := range cnts {
		if len(cnt.Devices) > 0 {
			for devIdx, dev := range cnt.Devices {
				framework.Logf("#%02d/%02d/%02d - %s/%s/%s   %s -> %s", podIdx, cntIdx, devIdx, ns, pr.GetName(), cnt.Name, dev.ResourceName, strings.Join(dev.DeviceIds, ", "))
				resources[dev.ResourceName]++
			}
		} else {
			framework.Logf("#%02d/%02d/%02d - %s/%s/%s   No resources", podIdx, cntIdx, 0, ns, pr.GetName(), cnt.Name)
		}
	}
	return resources
}

func getPodResources(cli podresourcesapi.PodResourcesListerClient) ([]*podresourcesapi.PodResources, []*podresourcesapi.PodResources) {
	resp, err := cli.List(context.TODO(), &podresourcesapi.ListPodResourcesRequest{})
	framework.ExpectNoError(err)

	res := []*podresourcesapi.PodResources{}
	noRes := []*podresourcesapi.PodResources{}
	for idx, podResource := range resp.GetPodResources() {
		if len(scanPodResources(idx, podresourcesapi.WatchPodAction(-1), podResource)) > 0 {
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

func NewTestPodData() *testPodData {
	return &testPodData{
		PodMap: make(map[string]*v1.Pod),
	}
}

func (tpd *testPodData) createPodsForTest(f *framework.Framework, podReqs []podDesc) {
	for _, podReq := range podReqs {
		pod := makePodResourcesTestPod(podReq.podName, "cnt-0", podReq.resourceName, podReq.resourceAmount)

		framework.Logf("creating pod: %s", pod.Name)
		pod = f.PodClient().CreateSync(pod)
		framework.Logf("created pod: %s - %v", pod.Name, pod.Spec.Containers[0].Resources)

		tpd.PodMap[podReq.podName] = pod
	}
}

func (tpd *testPodData) deletePodsForTest(f *framework.Framework) {
	for podName := range tpd.PodMap {
		deletePodSync(f, podName)
	}
}

func (tpd *testPodData) deletePod(f *framework.Framework, podName string) {
	_, ok := tpd.PodMap[podName]
	if !ok {
		return
	}
	deletePodSync(f, podName)
	delete(tpd.PodMap, podName)
}

func podresourcesListTests(f *framework.Framework, cli podresourcesapi.PodResourcesListerClient, sd *sriovData) {
	var podResources []*podresourcesapi.PodResources
	var noResources []*podresourcesapi.PodResources
	var tpd *testPodData

	ginkgo.By("list: checking the output when no pods are present")
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 0)
	framework.ExpectEqual(len(noResources), 1) // the sriovdp

	tpd = NewTestPodData()
	ginkgo.By("list: checking the output when only pods which don't require resources are present")
	tpd.createPodsForTest(f, []podDesc{
		{
			podName: "pod-00",
		},
		{
			podName: "pod-01",
		},
	})
	podResources, noResources = getPodResources(cli)
	framework.ExpectEqual(len(podResources), 0)
	framework.ExpectEqual(len(noResources), 2+1) // test pods + sriovdp
	tpd.deletePodsForTest(f)

	tpd = NewTestPodData()
	ginkgo.By("list: checking the output when only a subset of pods require resources")
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
	tpd.deletePodsForTest(f)

	tpd = NewTestPodData()
	ginkgo.By("list: checking the output when creating pods which require resources between calls")
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

	tpd = NewTestPodData()
	ginkgo.By("list: checking the output when deleting pods which require resources between calls")
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

func podresourcesWatchTests(f *framework.Framework, cli podresourcesapi.PodResourcesListerClient, sd *sriovData) {
	var tpd *testPodData

	tpd = NewTestPodData()
	ginkgo.By("watch: checking the output when creating pods which require resources between calls")
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

	watchCli, err := cli.Watch(context.TODO(), &podresourcesapi.WatchPodResourcesRequest{})
	framework.ExpectNoError(err)

	tpd.createPodsForTest(f, []podDesc{
		{
			podName:        "pod-03",
			resourceName:   sd.resourceName,
			resourceAmount: "1",
		},
	})

	resp, err := watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_ADDED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-03")
	podRes := scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 0)

	resp, err = watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_MODIFIED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-03")
	podRes = scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 1)
	framework.ExpectEqual(podRes[sd.resourceName], 1)

	// TODO check for specific pods
	tpd.deletePodsForTest(f)

	tpd = NewTestPodData()
	ginkgo.By("watch: checking the output when deleting pods which require resources between calls")
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

	watchCli, err = cli.Watch(context.TODO(), &podresourcesapi.WatchPodResourcesRequest{})
	framework.ExpectNoError(err)

	tpd.deletePod(f, "pod-01")

	resp, err = watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_MODIFIED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-01")
	podRes = scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 1)
	framework.ExpectEqual(podRes[sd.resourceName], 1)

	resp, err = watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_DELETED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-01")
	podRes = scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 1)
	framework.ExpectEqual(podRes[sd.resourceName], 1)

	tpd.deletePod(f, "pod-02")

	resp, err = watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_MODIFIED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-02")
	podRes = scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 0)

	resp, err = watchCli.Recv()
	framework.ExpectNoError(err)
	framework.ExpectEqual(resp.Action, podresourcesapi.WatchPodAction_DELETED)
	framework.ExpectEqual(len(resp.PodResources), 1)
	framework.ExpectEqual(resp.PodResources[0].Name, "pod-02")
	podRes = scanPodResources(0, resp.Action, resp.PodResources[0])
	framework.ExpectEqual(len(podRes), 0)

	tpd.deletePodsForTest(f)

	// TODO: check we DO NOT receive unexpected events?
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("POD Resources [Serial] [Feature:PODResources][NodeFeature:PODResources]", func() {
	f := framework.NewDefaultFramework("podresources-test")

	ginkgo.Context("With SRIOV devices in the system", func() {
		// this is a very rough check. We just want to rule out system that does NOT have any SRIOV device.
		if sriovdevCount, err := countSRIOVDevices(); false && (err != nil || sriovdevCount == 0) {
			e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
		}

		ginkgo.It("should return the expected responses from endpoints", func() {
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)
			defer teardownSRIOVConfigOrFail(f, sd)

			cli, conn, err := podresources.GetClient(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			defer conn.Close()

			podresourcesListTests(f, cli, sd)
			podresourcesWatchTests(f, cli, sd)
		})
	})
})
