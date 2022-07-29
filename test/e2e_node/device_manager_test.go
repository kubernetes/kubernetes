/*
Copyright 2021 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/util"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	devicePluginDir = "/var/lib/kubelet/device-plugins"
	checkpointName  = "kubelet_internal_checkpoint"
)

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Device Manager  [Serial] [Feature:DeviceManager][NodeFeature:DeviceManager]", func() {
	checkpointFullPath := filepath.Join(devicePluginDir, checkpointName)
	f := framework.NewDefaultFramework("devicemanager-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With SRIOV devices in the system", func() {
		// this test wants to reproduce what happened in https://github.com/kubernetes/kubernetes/issues/102880
		ginkgo.It("should be able to recover V1 (aka pre-1.20) checkpoint data and reject pods before device re-registration", func() {
			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount == 0 {
				e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
			}

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)
			sd := setupSRIOVConfigOrFail(f, configMap)

			waitForSRIOVResources(f, sd)

			cntName := "gu-container"
			// we create and delete a pod to make sure the internal device manager state contains a pod allocation
			ginkgo.By(fmt.Sprintf("Successfully admit one guaranteed pod with 1 core, 1 %s device", sd.resourceName))
			var initCtnAttrs []tmCtnAttribute
			ctnAttrs := []tmCtnAttribute{
				{
					ctnName:       cntName,
					cpuRequest:    "1000m",
					cpuLimit:      "1000m",
					deviceName:    sd.resourceName,
					deviceRequest: "1",
					deviceLimit:   "1",
				},
			}

			podName := "gu-pod-rec-pre-1"
			framework.Logf("creating pod %s attrs %v", podName, ctnAttrs)
			pod := makeTopologyManagerTestPod(podName, ctnAttrs, initCtnAttrs)
			pod = f.PodClient().CreateSync(pod)

			// now we need to simulate a node drain, so we remove all the pods, including the sriov device plugin.

			ginkgo.By("deleting the pod")
			// note we delete right now because we know the current implementation of devicemanager will NOT
			// clean up on pod deletion. When this changes, the deletion needs to be done after the test is done.
			deletePodSyncByName(f, pod.Name)
			waitForAllContainerRemoval(pod.Name, pod.Namespace)

			ginkgo.By("teardown the sriov device plugin")
			// since we will NOT be recreating the plugin, we clean up everything now
			teardownSRIOVConfigOrFail(f, sd)

			ginkgo.By("stopping the kubelet")
			killKubelet("SIGSTOP")

			ginkgo.By("rewriting the kubelet checkpoint file as v1")
			err := rewriteCheckpointAsV1(devicePluginDir, checkpointName)
			// make sure we remove any leftovers
			defer os.Remove(checkpointFullPath)
			framework.ExpectNoError(err)

			// this mimics a kubelet restart after the upgrade
			// TODO: is SIGTERM (less brutal) good enough?
			ginkgo.By("killing the kubelet")
			killKubelet("SIGKILL")

			ginkgo.By("waiting for the kubelet to be ready again")
			// Wait for the Kubelet to be ready.
			gomega.Eventually(func() bool {
				nodes, err := e2enode.TotalReady(f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrue())

			// note we DO NOT start the sriov device plugin. This is intentional.
			// issue#102880 reproduces because of a race on startup caused by corrupted device manager
			// state which leads to v1.Node object not updated on apiserver.
			// So to hit the issue we need to receive the pod *before* the device plugin registers itself.
			// The simplest and safest way to reproduce is just avoid to run the device plugin again

			podName = "gu-pod-rec-post-2"
			framework.Logf("creating pod %s attrs %v", podName, ctnAttrs)
			pod = makeTopologyManagerTestPod(podName, ctnAttrs, initCtnAttrs)

			pod = f.PodClient().Create(pod)
			err = e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
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

			framework.Logf("checking pod %s status reason (%s)", pod.Name, pod.Status.Reason)
			if !isUnexpectedAdmissionError(pod) {
				framework.Failf("pod %s failed for wrong reason: %q", pod.Name, pod.Status.Reason)
			}

			deletePodSyncByName(f, pod.Name)
		})

		ginkgo.It("should be able to recover V1 (aka pre-1.20) checkpoint data and update topology info on device re-registration", func() {
			if sriovdevCount, err := countSRIOVDevices(); err != nil || sriovdevCount == 0 {
				e2eskipper.Skipf("this test is meant to run on a system with at least one configured VF from SRIOV device")
			}

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			configMap := getSRIOVDevicePluginConfigMap(framework.TestContext.SriovdpConfigMapFile)

			sd := setupSRIOVConfigOrFail(f, configMap)
			waitForSRIOVResources(f, sd)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)

			resp, err := cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			conn.Close()
			framework.ExpectNoError(err)

			suitableDevs := 0
			for _, dev := range resp.GetDevices() {
				for _, node := range dev.GetTopology().GetNodes() {
					if node.GetID() != 0 {
						suitableDevs++
					}
				}
			}
			if suitableDevs == 0 {
				teardownSRIOVConfigOrFail(f, sd)
				e2eskipper.Skipf("no devices found on NUMA Cell other than 0")
			}

			cntName := "gu-container"
			// we create and delete a pod to make sure the internal device manager state contains a pod allocation
			ginkgo.By(fmt.Sprintf("Successfully admit one guaranteed pod with 1 core, 1 %s device", sd.resourceName))
			var initCtnAttrs []tmCtnAttribute
			ctnAttrs := []tmCtnAttribute{
				{
					ctnName:       cntName,
					cpuRequest:    "1000m",
					cpuLimit:      "1000m",
					deviceName:    sd.resourceName,
					deviceRequest: "1",
					deviceLimit:   "1",
				},
			}

			podName := "gu-pod-rec-pre-1"
			framework.Logf("creating pod %s attrs %v", podName, ctnAttrs)
			pod := makeTopologyManagerTestPod(podName, ctnAttrs, initCtnAttrs)
			pod = f.PodClient().CreateSync(pod)

			// now we need to simulate a node drain, so we remove all the pods, including the sriov device plugin.

			ginkgo.By("deleting the pod")
			// note we delete right now because we know the current implementation of devicemanager will NOT
			// clean up on pod deletion. When this changes, the deletion needs to be done after the test is done.
			deletePodSyncByName(f, pod.Name)
			waitForAllContainerRemoval(pod.Name, pod.Namespace)

			ginkgo.By("teardown the sriov device plugin")
			// no need to delete the config now (speed up later)
			deleteSRIOVPodOrFail(f, sd)

			ginkgo.By("stopping the kubelet")
			killKubelet("SIGSTOP")

			ginkgo.By("rewriting the kubelet checkpoint file as v1")
			err = rewriteCheckpointAsV1(devicePluginDir, checkpointName)
			// make sure we remove any leftovers
			defer os.Remove(checkpointFullPath)
			framework.ExpectNoError(err)

			// this mimics a kubelet restart after the upgrade
			// TODO: is SIGTERM (less brutal) good enough?
			ginkgo.By("killing the kubelet")
			killKubelet("SIGKILL")

			ginkgo.By("waiting for the kubelet to be ready again")
			// Wait for the Kubelet to be ready.
			gomega.Eventually(func() bool {
				nodes, err := e2enode.TotalReady(f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrue())

			sd2 := &sriovData{
				configMap:      sd.configMap,
				serviceAccount: sd.serviceAccount,
			}
			sd2.pod = createSRIOVPodOrFail(f)
			defer teardownSRIOVConfigOrFail(f, sd2)
			waitForSRIOVResources(f, sd2)

			compareSRIOVResources(sd, sd2)

			cli, conn, err = podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close()

			resp2, err := cli.GetAllocatableResources(context.TODO(), &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err)

			cntDevs := stringifyContainerDevices(resp.GetDevices())
			cntDevs2 := stringifyContainerDevices(resp2.GetDevices())
			if cntDevs != cntDevs2 {
				framework.Failf("different allocatable resources expected %v got %v", cntDevs, cntDevs2)
			}
		})

	})
})

func compareSRIOVResources(expected, got *sriovData) {
	if expected.resourceName != got.resourceName {
		framework.Failf("different SRIOV resource name: expected %q got %q", expected.resourceName, got.resourceName)
	}
	if expected.resourceAmount != got.resourceAmount {
		framework.Failf("different SRIOV resource amount: expected %d got %d", expected.resourceAmount, got.resourceAmount)
	}
}

func isUnexpectedAdmissionError(pod *v1.Pod) bool {
	re := regexp.MustCompile(`Unexpected.*Admission.*Error`)
	return re.MatchString(pod.Status.Reason)
}

func rewriteCheckpointAsV1(dir, name string) error {
	ginkgo.By(fmt.Sprintf("Creating temporary checkpoint manager (dir=%q)", dir))
	checkpointManager, err := checkpointmanager.NewCheckpointManager(dir)
	if err != nil {
		return err
	}
	cp := checkpoint.New(make([]checkpoint.PodDevicesEntry, 0), make(map[string][]string))
	err = checkpointManager.GetCheckpoint(name, cp)
	if err != nil {
		return err
	}

	ginkgo.By(fmt.Sprintf("Read checkpoint %q %#v", name, cp))

	podDevices, registeredDevs := cp.GetDataInLatestFormat()
	podDevicesV1 := convertPodDeviceEntriesToV1(podDevices)
	cpV1 := checkpoint.NewV1(podDevicesV1, registeredDevs)

	blob, err := cpV1.MarshalCheckpoint()
	if err != nil {
		return err
	}

	// TODO: why `checkpointManager.CreateCheckpoint(name, cpV1)` doesn't seem to work?
	ckPath := filepath.Join(dir, name)
	os.WriteFile(filepath.Join("/tmp", name), blob, 0600)
	return os.WriteFile(ckPath, blob, 0600)
}

func convertPodDeviceEntriesToV1(entries []checkpoint.PodDevicesEntry) []checkpoint.PodDevicesEntryV1 {
	entriesv1 := []checkpoint.PodDevicesEntryV1{}
	for _, entry := range entries {
		deviceIDs := []string{}
		for _, perNUMANodeDevIDs := range entry.DeviceIDs {
			deviceIDs = append(deviceIDs, perNUMANodeDevIDs...)
		}
		entriesv1 = append(entriesv1, checkpoint.PodDevicesEntryV1{
			PodUID:        entry.PodUID,
			ContainerName: entry.ContainerName,
			ResourceName:  entry.ResourceName,
			DeviceIDs:     deviceIDs,
			AllocResp:     entry.AllocResp,
		})
	}
	return entriesv1
}

func stringifyContainerDevices(devs []*kubeletpodresourcesv1.ContainerDevices) string {
	entries := []string{}
	for _, dev := range devs {
		devIDs := dev.GetDeviceIds()
		if devIDs != nil {
			for _, devID := range dev.DeviceIds {
				nodes := dev.GetTopology().GetNodes()
				if nodes != nil {
					for _, node := range nodes {
						entries = append(entries, fmt.Sprintf("%s[%s]@NUMA=%d", dev.ResourceName, devID, node.GetID()))
					}
				} else {
					entries = append(entries, fmt.Sprintf("%s[%s]@NUMA=none", dev.ResourceName, devID))
				}
			}
		} else {
			entries = append(entries, dev.ResourceName)
		}
	}
	sort.Strings(entries)
	return strings.Join(entries, ", ")
}
