/*
Copyright 2017 The Kubernetes Authors.

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

package scheduling

import (
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	testPodNamePrefix = "nvidia-gpu-"
	cosOSImage        = "Container-Optimized OS from Google"
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
)

type podCreationFuncType func() *v1.Pod

var (
	gpuResourceName v1.ResourceName
	dsYamlUrl       string
	podCreationFunc podCreationFuncType
)

func makeCudaAdditionTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nvidia-libraries",
							MountPath: "/usr/local/nvidia/lib64",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "nvidia-libraries",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/home/kubernetes/bin/nvidia/lib",
						},
					},
				},
			},
		},
	}
	return testPod
}

func makeCudaAdditionDevicePluginTestPod() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "vector-addition",
					Image: imageutils.GetE2EImage(imageutils.CudaVectorAdd),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpuResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
	return testPod
}

func isClusterRunningCOS(f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if !strings.Contains(node.Status.NodeInfo.OSImage, cosOSImage) {
			return false
		}
	}
	return true
}

func areGPUsAvailableOnAllSchedulableNodes(f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable {
			continue
		}
		framework.Logf("gpuResourceName %s", gpuResourceName)
		if val, ok := node.Status.Capacity[gpuResourceName]; !ok || val.Value() == 0 {
			framework.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false
		}
	}
	framework.Logf("Nvidia GPUs exist on all schedulable nodes")
	return true
}

func getGPUsAvailable(f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	var gpusAvailable int64
	for _, node := range nodeList.Items {
		if val, ok := node.Status.Capacity[gpuResourceName]; ok {
			gpusAvailable += (&val).Value()
		}
	}
	return gpusAvailable
}

func testNvidiaGPUsOnCOS(f *framework.Framework) {
	// Skip the test if the base image is not COS.
	// TODO: Add support for other base images.
	// CUDA apps require host mounts which is not portable across base images (yet).
	framework.Logf("Checking base image")
	if !isClusterRunningCOS(f) {
		Skip("Nvidia GPU tests are supproted only on Container Optimized OS image currently")
	}
	framework.Logf("Cluster is running on COS. Proceeding with test")

	if f.BaseName == "device-plugin-gpus" {
		dsYamlUrl = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml"
		gpuResourceName = framework.NVIDIAGPUResourceName
		podCreationFunc = makeCudaAdditionDevicePluginTestPod
	} else {
		dsYamlUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/master/cos-nvidia-gpu-installer/daemonset.yaml"
		gpuResourceName = v1.ResourceNvidiaGPU
		podCreationFunc = makeCudaAdditionTestPod
	}

	// Creates the DaemonSet that installs Nvidia Drivers.
	// The DaemonSet also runs nvidia device plugin for device plugin test.
	ds, err := framework.DsFromManifest(dsYamlUrl)
	Expect(err).NotTo(HaveOccurred())
	ds.Namespace = f.Namespace.Name
	_, err = f.ClientSet.ExtensionsV1beta1().DaemonSets(f.Namespace.Name).Create(ds)
	framework.ExpectNoError(err, "failed to create daemonset")
	framework.Logf("Successfully created daemonset to install Nvidia drivers.")

	pods, err := framework.WaitForControlledPods(f.ClientSet, ds.Namespace, ds.Name, extensionsinternal.Kind("DaemonSet"))
	framework.ExpectNoError(err, "getting pods controlled by the daemonset")
	devicepluginPods, err := framework.WaitForControlledPods(f.ClientSet, "kube-system", "nvidia-gpu-device-plugin", extensionsinternal.Kind("DaemonSet"))
	if err == nil {
		framework.Logf("Adding deviceplugin addon pod.")
		pods.Items = append(pods.Items, devicepluginPods.Items...)
	}
	framework.Logf("Starting ResourceUsageGather for the created DaemonSet pods.")
	rsgather, err := framework.NewResourceUsageGatherer(f.ClientSet, framework.ResourceGathererOptions{false, false, 2 * time.Second, 2 * time.Second, true}, pods)
	framework.ExpectNoError(err, "creating ResourceUsageGather for the daemonset pods")
	go rsgather.StartGatheringData()

	// Wait for Nvidia GPUs to be available on nodes
	framework.Logf("Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
	Eventually(func() bool {
		return areGPUsAvailableOnAllSchedulableNodes(f)
	}, driverInstallTimeout, time.Second).Should(BeTrue())

	framework.Logf("Creating as many pods as there are Nvidia GPUs and have the pods run a CUDA app")
	podList := []*v1.Pod{}
	for i := int64(0); i < getGPUsAvailable(f); i++ {
		podList = append(podList, f.PodClient().Create(podCreationFunc()))
	}
	framework.Logf("Wait for all test pods to succeed")
	// Wait for all pods to succeed
	for _, po := range podList {
		f.PodClient().WaitForSuccess(po.Name, 5*time.Minute)
	}

	framework.Logf("Stopping ResourceUsageGather")
	constraints := make(map[string]framework.ResourceConstraint)
	// For now, just gets summary. Can pass valid constraints in the future.
	summary, err := rsgather.StopAndSummarize([]int{50, 90, 100}, constraints)
	f.TestSummaries = append(f.TestSummaries, summary)
	framework.ExpectNoError(err, "getting resource usage summary")
}

var _ = SIGDescribe("[Feature:GPU]", func() {
	f := framework.NewDefaultFramework("gpus")
	It("run Nvidia GPU tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})

var _ = SIGDescribe("[Feature:GPUDevicePlugin]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus")
	It("run Nvidia GPU Device Plugin tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})
