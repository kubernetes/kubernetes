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

func areGPUsAvailableOnAnySchedulableNodes(f *framework.Framework) bool {
	framework.Logf("Getting list of Nodes from API server")
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable {
			continue
		}
		framework.Logf("gpuResourceName %s", gpuResourceName)
		if val, ok := node.Status.Capacity[gpuResourceName]; ok && val.Value() > 0 {
			framework.Logf("Nvidia GPUs available on Node: %q", node.Name)
			return true
		}
	}
	framework.Logf("Nvidia GPUs don't exist on all schedulable nodes")
	return false
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
		dsYamlUrl = framework.GPUDevicePluginDSYAML
		gpuResourceName = framework.NVIDIAGPUResourceName
		podCreationFunc = makeCudaAdditionDevicePluginTestPod
	} else {
		dsYamlUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/master/cos-nvidia-gpu-installer/daemonset.yaml"
		gpuResourceName = v1.ResourceNvidiaGPU
		podCreationFunc = makeCudaAdditionTestPod
	}

	// GPU drivers might have already been installed.
	if !areGPUsAvailableOnAllSchedulableNodes(f) {
		// Install Nvidia Drivers.
		ds, err := framework.DsFromManifest(dsYamlUrl)
		Expect(err).NotTo(HaveOccurred())
		ds.Namespace = f.Namespace.Name
		_, err = f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).Create(ds)
		framework.ExpectNoError(err, "failed to create daemonset")
		framework.Logf("Successfully created daemonset to install Nvidia drivers. Waiting for drivers to be installed and GPUs to be available in Node Capacity...")
		// Wait for Nvidia GPUs to be available on nodes
		Eventually(func() bool {
			return areGPUsAvailableOnAllSchedulableNodes(f)
		}, driverInstallTimeout, time.Second).Should(BeTrue())
	}
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
		// 1. Verifies GPU resource is successfully advertised on the nodes
		// and we can run pods using GPUs.
		By("Starting device plugin daemonset and running GPU pods")
		testNvidiaGPUsOnCOS(f)

		// 2. Verifies that when the device plugin DaemonSet is removed, resource capacity drops to zero.
		By("Deleting device plugin daemonset")
		ds, err := framework.DsFromManifest(dsYamlUrl)
		Expect(err).NotTo(HaveOccurred())
		falseVar := false
		err = f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).Delete(ds.Name, &metav1.DeleteOptions{OrphanDependents: &falseVar})
		framework.ExpectNoError(err, "failed to delete daemonset")
		framework.Logf("Successfully deleted device plugin daemonset. Wait for resource to be removed.")
		// Wait for Nvidia GPUs to be unavailable on all nodes.
		Eventually(func() bool {
			return !areGPUsAvailableOnAnySchedulableNodes(f)
		}, 10*time.Minute, time.Second).Should(BeTrue())

		// 3. Restarts the device plugin DaemonSet. Verifies GPU resource is successfully advertised
		// on the nodes and we can run pods using GPUs.
		By("Restarting device plugin daemonset and running GPU pods")
		testNvidiaGPUsOnCOS(f)
	})
})
