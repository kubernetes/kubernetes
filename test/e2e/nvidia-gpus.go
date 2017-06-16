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

package e2e

import (
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	testPodNamePrefix = "nvidia-gpu-"
	testCUDAImage     = "gcr.io/google_containers/cuda-vector-add:v0.1"
	cosOSImage        = "Container-Optimized OS from Google"
	// Nvidia driver installation can take upwards of 5 minutes.
	driverInstallTimeout = 10 * time.Minute
	// Nvidia COS driver installer daemonset.
	cosNvidiaDriverInstallerUrl = "https://raw.githubusercontent.com/ContainerEngine/accelerators/stable/cos-nvidia-gpu-installer/daemonset.yaml"
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
					Image: testCUDAImage,
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceNvidiaGPU: *resource.NewQuantity(1, resource.DecimalSI),
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

func isClusterRunningCOS(f *framework.Framework) bool {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
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
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Spec.Unschedulable {
			continue
		}
		if node.Status.Capacity.NvidiaGPU().Value() == 0 {
			framework.Logf("Nvidia GPUs not available on Node: %q", node.Name)
			return false
		}
	}
	framework.Logf("Nvidia GPUs exist on all schedulable nodes")
	return true
}

func getGPUsAvailable(f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	var gpusAvailable int64
	for _, node := range nodeList.Items {
		gpusAvailable += node.Status.Capacity.NvidiaGPU().Value()
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
	// GPU drivers might have already been installed.
	if !areGPUsAvailableOnAllSchedulableNodes(f) {
		// Install Nvidia Drivers.
		ds := dsFromManifest(cosNvidiaDriverInstallerUrl)
		ds.Namespace = f.Namespace.Name
		_, err := f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).Create(ds)
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
		podList = append(podList, f.PodClient().Create(makeCudaAdditionTestPod()))
	}
	framework.Logf("Wait for all test pods to succeed")
	// Wait for all pods to succeed
	for _, po := range podList {
		f.PodClient().WaitForSuccess(po.Name, 5*time.Minute)
	}
}

// dsFromManifest reads a .json/yaml file and returns the daemonset in it.
func dsFromManifest(url string) *extensions.DaemonSet {
	var controller extensions.DaemonSet
	framework.Logf("Parsing ds from %v", url)

	var response *http.Response
	var err error
	for i := 1; i <= 5; i++ {
		response, err = http.Get(url)
		if err == nil && response.StatusCode == 200 {
			break
		}
		time.Sleep(time.Duration(i) * time.Second)
	}
	Expect(err).NotTo(HaveOccurred())
	Expect(response.StatusCode).To(Equal(200))
	defer response.Body.Close()

	data, err := ioutil.ReadAll(response.Body)
	Expect(err).NotTo(HaveOccurred())

	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &controller)).NotTo(HaveOccurred())
	return &controller
}

var _ = framework.KubeDescribe("[Feature:GPU]", func() {
	f := framework.NewDefaultFramework("gpus")
	It("run Nvidia GPU tests on Container Optimized OS only", func() {
		testNvidiaGPUsOnCOS(f)
	})
})
