/*
Copyright 2024 The Kubernetes Authors.

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

package node

import (
	"context"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2egpu "k8s.io/kubernetes/test/e2e/framework/gpu"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe(feature.GPUDevicePlugin, "Sanity test for Nvidia Device", func() {

	f := framework.NewDefaultFramework("nvidia-gpu")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("aws")
		podClient = e2epod.NewPodClient(f)
	})

	f.It("should run nvidia-smi cli", func(ctx context.Context) {
		checkEnvironmentAndSkipIfNeeded(ctx, f.ClientSet)
		pod := testNvidiaCLIPod()
		pod.Spec.Containers[0].Command = []string{"nvidia-smi"}

		ginkgo.By("Creating a pod that runs nvidia-smi")
		createAndValidatePod(ctx, f, podClient, pod)

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking output from nvidia-smi")
		gomega.Expect(log).To(gomega.ContainSubstring("NVIDIA-SMI"))
		gomega.Expect(log).To(gomega.ContainSubstring("Driver Version:"))
		gomega.Expect(log).To(gomega.ContainSubstring("CUDA Version:"))
	})

	f.It("should run gpu based matrix multiplication", func(ctx context.Context) {
		checkEnvironmentAndSkipIfNeeded(ctx, f.ClientSet)
		pod := testMatrixMultiplicationPod()

		ginkgo.By("Creating a pod that runs matrix multiplication")
		createAndValidatePod(ctx, f, podClient, pod)

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking output from nvidia-smi")
		gomega.Expect(log).To(gomega.ContainSubstring("TensorFlow version"))
		gomega.Expect(log).To(gomega.ContainSubstring("Matrix multiplication result:"))
		gomega.Expect(log).To(gomega.ContainSubstring("Time taken for 5000x5000 matrix multiplication"))
	})
})

func createAndValidatePod(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, pod *v1.Pod) {
	pod = podClient.Create(ctx, pod)

	ginkgo.By("Watching for error events or started pod")
	ev, err := podClient.WaitForErrorEventOrSuccess(ctx, pod)
	framework.ExpectNoError(err)
	gomega.Expect(ev).To(gomega.BeNil())

	ginkgo.By("Waiting for pod completion")
	err = e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
	framework.ExpectNoError(err)
	pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Checking that the pod succeeded")
	gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
}

func testNvidiaCLIPod() *v1.Pod {
	podName := "gpu-cli-" + string(uuid.NewUUID())
	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nvidia-smi",
					Image: "nvidia/cuda:12.3.2-runtime-ubuntu22.04",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"nvidia.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return &pod
}

func testMatrixMultiplicationPod() *v1.Pod {
	podName := "gpu-matmul-" + string(uuid.NewUUID())
	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "gpu-matmul",
					Image: "tensorflow/tensorflow:latest-gpu",
					Command: []string{
						"python",
						"-c",
						`
import tensorflow as tf
import time

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Simple matrix multiplication test
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print("Matrix multiplication result:", c.numpy())

# Performance test
n = 5000
start_time = time.time()
with tf.device('/GPU:0'):
    matrix1 = tf.random.normal((n, n))
    matrix2 = tf.random.normal((n, n))
    result = tf.matmul(matrix1, matrix2)
end_time = time.time()

print(f"Time taken for {n}x{n} matrix multiplication: {end_time - start_time:.2f} seconds")
`,
					},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"nvidia.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return &pod
}

func checkEnvironmentAndSkipIfNeeded(ctx context.Context, clientSet clientset.Interface) {
	nodes, err := e2enode.GetReadySchedulableNodes(ctx, clientSet)
	framework.ExpectNoError(err)
	capacity := 0
	allocatable := 0
	for _, node := range nodes.Items {
		val, ok := node.Status.Capacity[e2egpu.NVIDIAGPUResourceName]
		if !ok {
			continue
		}
		capacity += int(val.Value())
		val, ok = node.Status.Allocatable[e2egpu.NVIDIAGPUResourceName]
		if !ok {
			continue
		}
		allocatable += int(val.Value())
	}
	if capacity == 0 {
		e2eskipper.Skipf("%d ready nodes do not have any Nvidia GPU(s). Skipping...", len(nodes.Items))
	}
	if allocatable == 0 {
		e2eskipper.Skipf("%d ready nodes do not have any allocatable Nvidia GPU(s). Skipping...", len(nodes.Items))
	}
}
