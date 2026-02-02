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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	draDriverNamespace = "kube-system"
	draDeviceClassName = "gpu.nvidia.com"
	draDriverName      = "gpu.nvidia.com"

	// DaemonSet name from helm chart (kubetest2-ec2 uses helm template)
	draDaemonSetName = "nvidia-dra-driver-gpu-kubelet-plugin"
)

var _ = SIGDescribe(feature.DynamicResourceAllocation, "GPU", framework.WithSerial(), func() {

	f := framework.NewDefaultFramework("dra-gpu")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("aws")
		checkDRADriverReady(ctx, f)
	})

	f.It("should detect GPUs via DRA ResourceSlice", func(ctx context.Context) {
		ginkgo.By("Listing ResourceSlices")
		slices, err := f.ClientSet.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking for NVIDIA GPU devices in ResourceSlices")
		var gpuCount int
		for _, slice := range slices.Items {
			if slice.Spec.Driver == draDriverName {
				gpuCount += len(slice.Spec.Devices)
				framework.Logf("Found ResourceSlice %s with %d devices from driver %s",
					slice.Name, len(slice.Spec.Devices), slice.Spec.Driver)
			}
		}
		gomega.Expect(gpuCount).To(gomega.BeNumerically(">", 0),
			"Expected at least one GPU in ResourceSlices from driver %s", draDriverName)
		framework.Logf("Total GPUs found via DRA: %d", gpuCount)
	})

	f.It("should allocate single GPU via ResourceClaim and run nvidia-smi and cuda-demo-suite", func(ctx context.Context) {
		ginkgo.By("Creating a ResourceClaim for a single GPU")
		claim := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "single-gpu-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "gpu",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: draDeviceClassName,
						},
					}},
				},
			},
		}
		claim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created ResourceClaim: %s", claim.Name)

		ginkgo.By("Creating a pod that uses the ResourceClaim")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-dra-test-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				ResourceClaims: []v1.PodResourceClaim{{
					Name:              "gpu",
					ResourceClaimName: &claim.Name,
				}},
				Containers: []v1.Container{{
					Name:  "nvidia-smi",
					Image: "nvidia/cuda:12.5.0-devel-ubuntu22.04",
					Command: []string{
						"bash",
						"-c",
						`
nvidia-smi
apt-get update -y && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated cuda-demo-suite-12-5
/usr/local/cuda/extras/demo_suite/deviceQuery
/usr/local/cuda/extras/demo_suite/vectorAdd
/usr/local/cuda/extras/demo_suite/bandwidthTest --device=all --csv
/usr/local/cuda/extras/demo_suite/busGrind -a
`,
					},
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "gpu"}},
					},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod = e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for pod to complete")
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking pod logs for GPU detection and CUDA tests")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "nvidia-smi")
		framework.ExpectNoError(err)
		framework.Logf("nvidia-smi and cuda-demo-suite output:\n%s", logs)
		gomega.Expect(logs).To(gomega.ContainSubstring("NVIDIA-SMI"))
		gomega.Expect(logs).To(gomega.ContainSubstring("Driver Version:"))
		gomega.Expect(logs).To(gomega.ContainSubstring("CUDA Version:"))
		gomega.Expect(logs).To(gomega.ContainSubstring("deviceQuery, CUDA Driver"))
		gomega.Expect(logs).To(gomega.ContainSubstring("Result = PASS"))
	})

	f.It("should run TensorFlow matrix multiplication via DRA", func(ctx context.Context) {
		ginkgo.By("Creating a ResourceClaim for TensorFlow GPU test")
		claim := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "tf-gpu-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "gpu",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: draDeviceClassName,
						},
					}},
				},
			},
		}
		claim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Creating a TensorFlow pod that uses the ResourceClaim")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "tf-matmul-dra-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				ResourceClaims: []v1.PodResourceClaim{{
					Name:              "gpu",
					ResourceClaimName: &claim.Name,
				}},
				Containers: []v1.Container{{
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
						Claims: []v1.ResourceClaim{{Name: "gpu"}},
					},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod = e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for TensorFlow pod to complete")
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking TensorFlow output")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "gpu-matmul")
		framework.ExpectNoError(err)
		framework.Logf("TensorFlow output:\n%s", logs)
		gomega.Expect(logs).To(gomega.ContainSubstring("TensorFlow version"))
		gomega.Expect(logs).To(gomega.ContainSubstring("Matrix multiplication result:"))
		gomega.Expect(logs).To(gomega.ContainSubstring("Time taken for 5000x5000 matrix multiplication"))
	})

	f.It("should share GPU between containers in same pod via DRA", func(ctx context.Context) {
		ginkgo.By("Creating a ResourceClaim for shared GPU")
		claim := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "shared-gpu-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "gpu",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: draDeviceClassName,
						},
					}},
				},
			},
		}
		claim, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with two containers sharing the same GPU")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "shared-gpu-pod-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				ResourceClaims: []v1.PodResourceClaim{{
					Name:              "gpu",
					ResourceClaimName: &claim.Name,
				}},
				Containers: []v1.Container{
					{
						Name:    "container-1",
						Image:   "nvidia/cuda:12.5.0-devel-ubuntu22.04",
						Command: []string{"nvidia-smi", "-L"},
						Resources: v1.ResourceRequirements{
							Claims: []v1.ResourceClaim{{Name: "gpu"}},
						},
					},
					{
						Name:    "container-2",
						Image:   "nvidia/cuda:12.5.0-devel-ubuntu22.04",
						Command: []string{"nvidia-smi", "-L"},
						Resources: v1.ResourceRequirements{
							Claims: []v1.ResourceClaim{{Name: "gpu"}},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod = e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for shared GPU pod to complete")
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying both containers see the GPU")
		logs1, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "container-1")
		framework.ExpectNoError(err)
		logs2, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "container-2")
		framework.ExpectNoError(err)

		framework.Logf("Container 1 output:\n%s", logs1)
		framework.Logf("Container 2 output:\n%s", logs2)

		gomega.Expect(logs1).To(gomega.ContainSubstring("GPU"))
		gomega.Expect(logs2).To(gomega.ContainSubstring("GPU"))
	})
})

// checkDRADriverReady verifies the NVIDIA DRA driver is deployed and ready.
// Fails the test if the driver is not available.
func checkDRADriverReady(ctx context.Context, f *framework.Framework) {
	ginkgo.By("Checking for NVIDIA DRA driver")

	// Check DaemonSet exists and is ready
	ds, err := f.ClientSet.AppsV1().DaemonSets(draDriverNamespace).Get(ctx, draDaemonSetName, metav1.GetOptions{})
	framework.ExpectNoError(err, "NVIDIA DRA driver DaemonSet %s/%s not deployed", draDriverNamespace, draDaemonSetName)
	gomega.Expect(ds.Status.NumberReady).To(gomega.BeNumerically(">", 0),
		"NVIDIA DRA driver not ready: 0 pods ready")

	// Check DeviceClass exists
	_, err = f.ClientSet.ResourceV1().DeviceClasses().Get(ctx, draDeviceClassName, metav1.GetOptions{})
	framework.ExpectNoError(err, "DeviceClass %s not found", draDeviceClassName)

	// Check ResourceSlices have GPUs
	slices, err := f.ClientSet.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to list ResourceSlices")
	var gpuCount int
	for _, slice := range slices.Items {
		if slice.Spec.Driver == draDriverName {
			gpuCount += len(slice.Spec.Devices)
		}
	}
	gomega.Expect(gpuCount).To(gomega.BeNumerically(">", 0),
		"No GPUs found in ResourceSlices from driver %s", draDriverName)

	framework.Logf("DRA driver ready: %d pods, %d GPUs", ds.Status.NumberReady, gpuCount)
}
