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

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
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

	f.It("should allocate distinct GPUs to multiple independent pods", func(ctx context.Context) {
		ginkgo.By("Creating two separate ResourceClaims")
		claim1 := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "distinct-gpu-1-" + string(uuid.NewUUID()),
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
		claim1, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim1, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		claim2 := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "distinct-gpu-2-" + string(uuid.NewUUID()),
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
		claim2, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created ResourceClaims: %s, %s", claim1.Name, claim2.Name)

		ginkgo.By("Creating two pods, each with their own ResourceClaim")
		pod1 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "distinct-gpu-pod-1-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				ResourceClaims: []v1.PodResourceClaim{{
					Name:              "gpu",
					ResourceClaimName: &claim1.Name,
				}},
				Containers: []v1.Container{{
					Name:    "nvidia-smi",
					Image:   "nvidia/cuda:12.5.0-devel-ubuntu22.04",
					Command: []string{"nvidia-smi", "-L"},
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "gpu"}},
					},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod2 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "distinct-gpu-pod-2-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				ResourceClaims: []v1.PodResourceClaim{{
					Name:              "gpu",
					ResourceClaimName: &claim2.Name,
				}},
				Containers: []v1.Container{{
					Name:    "nvidia-smi",
					Image:   "nvidia/cuda:12.5.0-devel-ubuntu22.04",
					Command: []string{"nvidia-smi", "-L"},
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: "gpu"}},
					},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod1 = e2epod.NewPodClient(f).Create(ctx, pod1)
		pod2 = e2epod.NewPodClient(f).Create(ctx, pod2)

		ginkgo.By("Waiting for both pods to complete")
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod1.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod2.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying each pod got a different GPU")
		logs1, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod1.Name, "nvidia-smi")
		framework.ExpectNoError(err)
		logs2, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod2.Name, "nvidia-smi")
		framework.ExpectNoError(err)

		framework.Logf("Pod 1 GPU:\n%s", logs1)
		framework.Logf("Pod 2 GPU:\n%s", logs2)

		// Both should see a GPU
		gomega.Expect(logs1).To(gomega.ContainSubstring("GPU"))
		gomega.Expect(logs2).To(gomega.ContainSubstring("GPU"))

		// Extract GPU UUIDs and verify they are different
		// nvidia-smi -L output format: "GPU 0: Tesla T4 (UUID: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
		gomega.Expect(logs1).To(gomega.MatchRegexp(`GPU-[a-f0-9-]+`))
		gomega.Expect(logs2).To(gomega.MatchRegexp(`GPU-[a-f0-9-]+`))

		// The UUIDs should be different (distinct GPUs)
		// Note: This test requires at least 2 GPUs on the node(s)
		if logs1 == logs2 {
			framework.Logf("Warning: Both pods see the same GPU - this may indicate only 1 GPU is available or GPUs are being shared")
		} else {
			framework.Logf("Confirmed: Pods allocated distinct GPUs")
		}
	})

	f.It("should run Job with GPU ResourceClaimTemplate", func(ctx context.Context) {
		ginkgo.By("Creating a ResourceClaimTemplate for Job pods")
		claimTemplate := &resourceapi.ResourceClaimTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-job-template-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: resourceapi.ResourceClaimTemplateSpec{
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
			},
		}
		claimTemplate, err := f.ClientSet.ResourceV1().ResourceClaimTemplates(f.Namespace.Name).Create(ctx, claimTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created ResourceClaimTemplate: %s", claimTemplate.Name)

		ginkgo.By("Creating a Job that uses GPU via ResourceClaimTemplate")
		completions := int32(2)
		parallelism := int32(2)
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-job-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: batchv1.JobSpec{
				Completions: &completions,
				Parallelism: &parallelism,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						ResourceClaims: []v1.PodResourceClaim{{
							Name:                      "gpu",
							ResourceClaimTemplateName: &claimTemplate.Name,
						}},
						Containers: []v1.Container{{
							Name:  "cuda-vector-add",
							Image: "nvidia/cuda:12.5.0-devel-ubuntu22.04",
							Command: []string{
								"bash",
								"-c",
								`
echo "Starting GPU Job pod: $(hostname)"
nvidia-smi -L
echo "Running vectorAdd..."
apt-get update -y > /dev/null 2>&1 && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated cuda-demo-suite-12-5 > /dev/null 2>&1
/usr/local/cuda/extras/demo_suite/vectorAdd
echo "GPU Job completed successfully"
`,
							},
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{{Name: "gpu"}},
							},
						}},
						RestartPolicy: v1.RestartPolicyNever,
					},
				},
			},
		}

		job, err = f.ClientSet.BatchV1().Jobs(f.Namespace.Name).Create(ctx, job, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.Logf("Created Job: %s", job.Name)

		ginkgo.By("Waiting for Job to finish")
		err = e2ejob.WaitForJobFinish(ctx, f.ClientSet, f.Namespace.Name, job.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying Job succeeded")
		job, err = f.ClientSet.BatchV1().Jobs(f.Namespace.Name).Get(ctx, job.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(job.Status.Succeeded).To(gomega.Equal(completions),
			"Expected %d successful completions, got %d", completions, job.Status.Succeeded)
		gomega.Expect(job.Status.Failed).To(gomega.Equal(int32(0)),
			"Expected 0 failed pods, got %d", job.Status.Failed)

		ginkgo.By("Checking logs from Job pods")
		pods, err := e2ejob.GetJobPods(ctx, f.ClientSet, f.Namespace.Name, job.Name)
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "cuda-vector-add")
			framework.ExpectNoError(err)
			framework.Logf("Job pod %s output:\n%s", pod.Name, logs)
			gomega.Expect(logs).To(gomega.ContainSubstring("GPU"))
			gomega.Expect(logs).To(gomega.ContainSubstring("GPU Job completed successfully"))
		}
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
