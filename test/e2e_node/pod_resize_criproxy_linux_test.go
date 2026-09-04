/*
Copyright The Kubernetes Authors.

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
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e_node/criproxy"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Pod InPlace Resize Memory-Backed Volume with CRI Proxy", framework.WithFeatureGate(features.InPlacePodVerticalScalingMemoryBackedVolumes), func() {
	// Tests the coordinated pipelining of multiple, concurrent volume and container resize operations.
	//
	// The test creates a pod with two memory-backed emptyDir volumes (vol-1 and vol-2) and triggers a patch that:
	// 1. Downsizes vol-1 (100Mi -> 50Mi)
	// 2. Upsizes vol-2 (100Mi -> 250Mi)
	// 3. Upsizes the container's memory limit (220Mi -> 320Mi)
	//
	// By injecting a delay into the container's CRI UpdateContainerResources call, the test verifies that:
	// - The downsize of vol-1 executes immediately (as volume downsizing is the first step in a downsize sequence).
	// - The upsize of vol-2 is blocked and remains at 100Mi while container upsizing is pending (since volume upsizing must wait for container cgroup memory expansion to complete).
	// - After the container cgroup update unblocks, the upsize of vol-2 eventually completes.
	f := framework.NewDefaultFramework("pod-resize-memory-vol-criproxy")

	ginkgo.BeforeEach(func(ctx context.Context) {
		_, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	f.Context("with CRI Proxy delays", feature.CriProxy, func() {
		ginkgo.BeforeEach(func() {
			if e2eCriProxy == nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined. Please run with --cri-proxy-enabled=true")
			}
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
		})

		ginkgo.It("should actuate multiple volume resizing in the correct order, prioritizing downsize before upsize", func(ctx context.Context) {
			podClient := e2epod.NewPodClient(f)
			original := []podresize.ResizableContainerInfo{{
				Name:      "c1",
				Resources: &cgroups.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "220Mi", MemLim: "220Mi"},
			}}
			resizedContainers := []podresize.ResizableContainerInfo{{
				Name:      "c1",
				Resources: &cgroups.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "320Mi", MemLim: "320Mi"},
			}}

			ginkgo.By("creating and verifying pod with two memory-backed emptyDir volumes")
			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod := podresize.MakePodWithResizableContainers(f.Namespace.Name, "", tStamp, original, nil)
			testPod.GenerateName = "resize-vol-ordered-"

			limit100 := resource.MustParse("100Mi")
			testPod.Spec.Volumes = []v1.Volume{
				{
					Name: "vol-1", // to be downsized to 50Mi
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							Medium:    v1.StorageMediumMemory,
							SizeLimit: &limit100,
						},
					},
				},
				{
					Name: "vol-2", // to be upsized to 250Mi
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							Medium:    v1.StorageMediumMemory,
							SizeLimit: &limit100,
						},
					},
				},
			}

			testPod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
				{
					Name:      "vol-1",
					MountPath: "/cache1",
				},
				{
					Name:      "vol-2",
					MountPath: "/cache2",
				},
			}
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			newPod := podClient.CreateSync(ctx, testPod)
			podresize.VerifyPodResources(newPod, original, nil)

			ginkgo.By("Injecting delay into UpdateContainerResources calls")
			const delayTime = 40 * time.Second
			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.UpdateContainerResources {
					framework.Logf("Delaying UpdateContainerResources by %v", delayTime)
					time.Sleep(delayTime)
				}
				return nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("Patching pod: downsize vol-1, upsize vol-2, and upsize container")
			patchBytes := []byte(`{
				"spec": {
					"containers": [
						{
							"name": "c1",
							"resources": {
								"limits": {
									"memory": "320Mi"
								},
								"requests": {
									"memory": "320Mi"
								}
							}
						}
					],
					"volumes": [
						{
							"name": "vol-1",
							"emptyDir": {
								"sizeLimit": "50Mi"
							}
						},
						{
							"name": "vol-2",
							"emptyDir": {
								"sizeLimit": "250Mi"
							}
						}
					]
				}
			}`)
			patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "resize")
			framework.ExpectNoError(pErr, "failed to patch volume/container limits")

			// Actuation sequence:
			// 1. Downsize vol-1 to 50Mi (Volume downsizing is step 1 of downsizing).
			// 2. Call UpdateContainerResources to expand cgroup to 320Mi (Cgroup expansion is step 1 of upsizing).
			//    --> Delayed here for 40s.
			// 3. Upsize vol-2 to 250Mi (Volume upsizing is step 2 of upsizing).
			//
			// So, while blocked at step 2 (during the 40s delay):
			// - vol-1 (downsize) MUST be downsized immediately to 50Mi.
			// - vol-2 (upsize) MUST NOT be upsized yet and must remain at 100Mi.

			ginkgo.By("Verifying vol-1 is downsized immediately while vol-2 remains unchanged")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				stdout, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, newPod.Name, "c1", "df", "-m", "/cache1")
				return stdout, err
			}).WithTimeout(30 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.ContainSubstring("50"))

			// Verify that vol-2 remains at 100Mi during the CRI block
			stdout, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, newPod.Name, "c1", "df", "-m", "/cache2")
			framework.ExpectNoError(err)
			gomega.Expect(stdout).To(gomega.ContainSubstring("100"))

			ginkgo.By("Waiting for the entire actuation to complete after CRI unblocks")
			expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, resizedContainers)
			gotPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, patchedPod)
			podresize.ExpectPodResized(ctx, f, gotPod, expected)

			ginkgo.By("Verifying both volume resizes are eventually complete")

			stdout, _, err = e2epod.ExecCommandInContainerWithFullOutput(f, gotPod.Name, "c1", "df", "-m", "/cache1")
			framework.ExpectNoError(err)
			gomega.Expect(stdout).To(gomega.ContainSubstring("50"))

			stdout, _, err = e2epod.ExecCommandInContainerWithFullOutput(f, gotPod.Name, "c1", "df", "-m", "/cache2")
			framework.ExpectNoError(err)
			gomega.Expect(stdout).To(gomega.ContainSubstring("250"))

			ginkgo.By("deleting pod")
			podClient.DeleteSync(ctx, gotPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		})
	})
})
