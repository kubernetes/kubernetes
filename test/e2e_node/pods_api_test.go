/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/pods"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// podsAPISuite is a Ginkgo test suite for the Kubelet Pods API.
var _ = framework.SIGDescribe("kubelet-pods-api")

var _ = ginkgo.Describe("Kubelet Pods API", func() {
	f := framework.NewDefaultFramework("pods-api-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when the PodsAPI feature gate is enabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(kubefeatures.PodsAPI)] = true
		})

		socketPath := "/var/lib/kubelet/pods-api"

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for the local endpoint to be present")
			gomega.Eventually(ctx, func() error {
				_, err := os.Stat(filepath.Join(socketPath, pods.Socket+".sock"))
				return err
			}, 30*time.Second, 1*time.Second).Should(gomega.Succeed(), "Pods API socket not present")
		})

		ginkgo.It("should be able to list, get, and watch pods", func(ctx context.Context) {
			endpoint, err := util.LocalEndpoint(socketPath, pods.Socket)
			framework.ExpectNoError(err, "failed to get local endpoint for Pods API")

			conn, err := grpc.DialContext(ctx, endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
			framework.ExpectNoError(err, "failed to dial Pods API")
			defer func(conn *grpc.ClientConn) {
				_ = conn.Close()
			}(conn)

			client := podsv1alpha1.NewPodsClient(conn)

			// Create a test pod
			podName := "test-pod-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   "registry.k8s.io/busybox",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}

			ginkgo.By("creating a test pod")
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			ginkgo.By("listing pods and ensuring the test pod is present")
			listResp, err := client.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
			framework.ExpectNoError(err, "failed to list pods")
			var foundPod bool
			for _, p := range listResp.Pods {
				var pod v1.Pod
				err := json.Unmarshal(p, &pod)
				framework.ExpectNoError(err, "failed to unmarshal pod from list")
				if pod.ObjectMeta.UID == testPod.UID {
					foundPod = true
					break
				}
			}
			gomega.Expect(foundPod).To(gomega.BeTrueBecause("test pod not found in list"))

			ginkgo.By("getting the test pod by UID")
			getResp, err := client.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: string(testPod.UID)})
			framework.ExpectNoError(err, "failed to get pod")
			var podFromGet v1.Pod
			err = json.Unmarshal(getResp.Pod, &podFromGet)
			framework.ExpectNoError(err, "failed to unmarshal pod from get")
			gomega.Expect(podFromGet.ObjectMeta.UID).To(gomega.Equal(testPod.UID))

			ginkgo.By("watching for pod events")
			watchCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			watchClient, err := client.WatchPods(watchCtx, &podsv1alpha1.WatchPodsRequest{})
			framework.ExpectNoError(err, "failed to watch pods")

			// Expect to receive an ADDED event for the new pod
			event, err := watchClient.Recv()
			framework.ExpectNoError(err, "failed to receive watch event")
			gomega.Expect(event.Type).To(gomega.Equal(podsv1alpha1.EventType_ADDED))
			var podFromWatch v1.Pod
			err = json.Unmarshal(event.Pod, &podFromWatch)
			framework.ExpectNoError(err, "failed to unmarshal pod from watch event")
			gomega.Expect(podFromWatch.ObjectMeta.UID).To(gomega.Equal(testPod.UID))

			ginkgo.By("deleting the test pod")
			err = e2epod.NewPodClient(f).Delete(ctx, testPod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete pod")

			// Expect to receive a DELETED event
			event, err = watchClient.Recv()
			framework.ExpectNoError(err, "failed to receive watch event")
			gomega.Expect(event.Type).To(gomega.Equal(podsv1alpha1.EventType_DELETED))
			var podFromDeletedEvent v1.Pod
			err = json.Unmarshal(event.Pod, &podFromDeletedEvent)
			framework.ExpectNoError(err, "failed to unmarshal pod from delete event")
			gomega.Expect(podFromDeletedEvent.ObjectMeta.UID).To(gomega.Equal(testPod.UID))
		})

		ginkgo.It("should respect field masks", func(ctx context.Context) {
			endpoint, err := util.LocalEndpoint(socketPath, pods.Socket)
			framework.ExpectNoError(err, "failed to get local endpoint for Pods API")

			conn, err := grpc.DialContext(ctx, endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
			framework.ExpectNoError(err, "failed to dial Pods API")
			defer func(conn *grpc.ClientConn) {
				_ = conn.Close()
			}(conn)

			client := podsv1alpha1.NewPodsClient(conn)

			podName := "mask-test-pod-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					NodeName: framework.TestContext.NodeName,
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   "registry.k8s.io/busybox",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}

			ginkgo.By("creating a test pod")
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			ginkgo.By("listing pods with a field mask")
			md := metadata.New(map[string]string{pods.FieldMaskMetadataKey: "metadata.name,spec.nodeName"})
			ctxWithMask := metadata.NewOutgoingContext(ctx, md)
			listResp, err := client.ListPods(ctxWithMask, &podsv1alpha1.ListPodsRequest{})
			framework.ExpectNoError(err, "failed to list pods with field mask")

			var maskedPod *v1.Pod
			for _, p := range listResp.Pods {
				pod := &v1.Pod{}
				err := json.Unmarshal(p, pod)
				framework.ExpectNoError(err, "failed to unmarshal pod")
				if pod.ObjectMeta.Name == podName {
					maskedPod = pod
					break
				}
			}
			gomega.Expect(maskedPod).NotTo(gomega.BeNil(), "masked pod not found in list")
			gomega.Expect(maskedPod.Name).To(gomega.Equal(podName))
			gomega.Expect(maskedPod.Spec.NodeName).To(gomega.Equal(framework.TestContext.NodeName))
			gomega.Expect(maskedPod.Namespace).To(gomega.BeEmpty())
			gomega.Expect(maskedPod.Spec.Containers).To(gomega.BeEmpty())

			ginkgo.By("getting a pod with a field mask")
			md = metadata.New(map[string]string{pods.FieldMaskMetadataKey: "metadata.namespace,status.phase"})
			ctxWithMask = metadata.NewOutgoingContext(ctx, md)
			getResp, err := client.GetPod(ctxWithMask, &podsv1alpha1.GetPodRequest{PodUID: string(testPod.UID)})
			framework.ExpectNoError(err, "failed to get pod with field mask")

			maskedPod = &v1.Pod{}
			err = json.Unmarshal(getResp.Pod, maskedPod)
			framework.ExpectNoError(err, "failed to unmarshal pod")
			gomega.Expect(maskedPod.Name).To(gomega.BeEmpty())
			gomega.Expect(maskedPod.Namespace).To(gomega.Equal(f.Namespace.Name))
			gomega.Expect(maskedPod.Status.Phase).ToNot(gomega.BeEmpty())
			gomega.Expect(maskedPod.Spec.NodeName).To(gomega.BeEmpty())
		})
	})
})
