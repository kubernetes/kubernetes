//go:build linux

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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/pods"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"sigs.k8s.io/yaml"
)

// podsAPISuite is a Ginkgo test suite for the Kubelet Pods API.
var _ = SIGDescribe("Kubelet Pods API", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pods-api-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when the PodsAPI feature gate is enabled", func() {
		var (
			conn   *grpc.ClientConn
			client podsv1alpha1.PodsClient
		)
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(kubefeatures.PodsAPI)] = true
		})

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for the node to be ready")
			waitForNodeReady(ctx)

			ginkgo.By("Connecting to Pods API")
			endpoint, err := util.LocalEndpoint("/var/lib/kubelet/pods-api", pods.Socket)
			framework.ExpectNoError(err, "failed to get local endpoint for Pods API")

			gomega.Eventually(ctx, func(ctx context.Context) error {
				conn, err = grpc.NewClient(endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
				if err != nil {
					return err
				}
				client = podsv1alpha1.NewPodsClient(conn)
				// Make a simple call to ensure the server is responsive.
				_, err = client.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
				if err != nil {
					_ = conn.Close() // Close connection on failure to retry dialing
					return err
				}
				return nil
			}, "1m", "5s").Should(gomega.Succeed(), "failed to connect to Pods API")
		})

		ginkgo.AfterEach(func() {
			if conn != nil {
				_ = conn.Close()
			}
		})

		ginkgo.It("should be able to list, get, and watch pods", func(ctx context.Context) {
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
							Image:   "busybox:1.36",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}

			ginkgo.By("creating a test pod")
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			ginkgo.By("listing pods and ensuring the test pod is present")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				listResp, err := client.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
				if err != nil {
					framework.Logf("failed to list pods, will retry: %v", err)
					return false
				}
				for _, p := range listResp.Pods {
					var pod v1.Pod
					err := pod.Unmarshal(p)
					if err != nil {
						framework.Logf("failed to unmarshal pod, will retry: %v", err)
						return false
					}
					if pod.ObjectMeta.UID == testPod.UID {
						return true
					}
				}
				return false
			}, "1m", "5s").Should(gomega.BeTrueBecause("test pod should be present in the list"))

			ginkgo.By("getting the test pod by UID")
			getResp, err := client.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: string(testPod.UID)})
			framework.ExpectNoError(err, "failed to get pod")
			var podFromGet v1.Pod
			err = podFromGet.Unmarshal(getResp.Pod)
			framework.ExpectNoError(err, "failed to unmarshal pod from get")
			gomega.Expect(podFromGet.ObjectMeta.UID).To(gomega.Equal(testPod.UID))

			ginkgo.By("watching for pod events")
			watchCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			defer cancel()
			watchClient, err := client.WatchPods(watchCtx, &podsv1alpha1.WatchPodsRequest{})
			framework.ExpectNoError(err, "failed to watch pods")

			// Expect to receive an ADDED event for the new pod
			var podFromWatch v1.Pod
			gomega.Eventually(ctx, func(ctx context.Context) (types.UID, error) {
				event, err := watchClient.Recv()
				if err != nil {
					return "", err
				}
				if event.Type != podsv1alpha1.EventType_ADDED {
					return "", nil // Continue waiting
				}
				if err := podFromWatch.Unmarshal(event.Pod); err != nil {
					return "", err
				}
				return podFromWatch.ObjectMeta.UID, nil
			}, "1m", "1s").Should(gomega.Equal(testPod.UID), "did not receive ADDED event for the test pod")

			ginkgo.By("deleting the test pod")
			eventChan := make(chan *podsv1alpha1.WatchPodsEvent, 10)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer close(eventChan)
				for {
					event, err := watchClient.Recv()
					if err != nil {
						return // Error will be detected by Eventually timeout
					}
					eventChan <- event
				}
			}()

			gracePeriod := int64(0)
			err = e2epod.NewPodClient(f).Delete(ctx, testPod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "failed to delete pod")

			// Expect to receive a DELETED event
			gomega.Eventually(eventChan, "1m", "100ms").Should(gomega.Receive(gomega.SatisfyAll(
				gomega.WithTransform(func(event *podsv1alpha1.WatchPodsEvent) podsv1alpha1.EventType { return event.Type }, gomega.Equal(podsv1alpha1.EventType_DELETED)),
				gomega.WithTransform(
					func(event *podsv1alpha1.WatchPodsEvent) types.UID {
						var p v1.Pod
						if err := p.Unmarshal(event.Pod); err != nil {
							return ""
						}
						return p.UID
					},
					gomega.Equal(testPod.UID),
				),
			)), "did not receive DELETED event for the test pod")
		})

		ginkgo.It("should be able to list and watch static pods", func(ctx context.Context) {
			ginkgo.By("watching for pod events")
			watchCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			defer cancel()
			watchClient, err := client.WatchPods(watchCtx, &podsv1alpha1.WatchPodsRequest{})
			framework.ExpectNoError(err, "failed to watch pods")

			eventsCh := make(chan *podsv1alpha1.WatchPodsEvent, 100)
			go func() {
				defer ginkgo.GinkgoRecover()
				for {
					ev, err := watchClient.Recv()
					if err != nil {
						return
					}
					select {
					case eventsCh <- ev:
					case <-watchCtx.Done():
						return
					}
				}
			}()

			ginkgo.By("creating a static pod manifest")
			staticPodName := "static-pod-" + string(uuid.NewUUID())
			staticPodPath := filepath.Join(kubeletCfg.StaticPodPath, f.Namespace.Name+"-"+staticPodName+".yaml")

			pod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      staticPodName,
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   "busybox:1.36",
							Command: []string{"sleep", "3600"},
						},
					},
				},
			}

			podBytes, err := json.Marshal(pod)
			framework.ExpectNoError(err, "failed to marshal pod")
			podYaml, err := yaml.JSONToYAML(podBytes)
			framework.ExpectNoError(err, "failed to convert to yaml")

			err = os.WriteFile(staticPodPath, podYaml, 0644)
			framework.ExpectNoError(err, "failed to write static pod file")

			defer func() {
				_ = os.Remove(staticPodPath)
			}()

			ginkgo.By("waiting for the static pod ADDED event")
			gomega.Eventually(eventsCh, "2m").Should(gomega.Receive(gomega.SatisfyAll(
				gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) podsv1alpha1.EventType { return e.Type }, gomega.Equal(podsv1alpha1.EventType_ADDED)),
				gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) bool {
					var p v1.Pod
					if err := p.Unmarshal(e.Pod); err != nil {
						return false
					}
					return p.Namespace == f.Namespace.Name && p.Name == staticPodName+"-"+framework.TestContext.NodeName
				}, gomega.BeTrueBecause("static pod should match the expected name and namespace")),
			)), "did not receive ADDED event")

			ginkgo.By("deleting the static pod manifest")
			err = os.Remove(staticPodPath)
			framework.ExpectNoError(err, "failed to delete static pod file")

			ginkgo.By("waiting for the static pod DELETED event")
			gomega.Eventually(eventsCh, "2m").Should(gomega.Receive(gomega.SatisfyAll(
				gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) podsv1alpha1.EventType { return e.Type }, gomega.Equal(podsv1alpha1.EventType_DELETED)),
				gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) bool {
					var p v1.Pod
					if err := p.Unmarshal(e.Pod); err != nil {
						return false
					}
					return p.Namespace == f.Namespace.Name && p.Name == staticPodName+"-"+framework.TestContext.NodeName
				}, gomega.BeTrueBecause("static pod should match the expected name and namespace")),
			)), "did not receive DELETED event")
		})

	})
})
