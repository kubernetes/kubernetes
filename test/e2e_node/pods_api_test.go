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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
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

		ginkgo.It("should receive INITIAL_SYNC_COMPLETE and coherent updates", func(ctx context.Context) {
			ginkgo.By("watching for pod events")
			watchCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			defer cancel()
			watchClient, err := client.WatchPods(watchCtx, &podsv1alpha1.WatchPodsRequest{})
			framework.ExpectNoError(err, "failed to watch pods")

			type podEvent struct {
				eventType podsv1alpha1.EventType
				pod       *v1.Pod
			}

			// Channel to receive events for the test pod
			podEventsCh := make(chan podEvent, 1000)
			// Channel to signal initial sync complete
			initialSyncReceived := make(chan struct{}, 1)

			go func() {
				defer ginkgo.GinkgoRecover()
				defer close(podEventsCh)
				defer close(initialSyncReceived)

				for {
					event, err := watchClient.Recv()
					if err != nil {
						// Suppress log if it's a normal shutdown
						if err != context.Canceled && status.Code(err) != codes.Canceled {
							framework.Logf("Watch client Recv error: %v", err)
						}
						return
					}

					if event.Type == podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE {
						select {
						case initialSyncReceived <- struct{}{}:
						default:
						}
						continue
					}

					if event.Pod == nil {
						continue
					}

					var p v1.Pod
					if err := p.Unmarshal(event.Pod); err != nil {
						framework.Logf("Failed to unmarshal pod: %v", err)
						continue
					}
					podEventsCh <- podEvent{eventType: event.Type, pod: &p}
				}
			}()

			ginkgo.By("waiting for INITIAL_SYNC_COMPLETE event")
			gomega.Eventually(initialSyncReceived, "30s").Should(gomega.Receive(), "did not receive INITIAL_SYNC_COMPLETE event")

			ginkgo.By("creating a test pod")
			podName := "coherent-pod-" + string(uuid.NewUUID())
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
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			// Drain events until we see the pod creation
			ginkgo.By("waiting for the test pod ADDED event")
			gomega.Eventually(ctx, func() *v1.Pod {
				for {
					select {
					case e, ok := <-podEventsCh:
						if !ok {
							return nil
						}
						if e.pod.UID == testPod.UID {
							if e.eventType != podsv1alpha1.EventType_ADDED {
								framework.Failf("Received unexpected event type %v before ADDED for pod %v", e.eventType, testPod.UID)
							}
							return e.pod
						}
					default:
						return nil
					}
				}
			}, "1m").ShouldNot(gomega.BeNil(), "did not receive ADDED event for test pod")

			ginkgo.By("deleting the test pod to trigger updates")
			gracePeriod := int64(10) // Use a grace period to ensure we see the Terminating state
			err = e2epod.NewPodClient(f).Delete(ctx, testPod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "failed to delete pod")

			ginkgo.By("verifying status coherency and event ordering")
			// We expect the pod to transition to Terminating (DeletionTimestamp set).
			// We must NOT see it revert to non-terminating (DeletionTimestamp nil) after seeing it terminating.
			sawTerminating := false
			sawDeleted := false
			timeout := time.After(1 * time.Minute)

		Loop:
			for {
				select {
				case e, ok := <-podEventsCh:
					if !ok {
						break Loop
					}
					if e.pod.UID != testPod.UID {
						continue
					}
					if sawDeleted {
						framework.Failf("Received event %v after DELETED for pod %v", e.eventType, testPod.UID)
					}
					if e.eventType == podsv1alpha1.EventType_ADDED {
						framework.Failf("Received unexpected ADDED event after initial ADDED for pod %v", testPod.UID)
					}
					if e.eventType == podsv1alpha1.EventType_DELETED {
						sawDeleted = true
						break Loop
					}

					p := e.pod
					isTerminating := p.DeletionTimestamp != nil
					if sawTerminating && !isTerminating {
						framework.Failf("Pod status regressed: saw Terminating previously, but now got non-Terminating pod: %+v", p)
					}
					if isTerminating {
						sawTerminating = true
					}
				case <-timeout:
					break Loop
				}
			}
			gomega.Expect(sawTerminating).To(gomega.BeTrue(), "did not observe the pod terminating")
			gomega.Expect(sawDeleted).To(gomega.BeTrue(), "did not observe the pod deleted")

			ginkgo.By("waiting for pod to be gone from API server and ensuring no stray events")
			err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, testPod.Name, f.Namespace.Name, 1*time.Minute)
			framework.ExpectNoError(err, "pod did not disappear from API server")

			// Check for any stray events that might have arrived after DELETED
			// We use a small timeout here because if an event was going to arrive, it would likely have arrived by now
			// given that the pod is already gone from the API server.
			gomega.Consistently(ctx, podEventsCh, "2s").ShouldNot(gomega.Receive(gomega.WithTransform(func(e podEvent) types.UID {
				return e.pod.UID
			}, gomega.Equal(testPod.UID))), "received stray event after DELETED")
		})

		ginkgo.It("should receive events in the correct order: ADDED, MODIFIED, DELETED", func(ctx context.Context) {
			ginkgo.By("watching for pod events")
			watchCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			defer cancel()
			watchClient, err := client.WatchPods(watchCtx, &podsv1alpha1.WatchPodsRequest{})
			framework.ExpectNoError(err, "failed to watch pods")

			// Channel to receive events
			eventCh := make(chan *podsv1alpha1.WatchPodsEvent, 1000)

			go func() {
				defer ginkgo.GinkgoRecover()
				defer close(eventCh)
				for {
					event, err := watchClient.Recv()
					if err != nil {
						// Suppress log if it's a normal shutdown
						if err != context.Canceled && status.Code(err) != codes.Canceled {
							framework.Logf("Watch client Recv error in ordering test: %v", err)
						}
						return
					}
					eventCh <- event
				}
			}()

			// 1. Expect INITIAL_SYNC_COMPLETE
			// Note: We might get ADDED events for existing pods before this, but for our new pod,
			// the key is that we start strictly after this check or we identify our pod by UID.
			// Since we create the pod AFTER starting the watch, we expect correct ordering.
			gomega.Eventually(eventCh, "30s").Should(gomega.Receive(
				gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) podsv1alpha1.EventType { return e.Type }, gomega.Equal(podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE)),
			), "expected INITIAL_SYNC_COMPLETE event")

			ginkgo.By("creating a test pod")
			podName := "sequence-pod-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: f.Namespace.Name,
					Labels:    map[string]string{"state": "initial"},
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
			testPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)

			ginkgo.By("updating the pod to trigger MODIFIED events")
			patch := []byte(`{"metadata":{"labels":{"state":"modified-1"}}}`)
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Patch(ctx, testPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch pod (1)")

			patch = []byte(`{"metadata":{"labels":{"state":"modified-2"}}}`)
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Patch(ctx, testPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch pod (2)")

			// 2. Expect ADDED, MODIFIED..., DELETED
			ginkgo.By("deleting the test pod")
			gracePeriod := int64(10)
			err = e2epod.NewPodClient(f).Delete(ctx, testPod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "failed to delete pod")

			ginkgo.By("updating the pod during deletion to trigger more MODIFIED events")
			patch = []byte(`{"metadata":{"labels":{"state":"modified-3"}}}`)
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Patch(ctx, testPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			// It's possible the pod is already gone if the grace period was short, but with 10s it should work.
			framework.ExpectNoError(err, "failed to patch pod (3)")

			ginkgo.By("collecting events for the test pod")
			var podEvents []*podsv1alpha1.WatchPodsEvent
			foundDeleted := false
			timeout := time.After(1 * time.Minute)

		Loop:
			for {
				select {
				case event, ok := <-eventCh:
					if !ok {
						break Loop
					}
					if event.Type == podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE {
						continue
					}
					var p v1.Pod
					if err := p.Unmarshal(event.Pod); err == nil && p.UID == testPod.UID {
						if foundDeleted {
							framework.Failf("Received event %v after DELETED for pod %v", event.Type, testPod.UID)
						}
						podEvents = append(podEvents, event)
						if event.Type == podsv1alpha1.EventType_DELETED {
							foundDeleted = true
							// We've seen DELETED, but we stay in the loop to catch any immediate stray events
							// until our timeout or API server confirmation.
						}
					}
				case <-timeout:
					break Loop
				}
				if foundDeleted {
					break Loop
				}
			}

			gomega.Expect(foundDeleted).To(gomega.BeTrue(), "did not receive DELETED event")
			gomega.Expect(podEvents).NotTo(gomega.BeEmpty())

			ginkgo.By("waiting for pod to be gone from API server and ensuring no stray events")
			err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, testPod.Name, f.Namespace.Name, 1*time.Minute)
			framework.ExpectNoError(err, "pod did not disappear from API server")

			gomega.Consistently(ctx, eventCh, "2s").ShouldNot(gomega.Receive(gomega.WithTransform(func(e *podsv1alpha1.WatchPodsEvent) types.UID {
				var p v1.Pod
				if err := p.Unmarshal(e.Pod); err != nil {
					return ""
				}
				return p.UID
			}, gomega.Equal(testPod.UID))), "received stray event after DELETED")

			ginkgo.By("verifying event content and transitions")
			var previousPod *v1.Pod
			expectedStates := []string{"initial", "modified-1", "modified-2", "modified-3"}
			currentStateIdx := 0

			for i, e := range podEvents {
				var pod v1.Pod
				err := pod.Unmarshal(e.Pod)
				framework.ExpectNoError(err, "failed to unmarshal pod from event %d", i)

				podState := pod.Labels["state"]
				foundIdx := -1
				for j, s := range expectedStates {
					if podState == s {
						foundIdx = j
						break
					}
				}

				if foundIdx != -1 {
					if foundIdx < currentStateIdx {
						framework.Failf("Pod state regressed from %s to %s in event %d", expectedStates[currentStateIdx], podState, i)
					}
					currentStateIdx = foundIdx
				}

				// Verify Status is present (fixing the bug addressed earlier)
				gomega.Expect(pod.Status.Phase).NotTo(gomega.BeEmpty(), "Pod status phase should not be empty in event %d", i)

				if i == 0 {
					gomega.Expect(e.Type).To(gomega.Equal(podsv1alpha1.EventType_ADDED), "First event should be ADDED")
				} else if i == len(podEvents)-1 {
					gomega.Expect(e.Type).To(gomega.Equal(podsv1alpha1.EventType_DELETED), "Last event should be DELETED")
				} else {
					gomega.Expect(e.Type).To(gomega.Equal(podsv1alpha1.EventType_MODIFIED), "Intermediate events should be MODIFIED")
				}

				if previousPod != nil {
					// 1. DeletionTimestamp should not disappear
					if previousPod.DeletionTimestamp != nil {
						gomega.Expect(pod.DeletionTimestamp).NotTo(gomega.BeNil(), "DeletionTimestamp should not disappear in event %d", i)
					}

					// 2. Phase monotonicity (simple)
					// Once Running, don't go back to Pending
					if previousPod.Status.Phase == v1.PodRunning {
						gomega.Expect(pod.Status.Phase).NotTo(gomega.Equal(v1.PodPending), "Pod should not regress from Running to Pending in event %d", i)
					}

					// 3. UID must match (sanity check)
					gomega.Expect(pod.UID).To(gomega.Equal(previousPod.UID), "UID mismatch in event %d", i)
				}

				// Copy for next iteration
				podCopy := pod
				previousPod = &podCopy
			}

			gomega.Expect(currentStateIdx).To(gomega.Equal(len(expectedStates)-1), "did not observe all expected state transitions (last state seen: %s)", expectedStates[currentStateIdx])
		})

	})
})
