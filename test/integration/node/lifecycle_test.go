/*
Copyright 2020 The Kubernetes Authors.

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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	"k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// TestEvictionForNoExecuteTaintAddedByUser tests taint-based eviction for a node tainted NoExecute
func TestEvictionForNoExecuteTaintAddedByUser(t *testing.T) {
	// we need at least 2 nodes to prevent lifecycle manager from entering "fully-disrupted" mode
	nodeCount := 3
	nodeIndex := 1 // the exact node doesn't matter, pick one

	tests := map[string]struct {
		startStandaloneTaintEvictionController bool
		wantPodEvicted                         bool
	}{
		"Test eviction for NoExecute taint added by user; separate taint eviction controller not started": {
			startStandaloneTaintEvictionController: false,
			wantPodEvicted:                         false,
		},
		"Test eviction for NoExecute taint added by user; separate taint eviction controller started": {
			startStandaloneTaintEvictionController: true,
			wantPodEvicted:                         true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var nodes []*v1.Node
			for i := range nodeCount {
				node := &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name:   fmt.Sprintf("testnode-%d", i),
						Labels: map[string]string{"node.kubernetes.io/exclude-disruption": "true"},
					},
					Spec: v1.NodeSpec{},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:   v1.NodeReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				}
				nodes = append(nodes, node)
			}
			testPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "testpod",
				},
				Spec: v1.PodSpec{
					NodeName: nodes[nodeIndex].Name,
					Containers: []v1.Container{
						{Name: "container", Image: imageutils.GetPauseImageName()},
					},
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					Conditions: []v1.PodCondition{
						{
							Type:   v1.PodReady,
							Status: v1.ConditionTrue,
						},
					},
				},
			}
			testCtx := testutils.InitTestAPIServer(t, "taint-no-execute", nil)
			cs := testCtx.ClientSet

			// Build clientset and informers for controllers.
			externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
			externalClientConfig.QPS = -1
			externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
			externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)

			// Start NodeLifecycleController for taint.
			nc, err := nodelifecycle.NewNodeLifecycleController(
				testCtx.Ctx,
				externalInformers.Coordination().V1().Leases(),
				externalInformers.Core().V1().Pods(),
				externalInformers.Core().V1().Nodes(),
				externalInformers.Apps().V1().DaemonSets(),
				cs,
				1*time.Second,    // Node monitor grace period
				time.Minute,      // Node startup grace period
				time.Millisecond, // Node monitor period
				100,              // Eviction limiter QPS
				100,              // Secondary eviction limiter QPS
				50,               // Large cluster threshold
				0.55,             // Unhealthy zone threshold
			)
			if err != nil {
				t.Fatalf("Failed to create node controller: %v", err)
			}

			// Waiting for all controllers to sync
			externalInformers.Start(testCtx.Ctx.Done())
			externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

			// Run all controllers
			go nc.Run(testCtx.Ctx)

			// Start TaintManager
			if test.startStandaloneTaintEvictionController {
				tm, _ := tainteviction.New(
					testCtx.Ctx,
					testCtx.ClientSet,
					externalInformers.Core().V1().Pods(),
					externalInformers.Core().V1().Nodes(),
					names.TaintEvictionController,
				)
				go tm.Run(testCtx.Ctx)
			}

			for index := range nodes {
				nodes[index], err = cs.CoreV1().Nodes().Create(testCtx.Ctx, nodes[index], metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create node, err: %v", err)
				}
			}

			testPod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, testPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Test Failed: error: %v, while creating pod", err)
			}

			if err := testutils.AddTaintToNode(cs, nodes[nodeIndex].Name, v1.Taint{Key: "CustomTaintByUser", Effect: v1.TaintEffectNoExecute}); err != nil {
				t.Errorf("Failed to taint node in test %s <%s>, err: %v", name, nodes[nodeIndex].Name, err)
			}

			err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, time.Second*20, true, testutils.PodIsGettingEvicted(cs, testPod.Namespace, testPod.Name))
			if err != nil && test.wantPodEvicted {
				t.Fatalf("Test Failed: error %v while waiting for pod %q to be evicted", err, klog.KObj(testPod))
			} else if !wait.Interrupted(err) && !test.wantPodEvicted {
				t.Fatalf("Test Failed: unexpected eviction of pod %q", klog.KObj(testPod))
			}

			testPod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, testPod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Test Failed: error: %q, while getting updated pod", err)
			}
			_, cond := podutil.GetPodCondition(&testPod.Status, v1.DisruptionTarget)
			if test.wantPodEvicted && cond == nil {
				t.Errorf("Pod %q does not have the expected condition: %q", klog.KObj(testPod), v1.DisruptionTarget)
			} else if !test.wantPodEvicted && cond != nil {
				t.Errorf("Pod %q has an unexpected condition: %q", klog.KObj(testPod), v1.DisruptionTarget)
			}
		})
	}
}

// TestTaintBasedEvictions tests related cases for the TaintBasedEvictions feature
func TestTaintBasedEvictions(t *testing.T) {
	// we need at least 2 nodes to prevent lifecycle manager from entering "fully-disrupted" mode
	nodeCount := 3
	nodeIndex := 1 // the exact node doesn't matter, pick one
	zero := int64(0)
	gracePeriod := int64(1)
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testpod1", DeletionGracePeriodSeconds: &zero},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "container", Image: imageutils.GetPauseImageName()},
			},
			Tolerations: []v1.Toleration{
				{
					Key:      v1.TaintNodeNotReady,
					Operator: v1.TolerationOpExists,
					Effect:   v1.TaintEffectNoExecute,
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	tests := []struct {
		name                        string
		nodeTaints                  []v1.Taint
		nodeConditions              []v1.NodeCondition
		pod                         *v1.Pod
		tolerationSeconds           int64
		expectedWaitForPodCondition string
	}{
		{
			name:                        "Taint based evictions for NodeNotReady and 200 tolerationseconds",
			nodeTaints:                  []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			nodeConditions:              []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			pod:                         testPod.DeepCopy(),
			tolerationSeconds:           200,
			expectedWaitForPodCondition: "updated with tolerationSeconds of 200",
		},
		{
			name:           "Taint based evictions for NodeNotReady with no pod tolerations",
			nodeTaints:     []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			nodeConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "testpod1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container", Image: imageutils.GetPauseImageName()},
					},
				},
			},
			tolerationSeconds:           300,
			expectedWaitForPodCondition: "updated with tolerationSeconds=300",
		},
		{
			name:                        "Taint based evictions for NodeNotReady and 0 tolerationseconds",
			nodeTaints:                  []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			nodeConditions:              []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			pod:                         testPod.DeepCopy(),
			tolerationSeconds:           0,
			expectedWaitForPodCondition: "terminating",
		},
		{
			name:           "Taint based evictions for NodeUnreachable",
			nodeTaints:     []v1.Taint{{Key: v1.TaintNodeUnreachable, Effect: v1.TaintEffectNoExecute}},
			nodeConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionUnknown}},
		},
	}

	// Build admission chain handler.
	podTolerations := podtolerationrestriction.NewPodTolerationsPlugin(&pluginapi.Configuration{})
	defaultTolerationSeconds, err := newHandlerForTest()
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	admission := admission.NewChainHandler(
		podTolerations,
		defaultTolerationSeconds,
	)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testCtx := testutils.InitTestAPIServer(t, "taint-based-evictions", admission)

			// Build clientset and informers for controllers.
			externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
			externalClientConfig.QPS = -1
			externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
			externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)
			podTolerations.SetExternalKubeClientSet(externalClientset)
			podTolerations.SetExternalKubeInformerFactory(externalInformers)

			cs := testCtx.ClientSet

			// Start NodeLifecycleController for taint.
			nc, err := nodelifecycle.NewNodeLifecycleController(
				testCtx.Ctx,
				externalInformers.Coordination().V1().Leases(),
				externalInformers.Core().V1().Pods(),
				externalInformers.Core().V1().Nodes(),
				externalInformers.Apps().V1().DaemonSets(),
				cs,
				1*time.Second,    // Node monitor grace period
				time.Minute,      // Node startup grace period
				time.Millisecond, // Node monitor period
				100,              // Eviction limiter QPS
				100,              // Secondary eviction limiter QPS
				50,               // Large cluster threshold
				0.55,             // Unhealthy zone threshold
			)
			if err != nil {
				t.Fatalf("Failed to create node controller: %v", err)
			}

			// Waiting for all controllers to sync
			externalInformers.Start(testCtx.Ctx.Done())
			externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

			// Run the controller
			go nc.Run(testCtx.Ctx)

			// Start TaintEvictionController
			tm, _ := tainteviction.New(
				testCtx.Ctx,
				testCtx.ClientSet,
				externalInformers.Core().V1().Pods(),
				externalInformers.Core().V1().Nodes(),
				names.TaintEvictionController,
			)
			go tm.Run(testCtx.Ctx)

			nodeRes := v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4000m"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			}

			var nodes []*v1.Node
			for i := range nodeCount {
				node := &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("node-%d", i),
						Labels: map[string]string{
							v1.LabelTopologyRegion:                  "region1",
							v1.LabelTopologyZone:                    "zone1",
							"node.kubernetes.io/exclude-disruption": "true",
						},
					},
					Spec: v1.NodeSpec{},
					Status: v1.NodeStatus{
						Capacity:    nodeRes,
						Allocatable: nodeRes,
					},
				}
				if i == nodeIndex {
					node.Status.Conditions = append(node.Status.Conditions, test.nodeConditions...)
				} else {
					node.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
						Type:   v1.NodeReady,
						Status: v1.ConditionTrue,
					})
				}
				nodes = append(nodes, node)
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node: %q, err: %v", klog.KObj(node), err)
				}
			}

			if test.pod != nil {
				test.pod.Spec.NodeName = nodes[nodeIndex].Name
				test.pod.Name = "testpod"
				if len(test.pod.Spec.Tolerations) > 0 {
					test.pod.Spec.Tolerations[0].TolerationSeconds = &test.tolerationSeconds
				}

				test.pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, test.pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Test Failed: error: %q, while creating pod %q", err, klog.KObj(test.pod))
				}
			}

			if err := testutils.WaitForNodeTaints(testCtx.Ctx, cs, nodes[nodeIndex], test.nodeTaints); err != nil {
				t.Errorf("Failed to taint node %q, err: %v", klog.KObj(nodes[nodeIndex]), err)
			}

			if test.pod != nil {
				err = wait.PollImmediate(time.Second, time.Second*15, func() (bool, error) {
					pod, err := cs.CoreV1().Pods(test.pod.Namespace).Get(testCtx.Ctx, test.pod.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					// as node is unreachable, pod0 is expected to be in Terminating status
					// rather than getting deleted
					if test.tolerationSeconds == 0 {
						return pod.DeletionTimestamp != nil, nil
					}
					if seconds, err := testutils.GetTolerationSeconds(pod.Spec.Tolerations); err == nil {
						return seconds == test.tolerationSeconds, nil
					}
					return false, nil
				})
				if err != nil {
					pod, _ := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, test.pod.Name, metav1.GetOptions{})
					t.Fatalf("Error: %v, Expected test pod to be %s but it's %v", err, test.expectedWaitForPodCondition, pod)
				}
				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{test.pod})
			}
			testutils.CleanupNodes(cs, t)
		})
	}
}

// newHandlerForTest returns a handler configured for testing.
func newHandlerForTest() (*defaulttolerationseconds.Plugin, error) {
	handler := defaulttolerationseconds.NewDefaultTolerationSeconds()
	pluginInitializer := initializer.New(nil, nil, nil, nil, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	return handler, admission.ValidateInitialization(handler)
}

// failDeleteWrapper is a client-side HTTP round-trip wrapper used by
// TestTaintEvictionDurableRetryEndToEnd to inject exactly maxFail transient
// DELETE failures for a specific pod, then allow subsequent requests through.
//
// During the burst phase (attempts < maxFail):
//   - GET  for the pod path → returns a synthetic 200 with the cached pod JSON,
//     so addConditionAndDeletePod completes its internal Get without a live
//     round-trip and the burst loop runs in microseconds.
//   - PATCH to the pod status subresource → same synthetic 200, so
//     PatchPodStatus succeeds instantly without writing to etcd.
//   - DELETE → returns a transient error and increments the failure counter.
//
// After maxFail DELETEs the wrapper passes every request to the real server,
// so the durable retry's Get/Patch/Delete all hit the real API server and
// the real pod is deleted. successfulDeletes records when such a DELETE has
// received its response. The test must wait for that response before returning:
// observing a deletion timestamp alone can race the apiserver's completion of
// the DELETE request, and test cleanup cancels testCtx.Ctx.
//
// Keeping GET and PATCH synthetic during the burst phase is essential for
// Prow stability: without it, 5 × (real GET + real PATCH) can consume the
// test's entire time budget and leave the retry's Delete with a cancelled ctx.
type failDeleteWrapper struct {
	// podPath matches the URL path substring for the target pod.
	// Example: "/pods/test-pod". Matches both "/pods/test-pod" and
	// "/pods/test-pod/status" (for PATCH).
	podPath string
	// maxFail is the number of DELETE requests to reject before allowing through.
	maxFail int32
	// attempts counts every DELETE request that targets podPath.
	attempts atomic.Int32
	// successfulDeletes counts real DELETE requests that completed successfully.
	successfulDeletes atomic.Int32
	// podJSON holds a JSON-encoded copy of the created pod, used to serve
	// synthetic GET and PATCH responses during the burst phase.
	// Written once before the controller starts; safe to read concurrently.
	podJSON []byte
}

// completionTrackingBody records a successful request only after its response
// body has been consumed. http.RoundTripper returns as soon as response headers
// arrive, which is too early for a test that must not cancel the request while
// the apiserver is still writing its response.
type completionTrackingBody struct {
	io.ReadCloser
	onComplete func()
	once       sync.Once
}

func (b *completionTrackingBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		b.once.Do(b.onComplete)
	}
	return n, err
}

// setPod caches a JSON-encoded snapshot of pod for use in synthetic responses.
// Must be called after pod creation and before the controller starts.
func (w *failDeleteWrapper) setPod(pod *v1.Pod) error {
	b, err := json.Marshal(pod)
	if err != nil {
		return err
	}
	w.podJSON = b
	return nil
}

// syntheticOK returns an *http.Response with status 200 and body b.
func syntheticOK(b []byte) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(b)),
	}
}

// roundTrip implements testutils.RoundTripWrapper.
func (w *failDeleteWrapper) roundTrip(inner http.RoundTripper, req *http.Request) (*http.Response, error) {
	if !strings.Contains(req.URL.Path, w.podPath) {
		return inner.RoundTrip(req)
	}
	// inBurst is true after the first DELETE has been counted and while
	// fewer than maxFail DELETEs have been seen. During this window, GET and
	// PATCH for the target pod are served from the cached JSON so that the
	// burst loop iterations 2–maxFail complete without live server round-trips.
	// Iteration 1's GET/PATCH pass through (attempts==0) so the DisruptionTarget
	// condition is genuinely written to etcd on the first attempt.
	cur := w.attempts.Load()
	inBurst := cur >= 1 && cur < w.maxFail
	switch req.Method {
	case http.MethodDelete:
		n := w.attempts.Add(1)
		if n <= w.maxFail {
			// Transient failure: controller burst loop re-tries, eventually
			// calls addPodEvictionRetry and enqueues the durable retry item.
			return nil, fmt.Errorf("injected transient failure for DELETE attempt %d", n)
		}
		// Attempt 6+: pass to the real server so the pod is actually deleted.
		resp, err := inner.RoundTrip(req)
		if err == nil && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			resp.Body = &completionTrackingBody{
				ReadCloser: resp.Body,
				onComplete: func() {
					w.successfulDeletes.Add(1)
				},
			}
		}
		return resp, err
	case http.MethodGet:
		if inBurst && len(w.podJSON) > 0 {
			// Return the cached pod so addConditionAndDeletePod can read the
			// UID and condition list without a live API call.
			return syntheticOK(w.podJSON), nil
		}
	case http.MethodPatch:
		if inBurst && len(w.podJSON) > 0 {
			// Return the cached pod as the PATCH result so PatchPodStatus
			// succeeds instantly without writing to etcd.
			return syntheticOK(w.podJSON), nil
		}
	}
	return inner.RoundTrip(req)
}

// TestTaintEvictionDurableRetryEndToEnd exercises the taint-eviction controller
// end-to-end against a real API server. It verifies that:
//
//  1. A pod with a finite TolerationSeconds window is NOT evicted immediately
//     when a NoExecute taint is added to its node.
//  2. The pod IS evicted after the toleration window expires.
//  3. When the initial burst of Delete requests fails, the durable retry queue
//     is correctly utilized to eventually delete the pod.
//
// This covers the timed-worker → durable-retry chain introduced by
// https://github.com/kubernetes/kubernetes/pull/141568. The test uses the
// real tainteviction.Controller and a live API server so that transient API
// failures and the rate-limited retry queue are exercised under realistic
// conditions.
func TestTaintEvictionDurableRetryEndToEnd(t *testing.T) {
	// Use a short toleration window so the test completes quickly.
	tolerationSeconds := int64(2)
	noExecuteTaint := v1.Taint{
		Key:    "test/noexecute",
		Value:  "true",
		Effect: v1.TaintEffectNoExecute,
	}

	// Start the real test API server without any custom admission plugin.
	// Failure injection is handled client-side via the RoundTrip wrapper below,
	// which is simpler and avoids races between admission state and the retry queue.
	testCtx := testutils.InitTestAPIServer(t, "taint-eviction-retry", nil)
	cs := testCtx.ClientSet

	// Install a client-side transport wrapper that:
	//   - Rejects the first 5 DELETE requests for the pod with a transient error,
	//     so all 5 burst attempts in deletePodHandler fail and addPodEvictionRetry
	//     is called, exercising the durable retry queue.
	//   - Intercepts GET and PATCH for burst iterations 2–5 with synthetic
	//     responses, so those iterations complete in microseconds rather than
	//     making live round-trips that could consume the test's entire time budget.
	//   - Passes every request (GET, PATCH, DELETE) through to the real API server
	//     after maxFail DELETEs, so the retry queue's final deletion is real.
	deleteWrapper := &failDeleteWrapper{
		podPath: "/pods/test-pod",
		maxFail: 5,
	}
	fn := testutils.RoundTripWrapper(deleteWrapper.roundTrip)
	testCtx.RoundTrip.Store(&fn)

	// Build informers and controller using the same clientset.
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, 0)

	if err := controller.AddPodNodeNameIndexer(
		externalInformers.Core().V1().Pods().Informer(),
	); err != nil {
		t.Fatalf("Failed to add Pod node name indexer: %v", err)
	}

	tm, err := tainteviction.New(
		testCtx.Ctx,
		testCtx.ClientSet,
		externalInformers.Core().V1().Pods(),
		externalInformers.Core().V1().Nodes(),
		"taint-eviction-retry-test",
	)
	if err != nil {
		t.Fatalf("Failed to create taint eviction controller: %v", err)
	}

	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())
	go tm.Run(testCtx.Ctx)

	// Create a Ready node.
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
			}},
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
				v1.ResourcePods:   resource.MustParse("10"),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
				v1.ResourcePods:   resource.MustParse("10"),
			},
		},
	}
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}
	t.Cleanup(func() { testutils.CleanupNodes(cs, t) })

	// Create a pod assigned to the node with a finite toleration for the taint.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: testCtx.NS.Name,
		},
		Spec: v1.PodSpec{
			NodeName: node.Name,
			Containers: []v1.Container{
				{Name: "c", Image: "pause"},
			},
			Tolerations: []v1.Toleration{{
				Key:               noExecuteTaint.Key,
				Operator:          v1.TolerationOpExists,
				Effect:            v1.TaintEffectNoExecute,
				TolerationSeconds: &tolerationSeconds,
			}},
		},
	}
	createdPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	// Cache the created pod's JSON so the wrapper can serve synthetic GET/PATCH
	// responses during the burst phase (iterations 2-5) without hitting etcd.
	if err := deleteWrapper.setPod(createdPod); err != nil {
		t.Fatalf("Failed to cache pod JSON: %v", err)
	}
	// The controller's deletion is the result under test. On an early failure,
	// InitTestAPIServer's namespace cleanup removes the pod with a fresh context.

	// Add a NoExecute taint. The controller should start the toleration timer.
	if err := testutils.AddTaintToNode(cs, node.Name, noExecuteTaint); err != nil {
		t.Fatalf("Failed to add taint to node: %v", err)
	}

	// Poll briefly to confirm the pod is NOT evicted immediately.
	// It has a 2-second toleration window so deletion timestamp must not be
	// set within the first 500ms.
	evictedEarly := false
	_ = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 500*time.Millisecond, false,
		func(ctx context.Context) (bool, error) {
			p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, createdPod.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			if p.DeletionTimestamp != nil {
				evictedEarly = true
				return true, nil
			}
			return false, nil
		},
	)
	if evictedEarly {
		t.Error("Pod was evicted before its TolerationSeconds window expired")
	}

	// The toleration window is 2 seconds. When it expires, the controller will
	// attempt to delete the pod 5 times in a quick burst. Our transport wrapper
	// will reject all 5 attempts with a transient network error. The controller
	// will then hand off the eviction to the rate-limited durable retry queue
	// (podEvictionQueue). We wait up to 30 seconds for the eventual successful
	// deletion (the 6th attempt or later) via the durable retry queue. Wait for
	// the real DELETE round trip to finish, not merely for DeletionTimestamp:
	// the latter is stored before the handler returns, while test cleanup cancels
	// testCtx.Ctx as soon as this test function returns.
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 30*time.Second, true,
		func(context.Context) (bool, error) {
			return deleteWrapper.successfulDeletes.Load() > 0, nil
		}); err != nil {
		t.Errorf("Pod was not deleted by a completed durable retry within expected window: %v", err)
	}

	// Verify that the transport wrapper actually saw at least 6 DELETE attempts:
	// the initial 5 that were rejected plus at least 1 that was allowed and
	// resulted in the pod being evicted. This confirms the durable retry queue
	// was exercised end-to-end.
	totalAttempts := deleteWrapper.attempts.Load()
	if totalAttempts < 6 {
		t.Errorf("Expected at least 6 Delete attempts (5 failures + 1 success via retry queue), got %d", totalAttempts)
	}
	if completedDeletes := deleteWrapper.successfulDeletes.Load(); completedDeletes == 0 {
		t.Error("Expected a real Delete request to complete successfully via the durable retry queue")
	}
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, true,
		testutils.PodIsGettingEvicted(cs, testCtx.NS.Name, createdPod.Name)); err != nil {
		t.Errorf("Pod did not reflect the completed durable retry deletion: %v", err)
	}
}
