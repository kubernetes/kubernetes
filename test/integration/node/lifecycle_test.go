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
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	"k8s.io/kubernetes/pkg/features"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
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
			nc, err := nodelifecycle.NewNodeLifecycleController(testCtx.Ctx, nodelifecycle.ControllerOptions{
				LeaseInformer:               externalInformers.Coordination().V1().Leases(),
				PodInformer:                 externalInformers.Core().V1().Pods(),
				NodeInformer:                externalInformers.Core().V1().Nodes(),
				DaemonSetInformer:           externalInformers.Apps().V1().DaemonSets(),
				KubeClient:                  cs,
				NodeMonitorPeriod:           1 * time.Second,
				NodeStartupGracePeriod:      time.Minute,
				NodeMonitorGracePeriod:      time.Millisecond,
				EvictionLimiterQPS:          100,
				SecondaryEvictionLimiterQPS: 100,
				LargeClusterThreshold:       50,
				UnhealthyZoneThreshold:      0.55,
			})
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
			nc, err := nodelifecycle.NewNodeLifecycleController(testCtx.Ctx, nodelifecycle.ControllerOptions{
				LeaseInformer:               externalInformers.Coordination().V1().Leases(),
				PodInformer:                 externalInformers.Core().V1().Pods(),
				NodeInformer:                externalInformers.Core().V1().Nodes(),
				DaemonSetInformer:           externalInformers.Apps().V1().DaemonSets(),
				KubeClient:                  cs,
				NodeMonitorPeriod:           1 * time.Second,
				NodeStartupGracePeriod:      time.Minute,
				NodeMonitorGracePeriod:      time.Millisecond,
				EvictionLimiterQPS:          100,
				SecondaryEvictionLimiterQPS: 100,
				LargeClusterThreshold:       50,
				UnhealthyZoneThreshold:      0.55,
			})
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

func setPodReadyStatus(ctx context.Context, cs clientset.Interface, namespace, name string, status v1.ConditionStatus) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pod, err := cs.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		pod.Status.Phase = v1.PodRunning
		pod.Status.Conditions = []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: status,
			},
		}
		_, err = cs.CoreV1().Pods(namespace).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
		return err
	})
}

func updateNodeReadyCondition(ctx context.Context, cs clientset.Interface, nodeName string, cond v1.NodeCondition) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updated := false
		for i := range node.Status.Conditions {
			if node.Status.Conditions[i].Type == v1.NodeReady {
				node.Status.Conditions[i] = cond
				updated = true
				break
			}
		}
		if !updated {
			node.Status.Conditions = append(node.Status.Conditions, cond)
		}
		_, err = cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
		return err
	})
}

func hasPodReadyStatus(ctx context.Context, cs clientset.Interface, namespace, name string, want v1.ConditionStatus) (bool, error) {
	pod, err := cs.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return false, err
	}
	_, cond := podutil.GetPodCondition(&pod.Status, v1.PodReady)
	return cond != nil && cond.Status == want, nil
}

// TestExternalNodeLivenessSource verifies NodeLifecycleController in "external" mode:
// an external writer marking one node Ready=Unknown (without changing LastHeartbeatTime)
// causes that node's pods to lose readiness and the unreachable taint to be applied while
// leaving other lease-less nodes unaffected; writing Ready=True recovers the node with
// exactly one Node-scoped NodeNotReady event recorded.
func TestExternalNodeLivenessSource(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)

	const (
		nodeCount       = 3
		targetNodeIndex = 1
		monitorPeriod   = 200 * time.Millisecond
	)

	testCtx := testutils.InitTestAPIServer(t, "external-node-liveness", nil)
	cs := testCtx.ClientSet

	// metav1.Time serializes at whole-second RFC3339 granularity, so step timestamps by whole seconds.
	initialTime := metav1.NewTime(time.Now().UTC().Truncate(time.Second).Add(-10 * time.Second))

	nodes := make([]*v1.Node, nodeCount)
	pods := make([]*v1.Pod, nodeCount)
	for i := range nodeCount {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   fmt.Sprintf("ext-node-%d", i),
				Labels: map[string]string{"node.kubernetes.io/exclude-disruption": "true"},
			},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						Reason:             "KubeletReady",
						LastHeartbeatTime:  initialTime,
						LastTransitionTime: initialTime,
					},
				},
			},
		}
		createdNode, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create node %d: %v", i, err)
		}
		nodes[i] = createdNode

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("ext-pod-%d", i),
			},
			Spec: v1.PodSpec{
				NodeName: createdNode.Name,
				Containers: []v1.Container{
					{Name: "container", Image: imageutils.GetPauseImageName()},
				},
			},
		}
		createdPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod %d: %v", i, err)
		}
		// Pod status is cleared on Create by podStrategy; set PodReady=True via UpdateStatus.
		if err := setPodReadyStatus(testCtx.Ctx, cs, testCtx.NS.Name, createdPod.Name, v1.ConditionTrue); err != nil {
			t.Fatalf("failed to set PodReady=True on pod %d: %v", i, err)
		}
		pods[i] = createdPod
	}

	ctrlCtx, cancelCtrl := context.WithCancel(testCtx.Ctx)
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)

	nc, err := nodelifecycle.NewNodeLifecycleController(ctrlCtx, nodelifecycle.ControllerOptions{
		LeaseInformer:               externalInformers.Coordination().V1().Leases(),
		PodInformer:                 externalInformers.Core().V1().Pods(),
		NodeInformer:                externalInformers.Core().V1().Nodes(),
		DaemonSetInformer:           externalInformers.Apps().V1().DaemonSets(),
		KubeClient:                  cs,
		NodeMonitorPeriod:           monitorPeriod,
		NodeStartupGracePeriod:      time.Minute,
		NodeMonitorGracePeriod:      time.Millisecond, // Immediately expired; external mode must ignore it.
		EvictionLimiterQPS:          100,
		SecondaryEvictionLimiterQPS: 100,
		LargeClusterThreshold:       50,
		UnhealthyZoneThreshold:      0.55,
		NodeLivenessSource:          nodelifecycleconfig.NodeLivenessSourceExternal,
	})
	if err != nil {
		cancelCtrl()
		t.Fatalf("failed to create node lifecycle controller: %v", err)
	}

	externalInformers.Start(ctrlCtx.Done())
	externalInformers.WaitForCacheSync(ctrlCtx.Done())
	ctrlDone := make(chan struct{})
	go func() {
		defer close(ctrlDone)
		nc.Run(ctrlCtx)
	}()
	defer func() {
		cancelCtrl()
		<-ctrlDone
		externalInformers.Shutdown()
	}()

	// Wait until the controller has registered all three nodes on its initial pass.
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		events, err := cs.CoreV1().Events("").List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		registered := make(map[string]bool, nodeCount)
		for _, ev := range events.Items {
			if ev.InvolvedObject.Kind == "Node" && ev.Reason == "RegisteredNode" {
				registered[ev.InvolvedObject.Name] = true
			}
		}
		return len(registered) == nodeCount, nil
	}); err != nil {
		t.Fatalf("timed out waiting for controller to register nodes: %v", err)
	}
	// Allow an extra monitor pass so lastReady=True is recorded and confirm expired
	// NodeMonitorGracePeriod with no Leases did not mark any node Unknown.
	time.Sleep(2 * monitorPeriod)
	for i := range nodeCount {
		ok, err := hasPodReadyStatus(testCtx.Ctx, cs, testCtx.NS.Name, pods[i].Name, v1.ConditionTrue)
		if err != nil || !ok {
			t.Fatalf("expected pod %s to remain Ready before external transition (ok=%v, err=%v)", pods[i].Name, ok, err)
		}
	}

	// Simulate an external detector setting Ready=Unknown on targetNode without changing LastHeartbeatTime.
	targetNode := nodes[targetNodeIndex]
	targetPod := pods[targetNodeIndex]
	unknownTransitionTime := metav1.NewTime(initialTime.Add(time.Second))
	if err := updateNodeReadyCondition(testCtx.Ctx, cs, targetNode.Name, v1.NodeCondition{
		Type:               v1.NodeReady,
		Status:             v1.ConditionUnknown,
		Reason:             "ExternalDetectorUnreachable",
		Message:            "External detector lost contact with node.",
		LastHeartbeatTime:  initialTime,
		LastTransitionTime: unknownTransitionTime,
	}); err != nil {
		t.Fatalf("failed to write external Ready=Unknown on %s: %v", targetNode.Name, err)
	}

	// Target node's pod should lose readiness and the unreachable taint should be applied.
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		return hasPodReadyStatus(ctx, cs, testCtx.NS.Name, targetPod.Name, v1.ConditionFalse)
	}); err != nil {
		t.Fatalf("timed out waiting for %s to be marked NotReady: %v", targetPod.Name, err)
	}
	if err := testutils.WaitForNodeTaints(testCtx.Ctx, cs, targetNode, []v1.Taint{*nodelifecycle.UnreachableTaintTemplate}); err != nil {
		t.Fatalf("timed out waiting for unreachable taint on %s: %v", targetNode.Name, err)
	}

	// Other nodes (also without Leases) and their pods must remain Ready and untainted.
	for i := range nodeCount {
		if i == targetNodeIndex {
			continue
		}
		node, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodes[i].Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get node %s: %v", nodes[i].Name, err)
		}
		if taintutils.TaintExists(node.Spec.Taints, nodelifecycle.UnreachableTaintTemplate) {
			t.Errorf("unexpected unreachable taint on healthy node %s: %+v", node.Name, node.Spec.Taints)
		}
		ok, err := hasPodReadyStatus(testCtx.Ctx, cs, testCtx.NS.Name, pods[i].Name, v1.ConditionTrue)
		if err != nil || !ok {
			t.Errorf("expected pod %s on unaffected node %s to stay Ready (ok=%v, err=%v)", pods[i].Name, node.Name, ok, err)
		}
	}

	// Simulate recovery: Ready=True written back to targetNode.
	recoveryTransitionTime := metav1.NewTime(initialTime.Add(2 * time.Second))
	if err := updateNodeReadyCondition(testCtx.Ctx, cs, targetNode.Name, v1.NodeCondition{
		Type:               v1.NodeReady,
		Status:             v1.ConditionTrue,
		Reason:             "KubeletReady",
		Message:            "Kubelet is posting ready status.",
		LastHeartbeatTime:  initialTime,
		LastTransitionTime: recoveryTransitionTime,
	}); err != nil {
		t.Fatalf("failed to write Ready=True on %s: %v", targetNode.Name, err)
	}

	// Target node recovers (unreachable taints removed).
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		node, err := cs.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, taint := range node.Spec.Taints {
			if taint.Key == v1.TaintNodeUnreachable {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("timed out waiting for unreachable taint removal on %s: %v", targetNode.Name, err)
	}

	// Verify exactly one Node-scoped NodeNotReady event was recorded (with Count == 1),
	// and zero for the unaffected nodes.
	var targetNodeNotReadyEvents []*v1.Event
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		events, err := cs.CoreV1().Events("").List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		targetNodeNotReadyEvents = nil
		for i := range events.Items {
			ev := &events.Items[i]
			if ev.InvolvedObject.Kind != "Node" || ev.Reason != "NodeNotReady" {
				continue
			}
			if ev.InvolvedObject.Name != targetNode.Name {
				return false, fmt.Errorf("unexpected NodeNotReady event on unaffected node %s: %+v", ev.InvolvedObject.Name, ev)
			}
			targetNodeNotReadyEvents = append(targetNodeNotReadyEvents, ev)
		}
		return len(targetNodeNotReadyEvents) > 0, nil
	}); err != nil {
		t.Fatalf("failed waiting for Node-scoped NodeNotReady event on %s: %v", targetNode.Name, err)
	}
	if len(targetNodeNotReadyEvents) != 1 || targetNodeNotReadyEvents[0].Count != 1 {
		t.Fatalf("expected exactly 1 Node-scoped NodeNotReady event with Count=1 on %s, got %d (%+v)",
			targetNode.Name, len(targetNodeNotReadyEvents), targetNodeNotReadyEvents)
	}
}

// TestNodeLivenessWrongRolloutOrder verifies the KEP-6371 wrong-rollout-order failure mode:
// if kubelet Leases are disabled while NodeLifecycleController is still in "kubelet" mode,
// the controller repeatedly marks the lease-less node Ready=Unknown after NodeMonitorGracePeriod
// and the live kubelet repeatedly re-asserts Ready=True, causing the node to flap.
func TestNodeLivenessWrongRolloutOrder(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)

	const (
		nodeCount          = 3
		targetNodeIndex    = 1
		monitorPeriod      = 150 * time.Millisecond
		monitorGracePeriod = 500 * time.Millisecond
		flapCyclesToVerify = 2
	)

	testCtx := testutils.InitTestAPIServer(t, "node-liveness-wrong-rollout", nil)
	cs := testCtx.ClientSet

	heartbeatTime := metav1.NewTime(time.Now().UTC().Truncate(time.Second).Add(-10 * time.Second))

	nodes := make([]*v1.Node, nodeCount)
	for i := range nodeCount {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   fmt.Sprintf("flap-node-%d", i),
				Labels: map[string]string{"node.kubernetes.io/exclude-disruption": "true"},
			},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						Reason:             "KubeletReady",
						LastHeartbeatTime:  heartbeatTime,
						LastTransitionTime: heartbeatTime,
					},
				},
			},
		}
		createdNode, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create node %d: %v", i, err)
		}
		nodes[i] = createdNode
	}

	targetNode := nodes[targetNodeIndex]
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "flap-pod"},
		Spec: v1.PodSpec{
			NodeName: targetNode.Name,
			Containers: []v1.Container{
				{Name: "container", Image: imageutils.GetPauseImageName()},
			},
		},
	}
	createdPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}
	if err := setPodReadyStatus(testCtx.Ctx, cs, testCtx.NS.Name, createdPod.Name, v1.ConditionTrue); err != nil {
		t.Fatalf("failed to set initial PodReady=True: %v", err)
	}

	ctrlCtx, cancelCtrl := context.WithCancel(testCtx.Ctx)
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientset := clientset.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)

	nc, err := nodelifecycle.NewNodeLifecycleController(ctrlCtx, nodelifecycle.ControllerOptions{
		LeaseInformer:               externalInformers.Coordination().V1().Leases(),
		PodInformer:                 externalInformers.Core().V1().Pods(),
		NodeInformer:                externalInformers.Core().V1().Nodes(),
		DaemonSetInformer:           externalInformers.Apps().V1().DaemonSets(),
		KubeClient:                  cs,
		NodeMonitorPeriod:           monitorPeriod,
		NodeStartupGracePeriod:      time.Minute,
		NodeMonitorGracePeriod:      monitorGracePeriod,
		EvictionLimiterQPS:          100,
		SecondaryEvictionLimiterQPS: 100,
		LargeClusterThreshold:       50,
		UnhealthyZoneThreshold:      0.55,
		NodeLivenessSource:          nodelifecycleconfig.NodeLivenessSourceKubelet,
	})
	if err != nil {
		cancelCtrl()
		t.Fatalf("failed to create node lifecycle controller: %v", err)
	}

	externalInformers.Start(ctrlCtx.Done())
	externalInformers.WaitForCacheSync(ctrlCtx.Done())
	ctrlDone := make(chan struct{})
	go func() {
		defer close(ctrlDone)
		nc.Run(ctrlCtx)
	}()
	defer func() {
		cancelCtrl()
		<-ctrlDone
		externalInformers.Shutdown()
	}()

	for cycle := range flapCyclesToVerify {
		// 1. With no Lease renewed, kubelet-mode controller marks the node Ready=Unknown
		//    after NodeMonitorGracePeriod and marks its pod NotReady.
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
			node, err := cs.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			var readyUnknown bool
			for _, cond := range node.Status.Conditions {
				if cond.Type == v1.NodeReady && cond.Status == v1.ConditionUnknown && cond.Reason == "NodeStatusUnknown" {
					readyUnknown = true
					break
				}
			}
			if !readyUnknown {
				return false, nil
			}
			return hasPodReadyStatus(ctx, cs, testCtx.NS.Name, createdPod.Name, v1.ConditionFalse)
		}); err != nil {
			t.Fatalf("cycle %d: timed out waiting for controller to mark lease-less node Unknown and pod NotReady: %v", cycle, err)
		}

		// Confirm no Lease exists for the node.
		if _, err := cs.CoordinationV1().Leases(v1.NamespaceNodeLease).Get(testCtx.Ctx, targetNode.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
			t.Fatalf("cycle %d: expected NotFound for node lease, got err=%v", cycle, err)
		}

		// 2. Simulate the still-running kubelet (with Leases disabled) re-asserting Ready=True
		//    and reconciling its pod back to Ready=True, triggering the next flap cycle.
		heartbeatTime = metav1.NewTime(heartbeatTime.Add(time.Second))
		if err := updateNodeReadyCondition(testCtx.Ctx, cs, targetNode.Name, v1.NodeCondition{
			Type:               v1.NodeReady,
			Status:             v1.ConditionTrue,
			Reason:             "KubeletReady",
			LastHeartbeatTime:  heartbeatTime,
			LastTransitionTime: heartbeatTime,
		}); err != nil {
			t.Fatalf("cycle %d: failed to re-assert Ready=True from fake kubelet: %v", cycle, err)
		}
		if err := setPodReadyStatus(testCtx.Ctx, cs, testCtx.NS.Name, createdPod.Name, v1.ConditionTrue); err != nil {
			t.Fatalf("cycle %d: failed to reset PodReady=True: %v", cycle, err)
		}
	}
}
