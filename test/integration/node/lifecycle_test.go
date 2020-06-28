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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// poll is how often to poll pods, nodes and claims.
const poll = 2 * time.Second

type podCondition func(pod *v1.Pod) (bool, error)

// TestTaintBasedEvictions tests related cases for the TaintBasedEvictions feature
func TestTaintBasedEvictions(t *testing.T) {
	// we need at least 2 nodes to prevent lifecycle manager from entering "fully-disrupted" mode
	nodeCount := 3
	zero := int64(0)
	gracePeriod := int64(1)
	heartbeatInternal := time.Second * 2
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
	tolerationSeconds := []int64{200, 300, 0}
	tests := []struct {
		name                        string
		nodeTaints                  []v1.Taint
		nodeConditions              []v1.NodeCondition
		pod                         *v1.Pod
		expectedWaitForPodCondition string
	}{
		{
			name:                        "Taint based evictions for NodeNotReady and 200 tolerationseconds",
			nodeTaints:                  []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			nodeConditions:              []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			pod:                         testPod,
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
			expectedWaitForPodCondition: "updated with tolerationSeconds=300",
		},
		{
			name:                        "Taint based evictions for NodeNotReady and 0 tolerationseconds",
			nodeTaints:                  []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			nodeConditions:              []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			pod:                         testPod,
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
	admission := admission.NewChainHandler(
		podTolerations,
		defaulttolerationseconds.NewDefaultTolerationSeconds(),
	)
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testCtx := testutils.InitTestMaster(t, "taint-based-evictions", admission)

			// Build clientset and informers for controllers.
			externalClientset := kubernetes.NewForConfigOrDie(&restclient.Config{
				QPS:           -1,
				Host:          testCtx.HTTPServer.URL,
				ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)
			podTolerations.SetExternalKubeClientSet(externalClientset)
			podTolerations.SetExternalKubeInformerFactory(externalInformers)

			testCtx = testutils.InitTestScheduler(t, testCtx, true, nil)
			defer testutils.CleanupTest(t, testCtx)
			cs := testCtx.ClientSet
			_, err := cs.CoreV1().Namespaces().Create(context.TODO(), testCtx.NS, metav1.CreateOptions{})
			if err != nil {
				t.Errorf("Failed to create namespace %+v", err)
			}

			// Start NodeLifecycleController for taint.
			nc, err := nodelifecycle.NewNodeLifecycleController(
				externalInformers.Coordination().V1().Leases(),
				externalInformers.Core().V1().Pods(),
				externalInformers.Core().V1().Nodes(),
				externalInformers.Apps().V1().DaemonSets(),
				cs,
				5*time.Second,    // Node monitor grace period
				time.Minute,      // Node startup grace period
				time.Millisecond, // Node monitor period
				time.Second,      // Pod eviction timeout
				100,              // Eviction limiter QPS
				100,              // Secondary eviction limiter QPS
				50,               // Large cluster threshold
				0.55,             // Unhealthy zone threshold
				true,             // Run taint manager
			)
			if err != nil {
				t.Errorf("Failed to create node controller: %v", err)
				return
			}

			// Waiting for all controllers to sync
			externalInformers.Start(testCtx.Ctx.Done())
			externalInformers.WaitForCacheSync(testCtx.Ctx.Done())
			testutils.SyncInformerFactory(testCtx)

			// Run all controllers
			go nc.Run(testCtx.Ctx.Done())
			go testCtx.Scheduler.Run(testCtx.Ctx)

			nodeRes := v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4000m"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			}

			var nodes []*v1.Node
			for i := 0; i < nodeCount; i++ {
				nodes = append(nodes, &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name:   fmt.Sprintf("node-%d", i),
						Labels: map[string]string{v1.LabelZoneRegion: "region1", v1.LabelZoneFailureDomain: "zone1"},
					},
					Spec: v1.NodeSpec{},
					Status: v1.NodeStatus{
						Capacity:    nodeRes,
						Allocatable: nodeRes,
						Conditions: []v1.NodeCondition{
							{
								Type:              v1.NodeReady,
								Status:            v1.ConditionTrue,
								LastHeartbeatTime: metav1.Now(),
							},
						},
					},
				})
				if _, err := cs.CoreV1().Nodes().Create(context.TODO(), nodes[i], metav1.CreateOptions{}); err != nil {
					t.Errorf("Failed to create node, err: %v", err)
				}
			}

			neededNode := nodes[1]
			if test.pod != nil {
				test.pod.Name = fmt.Sprintf("testpod-%d", i)
				if len(test.pod.Spec.Tolerations) > 0 {
					test.pod.Spec.Tolerations[0].TolerationSeconds = &tolerationSeconds[i]
				}

				test.pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(context.TODO(), test.pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Test Failed: error: %v, while creating pod", err)
				}

				if err := testutils.WaitForPodToSchedule(cs, test.pod); err != nil {
					t.Errorf("Failed to schedule pod %s/%s on the node, err: %v",
						test.pod.Namespace, test.pod.Name, err)
				}
				test.pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(context.TODO(), test.pod.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Test Failed: error: %v, while creating pod", err)
				}
				neededNode, err = cs.CoreV1().Nodes().Get(context.TODO(), test.pod.Spec.NodeName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Error while getting node associated with pod %v with err %v", test.pod.Name, err)
				}
			}

			// Regularly send heartbeat event to APIServer so that the cluster doesn't enter fullyDisruption mode.
			// TODO(Huang-Wei): use "NodeDisruptionExclusion" feature to simply the below logic when it's beta.
			for i := 0; i < nodeCount; i++ {
				var conditions []v1.NodeCondition
				// If current node is not <neededNode>
				if neededNode.Name != nodes[i].Name {
					conditions = []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
						},
					}
				} else {
					c, err := testutils.NodeReadyStatus(test.nodeConditions)
					if err != nil {
						t.Error(err)
					}
					// Need to distinguish NodeReady/False and NodeReady/Unknown.
					// If we try to update the node with condition NotReady/False, i.e. expect a NotReady:NoExecute taint
					// we need to keep sending the update event to keep it alive, rather than just sending once.
					if c == v1.ConditionFalse {
						conditions = test.nodeConditions
					} else if c == v1.ConditionUnknown {
						// If it's expected to update the node with condition NotReady/Unknown,
						// i.e. expect a Unreachable:NoExecute taint,
						// we need to only send the update event once to simulate the network unreachable scenario.
						nodeCopy := testutils.NodeCopyWithConditions(nodes[i], test.nodeConditions)
						if err := testutils.UpdateNodeStatus(cs, nodeCopy); err != nil && !apierrors.IsNotFound(err) {
							t.Errorf("Cannot update node: %v", err)
						}
						continue
					}
				}
				// Keeping sending NodeReady/True or NodeReady/False events.
				go func(i int) {
					for {
						select {
						case <-testCtx.Ctx.Done():
							return
						case <-time.Tick(heartbeatInternal):
							nodeCopy := testutils.NodeCopyWithConditions(nodes[i], conditions)
							if err := testutils.UpdateNodeStatus(cs, nodeCopy); err != nil && !apierrors.IsNotFound(err) {
								t.Errorf("Cannot update node: %v", err)
							}
						}
					}
				}(i)
			}

			if err := testutils.WaitForNodeTaints(cs, neededNode, test.nodeTaints); err != nil {
				t.Errorf("Failed to taint node in test %d <%s>, err: %v", i, neededNode.Name, err)
			}

			if test.pod != nil {
				err = waitForPodCondition(cs, testCtx.NS.Name, test.pod.Name, test.expectedWaitForPodCondition, time.Second*15, func(pod *v1.Pod) (bool, error) {
					// as node is unreachable, pod0 is expected to be in Terminating status
					// rather than getting deleted
					if tolerationSeconds[i] == 0 {
						return pod.DeletionTimestamp != nil, nil
					}
					if seconds, err := testutils.GetTolerationSeconds(pod.Spec.Tolerations); err == nil {
						return seconds == tolerationSeconds[i], nil
					}
					return false, nil
				}, t)
				if err != nil {
					pod, _ := cs.CoreV1().Pods(testCtx.NS.Name).Get(context.TODO(), test.pod.Name, metav1.GetOptions{})
					t.Fatalf("Error: %v, Expected test pod to be %s but it's %v", err, test.expectedWaitForPodCondition, pod)
				}
				testutils.CleanupPods(cs, t, []*v1.Pod{test.pod})
			}
			testutils.CleanupNodes(cs, t)
			testutils.WaitForSchedulerCacheCleanup(testCtx.Scheduler, t)
		})
	}
}

// waitForPodCondition waits a pods to be matched to the given condition.
func waitForPodCondition(c clientset.Interface, ns, podName, desc string, timeout time.Duration, condition podCondition, t *testing.T) error {
	t.Logf("Waiting up to %v for pod %q in namespace %q to be %q", timeout, podName, ns, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				t.Logf("Pod %q in namespace %q not found. Error: %v", podName, ns, err)
				return err
			}
			t.Logf("Get pod %q in namespace %q failed, ignoring for %v. Error: %v", podName, ns, poll, err)
			continue
		}
		// log now so that current pod info is reported before calling `condition()`
		t.Logf("Pod %q: Phase=%q, Reason=%q, readiness=%t. Elapsed: %v",
			podName, pod.Status.Phase, pod.Status.Reason, podutil.IsPodReady(pod), time.Since(start))
		if done, err := condition(pod); done {
			if err == nil {
				t.Logf("Pod %q satisfied condition %q", podName, desc)
			}
			return err
		}
	}
	return fmt.Errorf("gave up after waiting %v for pod %q to be %q", timeout, podName, desc)
}
