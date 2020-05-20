/*
Copyright 2015 The Kubernetes Authors.

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

package scheduler

// This file tests the scheduler.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

type nodeMutationFunc func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

// TestSchedulerCreationFromConfigMap verifies that scheduler can be created
// from configurations provided by a ConfigMap object and then verifies that the
// configuration is applied correctly.
func TestSchedulerCreationFromConfigMap(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	ns := framework.CreateTestingNamespace("configmap", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	defer clientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	for i, test := range []struct {
		policy          string
		expectedPlugins map[string][]kubeschedulerconfig.Plugin
	}{
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "PodFitsResources"}
				],
				"priorities" : [
					{"name" : "ImageLocalityPriority", "weight" : 1}
				]
			}`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "ImageLocality", Weight: 1},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1"
			}`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "PodTopologySpread", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
				"ReservePlugin":   {{Name: "VolumeBinding"}},
				"UnreservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin":   {{Name: "VolumeBinding"}},
				"BindPlugin":      {{Name: "DefaultBinder"}},
				"PostBindPlugin":  {{Name: "VolumeBinding"}},
			},
		},
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [],
				"priorities" : []
			}`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		{
			policy: `apiVersion: v1
kind: Policy
predicates:
- name: PodFitsResources
priorities:
- name: ImageLocalityPriority
  weight: 1
`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "ImageLocality", Weight: 1},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
		{
			policy: `apiVersion: v1
kind: Policy
`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				},
				"PreScorePlugin": {
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
					{Name: "DefaultPodTopologySpread"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "PodTopologySpread", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
				"ReservePlugin":   {{Name: "VolumeBinding"}},
				"UnreservePlugin": {{Name: "VolumeBinding"}},
				"PreBindPlugin":   {{Name: "VolumeBinding"}},
				"BindPlugin":      {{Name: "DefaultBinder"}},
				"PostBindPlugin":  {{Name: "VolumeBinding"}},
			},
		},
		{
			policy: `apiVersion: v1
kind: Policy
predicates: []
priorities: []
`,
			expectedPlugins: map[string][]kubeschedulerconfig.Plugin{
				"QueueSortPlugin": {{Name: "PrioritySort"}},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "TaintToleration"},
				},
				"BindPlugin": {{Name: "DefaultBinder"}},
			},
		},
	} {
		// Add a ConfigMap object.
		configPolicyName := fmt.Sprintf("scheduler-custom-policy-config-%d", i)
		policyConfigMap := v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: configPolicyName},
			Data:       map[string]string{kubeschedulerconfig.SchedulerPolicyConfigMapKey: test.policy},
		}

		policyConfigMap.APIVersion = "v1"
		clientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &policyConfigMap, metav1.CreateOptions{})

		eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: clientSet.EventsV1beta1().Events("")})
		stopCh := make(chan struct{})
		eventBroadcaster.StartRecordingToSink(stopCh)

		defaultBindTimeout := int64(30)

		sched, err := scheduler.New(clientSet,
			informerFactory,
			scheduler.NewPodInformer(clientSet, 0),
			profile.NewRecorderFactory(eventBroadcaster),
			nil,
			scheduler.WithAlgorithmSource(kubeschedulerconfig.SchedulerAlgorithmSource{
				Policy: &kubeschedulerconfig.SchedulerPolicySource{
					ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
						Namespace: policyConfigMap.Namespace,
						Name:      policyConfigMap.Name,
					},
				},
			}),
			scheduler.WithBindTimeoutSeconds(defaultBindTimeout),
		)
		if err != nil {
			t.Fatalf("couldn't make scheduler config for test %d: %v", i, err)
		}

		schedPlugins := sched.Profiles[v1.DefaultSchedulerName].ListPlugins()
		if diff := cmp.Diff(test.expectedPlugins, schedPlugins); diff != "" {
			t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
		}
	}
}

// TestSchedulerCreationFromNonExistentConfigMap ensures that creation of the
// scheduler from a non-existent ConfigMap fails.
func TestSchedulerCreationFromNonExistentConfigMap(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	ns := framework.CreateTestingNamespace("configmap", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	defer clientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: clientSet.EventsV1beta1().Events("")})
	stopCh := make(chan struct{})
	eventBroadcaster.StartRecordingToSink(stopCh)

	defaultBindTimeout := int64(30)

	_, err := scheduler.New(clientSet,
		informerFactory,
		scheduler.NewPodInformer(clientSet, 0),
		profile.NewRecorderFactory(eventBroadcaster),
		nil,
		scheduler.WithAlgorithmSource(kubeschedulerconfig.SchedulerAlgorithmSource{
			Policy: &kubeschedulerconfig.SchedulerPolicySource{
				ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
					Namespace: "non-existent-config",
					Name:      "non-existent-config",
				},
			},
		}),
		scheduler.WithBindTimeoutSeconds(defaultBindTimeout))

	if err == nil {
		t.Fatalf("Creation of scheduler didn't fail while the policy ConfigMap didn't exist.")
	}
}

func TestUnschedulableNodes(t *testing.T) {
	testCtx := initTest(t, "unschedulable-nodes")
	defer testutils.CleanupTest(t, testCtx)

	nodeLister := testCtx.InformerFactory.Core().V1().Nodes().Lister()
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	goodCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: metav1.Time{Time: time.Now()},
	}
	// Create a new schedulable node, since we're first going to apply
	// the unschedulable condition and verify that pods aren't scheduled.
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-scheduling-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
			Conditions: []v1.NodeCondition{goodCondition},
		},
	}
	nodeKey, err := cache.MetaNamespaceKeyFunc(node)
	if err != nil {
		t.Fatalf("Couldn't retrieve key for node %v", node.Name)
	}

	// The test does the following for each nodeStateManager in this list:
	//	1. Create a new node
	//	2. Apply the makeUnSchedulable function
	//	3. Create a new pod
	//  4. Check that the pod doesn't get assigned to the node
	//  5. Apply the schedulable function
	//  6. Check that the pod *does* get assigned to the node
	//  7. Delete the pod and node.

	nodeModifications := []nodeStateManager{
		// Test node.Spec.Unschedulable=true/false
		{
			makeUnSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Spec.Unschedulable = true
				if _, err := c.CoreV1().Nodes().Update(context.TODO(), n, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update node with unschedulable=true: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					// An unschedulable node should still be present in the store
					// Nodes that are unschedulable or that are not ready or
					// have their disk full (Node.Spec.Conditions) are excluded
					// based on NodeConditionPredicate, a separate check
					return node != nil && node.(*v1.Node).Spec.Unschedulable == true
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=true: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Spec.Unschedulable = false
				if _, err := c.CoreV1().Nodes().Update(context.TODO(), n, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update node with unschedulable=false: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Spec.Unschedulable == false
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=false: %v", err)
				}
			},
		},
	}

	for i, mod := range nodeModifications {
		unSchedNode, err := testCtx.ClientSet.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Apply the unschedulable modification to the node, and wait for the reflection
		mod.makeUnSchedulable(t, unSchedNode, nodeLister, testCtx.ClientSet)

		// Create the new pod, note that this needs to happen post unschedulable
		// modification or we have a race in the test.
		myPod, err := createPausePodWithResource(testCtx.ClientSet, "node-scheduling-test-pod", testCtx.NS.Name, nil)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = testutils.WaitForPodToScheduleWithTimeout(testCtx.ClientSet, myPod, 2*time.Second)
		if err == nil {
			t.Errorf("Test %d: Pod scheduled successfully on unschedulable nodes", i)
		}
		if err != wait.ErrWaitTimeout {
			t.Errorf("Test %d: failed while trying to confirm the pod does not get scheduled on the node: %v", i, err)
		} else {
			t.Logf("Test %d: Pod did not get scheduled on an unschedulable node", i)
		}

		// Apply the schedulable modification to the node, and wait for the reflection
		schedNode, err := testCtx.ClientSet.CoreV1().Nodes().Get(context.TODO(), unSchedNode.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		mod.makeSchedulable(t, schedNode, nodeLister, testCtx.ClientSet)

		// Wait until the pod is scheduled.
		if err := testutils.WaitForPodToSchedule(testCtx.ClientSet, myPod); err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}
		// Clean up.
		if err := deletePod(testCtx.ClientSet, myPod.Name, myPod.Namespace); err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = testCtx.ClientSet.CoreV1().Nodes().Delete(context.TODO(), schedNode.Name, metav1.DeleteOptions{})
		if err != nil {
			t.Errorf("Failed to delete node: %v", err)
		}
	}
}

func TestMultipleSchedulers(t *testing.T) {
	// This integration tests the multi-scheduler feature in the following way:
	// 1. create a default scheduler
	// 2. create a node
	// 3. create 3 pods: testPodNoAnnotation, testPodWithAnnotationFitsDefault and testPodWithAnnotationFitsFoo
	//	  - note: the first two should be picked and scheduled by default scheduler while the last one should be
	//	          picked by scheduler of name "foo-scheduler" which does not exist yet.
	// 4. **check point-1**:
	//	   - testPodNoAnnotation, testPodWithAnnotationFitsDefault should be scheduled
	//	   - testPodWithAnnotationFitsFoo should NOT be scheduled
	// 5. create a scheduler with name "foo-scheduler"
	// 6. **check point-2**:
	//     - testPodWithAnnotationFitsFoo should be scheduled
	// 7. stop default scheduler
	// 8. create 2 pods: testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2
	//    - note: these two pods belong to default scheduler which no longer exists
	// 9. **check point-3**:
	//     - testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 should NOT be scheduled

	// 1. create and start default-scheduler
	testCtx := initTest(t, "multi-scheduler")
	defer testutils.CleanupTest(t, testCtx)

	// 2. create a node
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-multi-scheduler-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
		},
	}
	testCtx.ClientSet.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	// 3. create 3 pods for testing
	t.Logf("create 3 pods for testing")
	testPod, err := createPausePodWithResource(testCtx.ClientSet, "pod-without-scheduler-name", testCtx.NS.Name, nil)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	defaultScheduler := "default-scheduler"
	testPodFitsDefault, err := createPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{Name: "pod-fits-default", Namespace: testCtx.NS.Name, SchedulerName: defaultScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	fooScheduler := "foo-scheduler"
	testPodFitsFoo, err := createPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{Name: "pod-fits-foo", Namespace: testCtx.NS.Name, SchedulerName: fooScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 4. **check point-1**:
	//		- testPod, testPodFitsDefault should be scheduled
	//		- testPodFitsFoo should NOT be scheduled
	t.Logf("wait for pods scheduled")
	if err := testutils.WaitForPodToSchedule(testCtx.ClientSet, testPod); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPod.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPod.Name)
	}

	if err := testutils.WaitForPodToSchedule(testCtx.ClientSet, testPodFitsDefault); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodFitsDefault.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsDefault.Name)
	}

	if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.ClientSet, testPodFitsFoo, time.Second*5); err == nil {
		t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodFitsFoo.Name)
	}

	// 5. create and start a scheduler with name "foo-scheduler"
	fooProf := kubeschedulerconfig.KubeSchedulerProfile{SchedulerName: fooScheduler}
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, true, nil, time.Second, scheduler.WithProfiles(fooProf))
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	//	6. **check point-2**:
	//		- testPodWithAnnotationFitsFoo should be scheduled
	err = testutils.WaitForPodToSchedule(testCtx.ClientSet, testPodFitsFoo)
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsFoo.Name)
	}

	//	7. delete the pods that were scheduled by the default scheduler, and stop the default scheduler
	if err := deletePod(testCtx.ClientSet, testPod.Name, testCtx.NS.Name); err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}
	if err := deletePod(testCtx.ClientSet, testPodFitsDefault.Name, testCtx.NS.Name); err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}

	// The rest of this test assumes that closing StopEverything will cause the
	// scheduler thread to stop immediately.  It won't, and in fact it will often
	// schedule 1 more pod before finally exiting.  Comment out until we fix that.
	//
	// See https://github.com/kubernetes/kubernetes/issues/23715 for more details.

	/*
		close(schedulerConfig.StopEverything)

		//	8. create 2 pods: testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2
		//		- note: these two pods belong to default scheduler which no longer exists
		podWithNoAnnotation2 := createPod("pod-with-no-annotation2", nil)
		podWithAnnotationFitsDefault2 := createPod("pod-with-annotation-fits-default2", schedulerAnnotationFitsDefault)
		testPodNoAnnotation2, err := clientSet.CoreV1().Pods(ns.Name).Create(podWithNoAnnotation2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
		testPodWithAnnotationFitsDefault2, err := clientSet.CoreV1().Pods(ns.Name).Create(podWithAnnotationFitsDefault2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		//	9. **check point-3**:
		//		- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 should NOT be scheduled
		err = wait.Poll(time.Second, time.Second*5, testutils.PodScheduled(clientSet, testPodNoAnnotation2.Namespace, testPodNoAnnotation2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodNoAnnotation2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodNoAnnotation2.Name)
		}
		err = wait.Poll(time.Second, time.Second*5, testutils.PodScheduled(clientSet, testPodWithAnnotationFitsDefault2.Namespace, testPodWithAnnotationFitsDefault2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodWithAnnotationFitsDefault2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsDefault2.Name)
		}
	*/
}

func TestMultipleSchedulingProfiles(t *testing.T) {
	testCtx := initTest(t, "multi-scheduler", scheduler.WithProfiles(
		kubeschedulerconfig.KubeSchedulerProfile{
			SchedulerName: "default-scheduler",
		},
		kubeschedulerconfig.KubeSchedulerProfile{
			SchedulerName: "custom-scheduler",
		},
	))
	defer testutils.CleanupTest(t, testCtx)

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-multi-scheduler-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
		},
	}
	if _, err := testCtx.ClientSet.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	evs, err := testCtx.ClientSet.CoreV1().Events(testCtx.NS.Name).Watch(testCtx.Ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer evs.Stop()

	for _, pc := range []*pausePodConfig{
		{Name: "foo", Namespace: testCtx.NS.Name},
		{Name: "bar", Namespace: testCtx.NS.Name, SchedulerName: "unknown-scheduler"},
		{Name: "baz", Namespace: testCtx.NS.Name, SchedulerName: "default-scheduler"},
		{Name: "zet", Namespace: testCtx.NS.Name, SchedulerName: "custom-scheduler"},
	} {
		if _, err := createPausePod(testCtx.ClientSet, initPausePod(pc)); err != nil {
			t.Fatal(err)
		}
	}

	wantProfiles := map[string]string{
		"foo": "default-scheduler",
		"baz": "default-scheduler",
		"zet": "custom-scheduler",
	}

	gotProfiles := make(map[string]string)
	if err := wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		var ev watch.Event
		select {
		case ev = <-evs.ResultChan():
		case <-time.After(30 * time.Second):
			return false, nil
		}
		e, ok := ev.Object.(*v1.Event)
		if !ok || e.Reason != "Scheduled" {
			return false, nil
		}
		gotProfiles[e.InvolvedObject.Name] = e.ReportingController
		return len(gotProfiles) >= len(wantProfiles), nil
	}); err != nil {
		t.Errorf("waiting for scheduling events: %v", err)
	}

	if diff := cmp.Diff(wantProfiles, gotProfiles); diff != "" {
		t.Errorf("pods scheduled by the wrong profile (-want, +got):\n%s", diff)
	}
}

// This test will verify scheduler can work well regardless of whether kubelet is allocatable aware or not.
func TestAllocatable(t *testing.T) {
	testCtx := initTest(t, "allocatable")
	defer testutils.CleanupTest(t, testCtx)

	// 2. create a node without allocatable awareness
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
	}
	allocNode, err := createNode(testCtx.ClientSet, "node-allocatable-scheduler-test-node", nodeRes)
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 3. create resource pod which requires less than Capacity
	podName := "pod-test-allocatable"
	podRes := &v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(20, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(20, resource.BinarySI),
	}
	testAllocPod, err := createPausePodWithResource(testCtx.ClientSet, podName, testCtx.NS.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable unawareness failed to create pod: %v", err)
	}

	// 4. Test: this test pod should be scheduled since api-server will use Capacity as Allocatable
	err = testutils.WaitForPodToScheduleWithTimeout(testCtx.ClientSet, testAllocPod, time.Second*5)
	if err != nil {
		t.Errorf("Test allocatable unawareness: %s Pod not scheduled: %v", testAllocPod.Name, err)
	} else {
		t.Logf("Test allocatable unawareness: %s Pod scheduled", testAllocPod.Name)
	}

	// 5. Change the node status to allocatable aware, note that Allocatable is less than Pod's requirement
	allocNode.Status = v1.NodeStatus{
		Capacity: v1.ResourceList{
			v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			v1.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
		},
		Allocatable: v1.ResourceList{
			v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			v1.ResourceCPU:    *resource.NewMilliQuantity(10, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(10, resource.BinarySI),
		},
	}

	if _, err := testCtx.ClientSet.CoreV1().Nodes().UpdateStatus(context.TODO(), allocNode, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update node with Status.Allocatable: %v", err)
	}

	if err := deletePod(testCtx.ClientSet, testAllocPod.Name, testCtx.NS.Name); err != nil {
		t.Fatalf("Failed to remove the first pod: %v", err)
	}

	// 6. Make another pod with different name, same resource request
	podName2 := "pod-test-allocatable2"
	testAllocPod2, err := createPausePodWithResource(testCtx.ClientSet, podName2, testCtx.NS.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable awareness failed to create pod: %v", err)
	}

	// 7. Test: this test pod should not be scheduled since it request more than Allocatable
	if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.ClientSet, testAllocPod2, time.Second*5); err == nil {
		t.Errorf("Test allocatable awareness: %s Pod got scheduled unexpectedly, %v", testAllocPod2.Name, err)
	} else {
		t.Logf("Test allocatable awareness: %s Pod not scheduled as expected", testAllocPod2.Name)
	}
}

// TestSchedulerInformers tests that scheduler receives informer events and updates its cache when
// pods are scheduled by other schedulers.
func TestSchedulerInformers(t *testing.T) {
	// Initialize scheduler.
	testCtx := initTest(t, "scheduler-informer")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.BinarySI)},
	}
	defaultNodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.BinarySI),
	}

	type nodeConfig struct {
		name string
		res  *v1.ResourceList
	}

	tests := []struct {
		description         string
		nodes               []*nodeConfig
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		preemptedPodIndexes map[int]struct{}
	}{
		{
			description: "Pod cannot be scheduled when node is occupied by pods scheduled by other schedulers",
			nodes:       []*nodeConfig{{name: "node-1", res: defaultNodeRes}},
			existingPods: []*v1.Pod{
				initPausePod(&pausePodConfig{
					Name:          "pod1",
					Namespace:     testCtx.NS.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "foo-scheduler",
				}),
				initPausePod(&pausePodConfig{
					Name:          "pod2",
					Namespace:     testCtx.NS.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "bar-scheduler",
				}),
			},
			pod: initPausePod(&pausePodConfig{
				Name:      "unschedulable-pod",
				Namespace: testCtx.NS.Name,
				Resources: defaultPodRes,
			}),
			preemptedPodIndexes: map[int]struct{}{2: {}},
		},
	}

	for _, test := range tests {
		for _, nodeConf := range test.nodes {
			_, err := createNode(cs, nodeConf.name, nodeConf.res)
			if err != nil {
				t.Fatalf("Error creating node %v: %v", nodeConf.name, err)
			}
		}

		pods := make([]*v1.Pod, len(test.existingPods))
		var err error
		// Create and run existingPods.
		for i, p := range test.existingPods {
			if pods[i], err = runPausePod(cs, p); err != nil {
				t.Fatalf("Test [%v]: Error running pause pod: %v", test.description, err)
			}
		}
		// Create the new "pod".
		unschedulable, err := createPausePod(cs, test.pod)
		if err != nil {
			t.Errorf("Error while creating new pod: %v", err)
		}
		if err := waitForPodUnschedulable(cs, unschedulable); err != nil {
			t.Errorf("Pod %v got scheduled: %v", unschedulable.Name, err)
		}

		// Cleanup
		pods = append(pods, unschedulable)
		testutils.CleanupPods(cs, t, pods)
		cs.PolicyV1beta1().PodDisruptionBudgets(testCtx.NS.Name).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
		cs.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	}
}
