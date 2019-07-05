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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/test/integration/framework"
)

type nodeMutationFunc func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

func PredicateOne(pod *v1.Pod, meta predicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []predicates.PredicateFailureReason, error) {
	return true, nil, nil
}

func PredicateTwo(pod *v1.Pod, meta predicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []predicates.PredicateFailureReason, error) {
	return true, nil, nil
}

func PriorityOne(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	return []schedulerapi.HostPriority{}, nil
}

func PriorityTwo(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	return []schedulerapi.HostPriority{}, nil
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
	defer clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	// Pre-register some predicate and priority functions
	factory.RegisterFitPredicate("PredicateOne", PredicateOne)
	factory.RegisterFitPredicate("PredicateTwo", PredicateTwo)
	factory.RegisterPriorityFunction("PriorityOne", PriorityOne, 1)
	factory.RegisterPriorityFunction("PriorityTwo", PriorityTwo, 1)

	for i, test := range []struct {
		policy               string
		expectedPredicates   sets.String
		expectedPrioritizers sets.String
	}{
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "PredicateOne"},
					{"name" : "PredicateTwo"}
				],
				"priorities" : [
					{"name" : "PriorityOne", "weight" : 1},
					{"name" : "PriorityTwo", "weight" : 5}
				]
			}`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
				"PredicateOne",
				"PredicateTwo",
			),
			expectedPrioritizers: sets.NewString(
				"PriorityOne",
				"PriorityTwo",
			),
		},
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1"
			}`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
				"CheckNodeDiskPressure",
				"CheckNodeMemoryPressure",
				"CheckNodePIDPressure",
				"CheckVolumeBinding",
				"GeneralPredicates",
				"MatchInterPodAffinity",
				"MaxAzureDiskVolumeCount",
				"MaxCSIVolumeCountPred",
				"MaxEBSVolumeCount",
				"MaxGCEPDVolumeCount",
				"NoDiskConflict",
				"NoVolumeZoneConflict",
				"PodToleratesNodeTaints",
			),
			expectedPrioritizers: sets.NewString(
				"BalancedResourceAllocation",
				"InterPodAffinityPriority",
				"LeastRequestedPriority",
				"NodeAffinityPriority",
				"NodePreferAvoidPodsPriority",
				"SelectorSpreadPriority",
				"TaintTolerationPriority",
				"ImageLocalityPriority",
			),
		},
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [],
				"priorities" : []
			}`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
			),
			expectedPrioritizers: sets.NewString(),
		},
		{
			policy: `apiVersion: v1
kind: Policy
predicates:
- name: PredicateOne
- name: PredicateTwo
priorities:
- name: PriorityOne
  weight: 1
- name: PriorityTwo
  weight: 5
`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
				"PredicateOne",
				"PredicateTwo",
			),
			expectedPrioritizers: sets.NewString(
				"PriorityOne",
				"PriorityTwo",
			),
		},
		{
			policy: `apiVersion: v1
kind: Policy
`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
				"CheckNodeDiskPressure",
				"CheckNodeMemoryPressure",
				"CheckNodePIDPressure",
				"CheckVolumeBinding",
				"GeneralPredicates",
				"MatchInterPodAffinity",
				"MaxAzureDiskVolumeCount",
				"MaxCSIVolumeCountPred",
				"MaxEBSVolumeCount",
				"MaxGCEPDVolumeCount",
				"NoDiskConflict",
				"NoVolumeZoneConflict",
				"PodToleratesNodeTaints",
			),
			expectedPrioritizers: sets.NewString(
				"BalancedResourceAllocation",
				"InterPodAffinityPriority",
				"LeastRequestedPriority",
				"NodeAffinityPriority",
				"NodePreferAvoidPodsPriority",
				"SelectorSpreadPriority",
				"TaintTolerationPriority",
				"ImageLocalityPriority",
			),
		},
		{
			policy: `apiVersion: v1
kind: Policy
predicates: []
priorities: []
`,
			expectedPredicates: sets.NewString(
				"CheckNodeCondition", // mandatory predicate
			),
			expectedPrioritizers: sets.NewString(),
		},
	} {
		// Add a ConfigMap object.
		configPolicyName := fmt.Sprintf("scheduler-custom-policy-config-%d", i)
		policyConfigMap := v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: configPolicyName},
			Data:       map[string]string{kubeschedulerconfig.SchedulerPolicyConfigMapKey: test.policy},
		}

		policyConfigMap.APIVersion = "v1"
		clientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(&policyConfigMap)

		eventBroadcaster := record.NewBroadcaster()
		eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientSet.CoreV1().Events("")})

		defaultBindTimeout := int64(30)

		sched, err := scheduler.New(clientSet,
			informerFactory.Core().V1().Nodes(),
			factory.NewPodInformer(clientSet, 0),
			informerFactory.Core().V1().PersistentVolumes(),
			informerFactory.Core().V1().PersistentVolumeClaims(),
			informerFactory.Core().V1().ReplicationControllers(),
			informerFactory.Apps().V1().ReplicaSets(),
			informerFactory.Apps().V1().StatefulSets(),
			informerFactory.Core().V1().Services(),
			informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
			informerFactory.Storage().V1().StorageClasses(),
			informerFactory.Storage().V1beta1().CSINodes(),
			eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: v1.DefaultSchedulerName}),
			kubeschedulerconfig.SchedulerAlgorithmSource{
				Policy: &kubeschedulerconfig.SchedulerPolicySource{
					ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
						Namespace: policyConfigMap.Namespace,
						Name:      policyConfigMap.Name,
					},
				},
			},
			nil,
			schedulerframework.NewRegistry(),
			nil,
			[]kubeschedulerconfig.PluginConfig{},
			scheduler.WithName(v1.DefaultSchedulerName),
			scheduler.WithHardPodAffinitySymmetricWeight(v1.DefaultHardPodAffinitySymmetricWeight),
			scheduler.WithBindTimeoutSeconds(defaultBindTimeout),
		)
		if err != nil {
			t.Fatalf("couldn't make scheduler config: %v", err)
		}

		config := sched.Config()

		// Verify that the config is applied correctly.
		schedPredicates := sets.NewString()
		for k := range config.Algorithm.Predicates() {
			schedPredicates.Insert(k)
		}
		schedPrioritizers := sets.NewString()
		for _, p := range config.Algorithm.Prioritizers() {
			schedPrioritizers.Insert(p.Name)
		}
		if !schedPredicates.Equal(test.expectedPredicates) {
			t.Errorf("Expected predicates %v, got %v", test.expectedPredicates, schedPredicates)
		}
		if !schedPrioritizers.Equal(test.expectedPrioritizers) {
			t.Errorf("Expected priority functions %v, got %v", test.expectedPrioritizers, schedPrioritizers)
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
	defer clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientSet.CoreV1().Events("")})

	defaultBindTimeout := int64(30)

	_, err := scheduler.New(clientSet,
		informerFactory.Core().V1().Nodes(),
		factory.NewPodInformer(clientSet, 0),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Apps().V1().ReplicaSets(),
		informerFactory.Apps().V1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		informerFactory.Storage().V1beta1().CSINodes(),
		eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: v1.DefaultSchedulerName}),
		kubeschedulerconfig.SchedulerAlgorithmSource{
			Policy: &kubeschedulerconfig.SchedulerPolicySource{
				ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
					Namespace: "non-existent-config",
					Name:      "non-existent-config",
				},
			},
		},
		nil,
		schedulerframework.NewRegistry(),
		nil,
		[]kubeschedulerconfig.PluginConfig{},
		scheduler.WithName(v1.DefaultSchedulerName),
		scheduler.WithHardPodAffinitySymmetricWeight(v1.DefaultHardPodAffinitySymmetricWeight),
		scheduler.WithBindTimeoutSeconds(defaultBindTimeout))

	if err == nil {
		t.Fatalf("Creation of scheduler didn't fail while the policy ConfigMap didn't exist.")
	}
}

func TestUnschedulableNodes(t *testing.T) {
	context := initTest(t, "unschedulable-nodes")
	defer cleanupTest(t, context)

	nodeLister := context.schedulerConfigFactory.GetNodeLister()
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer context.clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})

	goodCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: metav1.Time{Time: time.Now()},
	}
	badCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionUnknown,
		Reason:            fmt.Sprintf("unschedulable condition"),
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
				if _, err := c.CoreV1().Nodes().Update(n); err != nil {
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
				if _, err := c.CoreV1().Nodes().Update(n); err != nil {
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
		// Test node.Status.Conditions=ConditionTrue/Unknown
		{
			makeUnSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Status = v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []v1.NodeCondition{badCondition},
				}
				if _, err = c.CoreV1().Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with bad status condition: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Status.Conditions[0].Status == v1.ConditionUnknown
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Status = v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []v1.NodeCondition{goodCondition},
				}
				if _, err = c.CoreV1().Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with healthy status condition: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Status.Conditions[0].Status == v1.ConditionTrue
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
		},
	}

	for i, mod := range nodeModifications {
		unSchedNode, err := context.clientSet.CoreV1().Nodes().Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Apply the unschedulable modification to the node, and wait for the reflection
		mod.makeUnSchedulable(t, unSchedNode, nodeLister, context.clientSet)

		// Create the new pod, note that this needs to happen post unschedulable
		// modification or we have a race in the test.
		myPod, err := createPausePodWithResource(context.clientSet, "node-scheduling-test-pod", context.ns.Name, nil)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = waitForPodToScheduleWithTimeout(context.clientSet, myPod, 2*time.Second)
		if err == nil {
			t.Errorf("Pod scheduled successfully on unschedulable nodes")
		}
		if err != wait.ErrWaitTimeout {
			t.Errorf("Test %d: failed while trying to confirm the pod does not get scheduled on the node: %v", i, err)
		} else {
			t.Logf("Test %d: Pod did not get scheduled on an unschedulable node", i)
		}

		// Apply the schedulable modification to the node, and wait for the reflection
		schedNode, err := context.clientSet.CoreV1().Nodes().Get(unSchedNode.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		mod.makeSchedulable(t, schedNode, nodeLister, context.clientSet)

		// Wait until the pod is scheduled.
		if err := waitForPodToSchedule(context.clientSet, myPod); err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}
		// Clean up.
		if err := deletePod(context.clientSet, myPod.Name, myPod.Namespace); err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = context.clientSet.CoreV1().Nodes().Delete(schedNode.Name, nil)
		if err != nil {
			t.Errorf("Failed to delete node: %v", err)
		}
	}
}

func TestMultiScheduler(t *testing.T) {
	/*
		This integration tests the multi-scheduler feature in the following way:
		1. create a default scheduler
		2. create a node
		3. create 3 pods: testPodNoAnnotation, testPodWithAnnotationFitsDefault and testPodWithAnnotationFitsFoo
			- note: the first two should be picked and scheduled by default scheduler while the last one should be
			        picked by scheduler of name "foo-scheduler" which does not exist yet.
		4. **check point-1**:
			- testPodNoAnnotation, testPodWithAnnotationFitsDefault should be scheduled
			- testPodWithAnnotationFitsFoo should NOT be scheduled
		5. create a scheduler with name "foo-scheduler"
		6. **check point-2**:
			- testPodWithAnnotationFitsFoo should be scheduled
		7. stop default scheduler
		8. create 2 pods: testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2
			- note: these two pods belong to default scheduler which no longer exists
		9. **check point-3**:
			- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 should NOT be scheduled
	*/

	// 1. create and start default-scheduler
	context := initTest(t, "multi-scheduler")
	defer cleanupTest(t, context)

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
	context.clientSet.CoreV1().Nodes().Create(node)

	// 3. create 3 pods for testing
	t.Logf("create 3 pods for testing")
	testPod, err := createPausePodWithResource(context.clientSet, "pod-without-scheduler-name", context.ns.Name, nil)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	defaultScheduler := "default-scheduler"
	testPodFitsDefault, err := createPausePod(context.clientSet, initPausePod(context.clientSet, &pausePodConfig{Name: "pod-fits-default", Namespace: context.ns.Name, SchedulerName: defaultScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	fooScheduler := "foo-scheduler"
	testPodFitsFoo, err := createPausePod(context.clientSet, initPausePod(context.clientSet, &pausePodConfig{Name: "pod-fits-foo", Namespace: context.ns.Name, SchedulerName: fooScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 4. **check point-1**:
	//		- testPod, testPodFitsDefault should be scheduled
	//		- testPodFitsFoo should NOT be scheduled
	t.Logf("wait for pods scheduled")
	if err := waitForPodToSchedule(context.clientSet, testPod); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPod.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPod.Name)
	}

	if err := waitForPodToSchedule(context.clientSet, testPodFitsDefault); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodFitsDefault.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsDefault.Name)
	}

	if err := waitForPodToScheduleWithTimeout(context.clientSet, testPodFitsFoo, time.Second*5); err == nil {
		t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodFitsFoo.Name)
	}

	// 5. create and start a scheduler with name "foo-scheduler"
	clientSet2 := clientset.NewForConfigOrDie(&restclient.Config{Host: context.httpServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	informerFactory2 := informers.NewSharedInformerFactory(context.clientSet, 0)
	podInformer2 := factory.NewPodInformer(context.clientSet, 0)

	stopCh := make(chan struct{})
	defer close(stopCh)

	schedulerConfigFactory2 := createConfiguratorWithPodInformer(fooScheduler, clientSet2, podInformer2, informerFactory2, schedulerframework.NewRegistry(),
		nil, []kubeschedulerconfig.PluginConfig{}, stopCh)
	schedulerConfig2, err := schedulerConfigFactory2.Create()
	if err != nil {
		t.Errorf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster2 := record.NewBroadcaster()
	schedulerConfig2.Recorder = eventBroadcaster2.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: fooScheduler})
	eventBroadcaster2.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientSet2.CoreV1().Events("")})

	sched2 := scheduler.NewFromConfig(schedulerConfig2)
	scheduler.AddAllEventHandlers(sched2,
		fooScheduler,
		context.informerFactory.Core().V1().Nodes(),
		podInformer2,
		context.informerFactory.Core().V1().PersistentVolumes(),
		context.informerFactory.Core().V1().PersistentVolumeClaims(),
		context.informerFactory.Core().V1().Services(),
		context.informerFactory.Storage().V1().StorageClasses(),
		context.informerFactory.Storage().V1beta1().CSINodes(),
	)

	go podInformer2.Informer().Run(stopCh)
	informerFactory2.Start(stopCh)
	sched2.Run()

	//	6. **check point-2**:
	//		- testPodWithAnnotationFitsFoo should be scheduled
	err = waitForPodToSchedule(context.clientSet, testPodFitsFoo)
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsFoo.Name)
	}

	//	7. delete the pods that were scheduled by the default scheduler, and stop the default scheduler
	if err := deletePod(context.clientSet, testPod.Name, context.ns.Name); err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}
	if err := deletePod(context.clientSet, testPodFitsDefault.Name, context.ns.Name); err != nil {
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
		err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodNoAnnotation2.Namespace, testPodNoAnnotation2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodNoAnnotation2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodNoAnnotation2.Name)
		}
		err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodWithAnnotationFitsDefault2.Namespace, testPodWithAnnotationFitsDefault2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodWithAnnotationFitsDefault2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsDefault2.Name)
		}
	*/
}

// This test will verify scheduler can work well regardless of whether kubelet is allocatable aware or not.
func TestAllocatable(t *testing.T) {
	context := initTest(t, "allocatable")
	defer cleanupTest(t, context)

	// 2. create a node without allocatable awareness
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
	}
	allocNode, err := createNode(context.clientSet, "node-allocatable-scheduler-test-node", nodeRes)
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 3. create resource pod which requires less than Capacity
	podName := "pod-test-allocatable"
	podRes := &v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(20, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(20, resource.BinarySI),
	}
	testAllocPod, err := createPausePodWithResource(context.clientSet, podName, context.ns.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable unawareness failed to create pod: %v", err)
	}

	// 4. Test: this test pod should be scheduled since api-server will use Capacity as Allocatable
	err = waitForPodToScheduleWithTimeout(context.clientSet, testAllocPod, time.Second*5)
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

	if _, err := context.clientSet.CoreV1().Nodes().UpdateStatus(allocNode); err != nil {
		t.Fatalf("Failed to update node with Status.Allocatable: %v", err)
	}

	if err := deletePod(context.clientSet, testAllocPod.Name, context.ns.Name); err != nil {
		t.Fatalf("Failed to remove the first pod: %v", err)
	}

	// 6. Make another pod with different name, same resource request
	podName2 := "pod-test-allocatable2"
	testAllocPod2, err := createPausePodWithResource(context.clientSet, podName2, context.ns.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable awareness failed to create pod: %v", err)
	}

	// 7. Test: this test pod should not be scheduled since it request more than Allocatable
	if err := waitForPodToScheduleWithTimeout(context.clientSet, testAllocPod2, time.Second*5); err == nil {
		t.Errorf("Test allocatable awareness: %s Pod got scheduled unexpectedly, %v", testAllocPod2.Name, err)
	} else {
		t.Logf("Test allocatable awareness: %s Pod not scheduled as expected", testAllocPod2.Name)
	}
}

// TestSchedulerInformers tests that scheduler receives informer events and updates its cache when
// pods are scheduled by other schedulers.
func TestSchedulerInformers(t *testing.T) {
	// Initialize scheduler.
	context := initTest(t, "scheduler-informer")
	defer cleanupTest(t, context)
	cs := context.clientSet

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
				initPausePod(context.clientSet, &pausePodConfig{
					Name:          "pod1",
					Namespace:     context.ns.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "foo-scheduler",
				}),
				initPausePod(context.clientSet, &pausePodConfig{
					Name:          "pod2",
					Namespace:     context.ns.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "bar-scheduler",
				}),
			},
			pod: initPausePod(cs, &pausePodConfig{
				Name:      "unschedulable-pod",
				Namespace: context.ns.Name,
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
		cleanupPods(cs, t, pods)
		cs.PolicyV1beta1().PodDisruptionBudgets(context.ns.Name).DeleteCollection(nil, metav1.ListOptions{})
		cs.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
	}
}
