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

package podgc

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/util/workqueue"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/podgc/metrics"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

// testFinalizer keeps a pod around after its deletion has been accepted by the API server.
const testFinalizer = "example.com/some-finalizer"

func alwaysReady() bool { return true }

func NewFromClient(ctx context.Context, kubeClient clientset.Interface, terminatedPodThreshold int) (*PodGCController, coreinformers.PodInformer, coreinformers.NodeInformer) {
	informerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	nodeInformer := informerFactory.Core().V1().Nodes()
	controller := NewPodGC(ctx, kubeClient, podInformer, nodeInformer, terminatedPodThreshold)
	controller.podListerSynced = alwaysReady
	return controller, podInformer, nodeInformer
}

func TestGCTerminated(t *testing.T) {
	type nameToPhase struct {
		name                       string
		phase                      v1.PodPhase
		reason                     string
		deletionTimeStamp          *metav1.Time
		deletionGracePeriodSeconds *int64
		finalizers                 []string
	}

	testCases := []struct {
		name            string
		pods            []nameToPhase
		threshold       int
		deletedPodNames sets.Set[string]
		patchedPodNames sets.Set[string]
	}{
		{
			name: "delete pod a which is PodFailed and pod b which is PodSucceeded",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.New("a", "b"),
		},
		{
			name: "threshold = 0, disables terminated pod deletion",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold: 0,
			// threshold = 0 disables terminated pod deletion
		},
		{
			name: "threshold = 1, delete pod a which is PodFailed and pod b which is PodSucceeded",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.New("a", "b"),
		},
		{
			name: "threshold = 1, delete pod b which is PodSucceeded",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodRunning},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.New("b"),
		},
		{
			name: "threshold = 1, delete pod a which is PodFailed",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       1,
			deletedPodNames: sets.New("a"),
		},
		{
			name: "threshold = 5, don't delete pod",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold: 5,
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed, reason: eviction.Reason},
			},
			threshold:       1,
			deletedPodNames: sets.New("c", "a"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodRunning},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed, reason: eviction.Reason},
			},
			threshold:       1,
			deletedPodNames: sets.New("c"),
		},
		{
			name: "pods that are only waiting for their finalizers are not deleted again",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "b", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
			},
			threshold: 1,
		},
		{
			name: "pods that are only waiting for their finalizers still count towards the threshold",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "b", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "c", phase: v1.PodSucceeded},
			},
			threshold:       2,
			deletedPodNames: sets.New("c"),
		},
		{
			name: "the number of deletion candidates limits the number of deleted pods",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "b", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "c", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}},
				{name: "d", phase: v1.PodSucceeded},
			},
			threshold:       1,
			deletedPodNames: sets.New("d"),
		},
		{
			name: "terminated pods that are not only waiting for their finalizers are still deleted",
			pods: []nameToPhase{
				// the grace period still has to be shortened
				{name: "a", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](30), finalizers: []string{testFinalizer}},
				// nothing holds the pod back, so resending the deletion can still make it go away
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0)},
				// the deletion has not been accepted yet
				{name: "c", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, finalizers: []string{testFinalizer}},
				{name: "d", phase: v1.PodSucceeded, finalizers: []string{testFinalizer}},
			},
			threshold:       1,
			deletedPodNames: sets.New("a", "b", "c"),
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			_, ctx := ktesting.NewTestContext(t)
			creationTime := time.Unix(0, 0)
			nodes := []*v1.Node{testutil.NewNode("node")}

			pods := make([]*v1.Pod, 0, len(test.pods))
			for _, pod := range test.pods {
				creationTime = creationTime.Add(1 * time.Hour)
				pods = append(pods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:                       pod.name,
						Namespace:                  metav1.NamespaceDefault,
						CreationTimestamp:          metav1.Time{Time: creationTime},
						DeletionTimestamp:          pod.deletionTimeStamp,
						DeletionGracePeriodSeconds: pod.deletionGracePeriodSeconds,
						Finalizers:                 pod.finalizers,
					},
					Status: v1.PodStatus{Phase: pod.phase, Reason: pod.reason},
					Spec:   v1.PodSpec{NodeName: "node"},
				})
			}
			client := setupNewSimpleClient(nodes, pods)
			gcc, podInformer, _ := NewFromClient(ctx, client, test.threshold)
			for _, pod := range pods {
				podInformer.Informer().GetStore().Add(pod)
			}

			gcc.gc(ctx)

			verifyDeletedAndPatchedPods(t, client, test.deletedPodNames, test.patchedPodNames)
			testDeletingPodsMetrics(t, len(test.deletedPodNames), metrics.PodGCReasonTerminated)
		})
	}
}

func makePod(name string, nodeName string, phase v1.PodPhase) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec:   v1.PodSpec{NodeName: nodeName},
		Status: v1.PodStatus{Phase: phase},
	}
}

// makeForceDeletedPod returns a pod whose immediate deletion has already been accepted by
// the API server, so that only the removal of its finalizers can make it go away.
func makeForceDeletedPod(name string, nodeName string, phase v1.PodPhase, finalizers ...string) *v1.Pod {
	pod := makePod(name, nodeName, phase)
	pod.DeletionTimestamp = &metav1.Time{}
	pod.DeletionGracePeriodSeconds = ptr.To[int64](0)
	pod.Finalizers = finalizers
	return pod
}

func waitForAdded(q workqueue.TypedDelayingInterface[string], depth int) error {
	return wait.Poll(1*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		if q.Len() == depth {
			return true, nil
		}

		return false, nil
	})
}

func TestGCOrphaned(t *testing.T) {
	testCases := []struct {
		name                 string
		initialClientNodes   []*v1.Node
		initialInformerNodes []*v1.Node
		delay                time.Duration
		addedClientNodes     []*v1.Node
		deletedClientNodes   []*v1.Node
		addedInformerNodes   []*v1.Node
		deletedInformerNodes []*v1.Node
		pods                 []*v1.Pod
		itemsInQueue         int
		deletedPodNames      sets.Set[string]
		patchedPodNames      sets.Set[string]
	}{
		{
			name: "nodes present in lister",
			initialInformerNodes: []*v1.Node{
				testutil.NewNode("existing1"),
				testutil.NewNode("existing2"),
			},
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "existing1", v1.PodRunning),
				makePod("b", "existing2", v1.PodFailed),
				makePod("c", "existing2", v1.PodSucceeded),
			},
			itemsInQueue: 0,
		},
		{
			name: "nodes present in client",
			initialClientNodes: []*v1.Node{
				testutil.NewNode("existing1"),
				testutil.NewNode("existing2"),
			},
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "existing1", v1.PodRunning),
				makePod("b", "existing2", v1.PodFailed),
				makePod("c", "existing2", v1.PodSucceeded),
			},
			itemsInQueue: 2,
		},
		{
			name:  "no nodes",
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
				makePod("b", "deleted", v1.PodSucceeded),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("a", "b"),
		},
		{
			name:  "no nodes, one running pod",
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
				makePod("b", "deleted", v1.PodSucceeded),
				makePod("c", "deleted", v1.PodRunning),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("a", "b", "c"),
			patchedPodNames: sets.New("c"),
		},
		{
			name:  "no nodes, pods that are only waiting for their finalizers",
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makeForceDeletedPod("a", "deleted", v1.PodFailed, testFinalizer),
				makePod("b", "deleted", v1.PodRunning),
				makePod("c", "deleted", v1.PodSucceeded),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("b", "c"),
			patchedPodNames: sets.New("b"),
		},
		{
			name:  "quarantine not finished",
			delay: quarantineTime / 2,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
			},
			itemsInQueue: 0,
		},
		{
			name:                 "wrong nodes",
			initialInformerNodes: []*v1.Node{testutil.NewNode("existing")},
			delay:                2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodRunning),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("a"),
			patchedPodNames: sets.New("a"),
		},
		{
			name:                 "some nodes missing",
			initialInformerNodes: []*v1.Node{testutil.NewNode("existing")},
			delay:                2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
				makePod("b", "existing", v1.PodFailed),
				makePod("c", "deleted", v1.PodSucceeded),
				makePod("d", "deleted", v1.PodRunning),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("a", "c", "d"),
			patchedPodNames: sets.New("d"),
		},
		{
			name:             "node added to client after quarantine",
			delay:            2 * quarantineTime,
			addedClientNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodRunning),
			},
			itemsInQueue: 1,
		},
		{
			name:               "node added to informer after quarantine",
			delay:              2 * quarantineTime,
			addedInformerNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodFailed),
			},
			itemsInQueue: 1,
		},
		{
			// It shouldn't happen that client will be lagging behind informer.
			// This test case is more a sanity check.
			name:               "node deleted from client after quarantine",
			initialClientNodes: []*v1.Node{testutil.NewNode("node")},
			delay:              2 * quarantineTime,
			deletedClientNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodFailed),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.New("a"),
		},
		{
			name:                 "node deleted from informer after quarantine",
			initialInformerNodes: []*v1.Node{testutil.NewNode("node")},
			delay:                2 * quarantineTime,
			deletedInformerNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodSucceeded),
			},
			itemsInQueue: 0,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			_, ctx := ktesting.NewTestContext(t)

			client := setupNewSimpleClient(test.initialClientNodes, test.pods)
			gcc, podInformer, nodeInformer := NewFromClient(ctx, client, -1)
			for _, node := range test.initialInformerNodes {
				nodeInformer.Informer().GetStore().Add(node)
			}
			for _, pod := range test.pods {
				podInformer.Informer().GetStore().Add(pod)
			}
			// Overwrite queue
			fakeClock := testingclock.NewFakeClock(time.Now())
			gcc.nodeQueue.ShutDown()
			gcc.nodeQueue = workqueue.NewTypedDelayingQueueWithConfig(workqueue.TypedDelayingQueueConfig[string]{Clock: fakeClock, Name: "podgc_test_queue"})

			// First GC of orphaned pods
			gcc.gc(ctx)
			deletedPodNames := getDeletedPodNames(client)

			if len(deletedPodNames) > 0 {
				t.Errorf("no pods should be deleted at this point.\n\tactual: %v", deletedPodNames)
			}

			// Move clock forward
			fakeClock.Step(test.delay)
			// Wait for queue goroutine to process items
			if test.itemsInQueue > 0 {
				err := waitForAdded(gcc.nodeQueue, test.itemsInQueue)
				if err != nil {
					t.Errorf("wrong number of items in the node queue.\n\texpected: %v\n\tactual: %v",
						test.itemsInQueue, gcc.nodeQueue.Len())
				}
			}

			// Execute planned nodes changes
			for _, node := range test.addedClientNodes {
				client.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
			}
			for _, node := range test.deletedClientNodes {
				client.CoreV1().Nodes().Delete(context.TODO(), node.Name, metav1.DeleteOptions{})
			}
			for _, node := range test.addedInformerNodes {
				nodeInformer.Informer().GetStore().Add(node)
			}
			for _, node := range test.deletedInformerNodes {
				nodeInformer.Informer().GetStore().Delete(node)
			}

			// Actual pod deletion
			gcc.gc(context.TODO())
			verifyDeletedAndPatchedPods(t, client, test.deletedPodNames, test.patchedPodNames)
			testDeletingPodsMetrics(t, len(test.deletedPodNames), metrics.PodGCReasonOrphaned)
		})
	}
}

func TestGCUnscheduledTerminating(t *testing.T) {
	type nameToPhase struct {
		name                       string
		phase                      v1.PodPhase
		deletionTimeStamp          *metav1.Time
		deletionGracePeriodSeconds *int64
		finalizers                 []string
		nodeName                   string
	}

	testCases := []struct {
		name            string
		pods            []nameToPhase
		deletedPodNames sets.Set[string]
		patchedPodNames sets.Set[string]
	}{
		{
			name: "Unscheduled pod in any phase must be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
			},
			deletedPodNames: sets.New("a", "b", "c"),
			patchedPodNames: sets.New("c"),
		},
		{
			name: "Scheduled pod in any phase must not be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: nil, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: nil, nodeName: "node"},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: "node"},
			},
		},
		{
			name: "Unscheduled pod that is only waiting for its finalizers must not be deleted again",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}, nodeName: ""},
				{name: "b", phase: v1.PodPending, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}, nodeName: ""},
				{name: "c", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), nodeName: ""},
			},
			deletedPodNames: sets.New("b", "c"),
			patchedPodNames: sets.New("b"),
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			_, ctx := ktesting.NewTestContext(t)
			creationTime := time.Unix(0, 0)

			pods := make([]*v1.Pod, 0, len(test.pods))
			for _, pod := range test.pods {
				creationTime = creationTime.Add(1 * time.Hour)
				pods = append(pods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:                       pod.name,
						Namespace:                  metav1.NamespaceDefault,
						CreationTimestamp:          metav1.Time{Time: creationTime},
						DeletionTimestamp:          pod.deletionTimeStamp,
						DeletionGracePeriodSeconds: pod.deletionGracePeriodSeconds,
						Finalizers:                 pod.finalizers,
					},
					Status: v1.PodStatus{Phase: pod.phase},
					Spec:   v1.PodSpec{NodeName: pod.nodeName},
				})
			}
			nodes := []*v1.Node{}
			client := setupNewSimpleClient(nodes, pods)
			gcc, podInformer, _ := NewFromClient(ctx, client, -1)

			for _, pod := range pods {
				podInformer.Informer().GetStore().Add(pod)
			}

			pods, err := podInformer.Lister().List(labels.Everything())
			if err != nil {
				t.Errorf("Error while listing all Pods: %v", err)
				return
			}
			gcc.gcUnscheduledTerminating(ctx, pods)
			verifyDeletedAndPatchedPods(t, client, test.deletedPodNames, test.patchedPodNames)
			testDeletingPodsMetrics(t, len(test.deletedPodNames), metrics.PodGCReasonTerminatingUnscheduled)
		})
	}
}

func TestGCTerminating(t *testing.T) {
	type node struct {
		name           string
		readyCondition v1.ConditionStatus
		taints         []v1.Taint
	}

	type nameToPodConfig struct {
		name                       string
		phase                      v1.PodPhase
		deletionTimeStamp          *metav1.Time
		deletionGracePeriodSeconds *int64
		finalizers                 []string
		nodeName                   string
	}

	testCases := []struct {
		name            string
		pods            []nameToPodConfig
		nodes           []node
		deletedPodNames sets.Set[string]
		patchedPodNames sets.Set[string]
	}{
		{
			name: "pods have deletion timestamp set and the corresponding nodes are not ready",
			nodes: []node{
				{name: "worker-0", readyCondition: v1.ConditionFalse},
				{name: "worker-1", readyCondition: v1.ConditionFalse},
			},
			pods: []nameToPodConfig{
				{name: "a", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-0"},
				{name: "b", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-1"},
			},
		},

		{
			name: "some pods have deletion timestamp and/or phase set and some of the corresponding nodes have an" +
				"outofservice taint that are not ready",
			nodes: []node{
				// terminated pods on this node should be force deleted
				{name: "worker-0", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Effect: v1.TaintEffectNoExecute}}},
				// terminated pods on this node should not be force deleted
				{name: "worker-1", readyCondition: v1.ConditionFalse},
				// terminated pods on this node should not be force deleted
				{name: "worker-2", readyCondition: v1.ConditionTrue},
				// terminated pods on this node should be force deleted
				{name: "worker-3", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Effect: v1.TaintEffectNoSchedule}}},
				// terminated pods on this node should be force deleted
				{name: "worker-4", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Effect: v1.TaintEffectPreferNoSchedule}}},
				// terminated pods on this node should be force deleted
				{name: "worker-5", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Value: "any-value", Effect: v1.TaintEffectNoExecute}}},
			},
			pods: []nameToPodConfig{
				// pods a1, b1, c1, d1 and e1 are on node worker-0
				{name: "a1", nodeName: "worker-0"},
				{name: "b1", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-0"},
				{name: "c1", phase: v1.PodPending, nodeName: "worker-0"},
				{name: "d1", phase: v1.PodRunning, nodeName: "worker-0"},
				{name: "e1", phase: v1.PodUnknown, nodeName: "worker-0"},

				// pods a2, b2, c2, d2 and e2 are on node worker-1
				{name: "a2", nodeName: "worker-1"},
				{name: "b2", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-1"},
				{name: "c2", phase: v1.PodPending, nodeName: "worker-1"},
				{name: "d2", phase: v1.PodRunning, nodeName: "worker-1"},
				{name: "e2", phase: v1.PodUnknown, nodeName: "worker-1"},

				// pods a3, b3, c3, d3 and e3 are on node worker-2
				{name: "a3", nodeName: "worker-2"},
				{name: "b3", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-2"},
				{name: "c3", phase: v1.PodPending, nodeName: "worker-2"},
				{name: "d3", phase: v1.PodRunning, nodeName: "worker-2"},
				{name: "e3", phase: v1.PodUnknown, nodeName: "worker-2"},

				// pods a4, b4, c4, d4 and e4 are on node worker-3
				{name: "a4", nodeName: "worker-3"},
				{name: "b4", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-3"},
				{name: "c4", phase: v1.PodPending, nodeName: "worker-3"},
				{name: "d4", phase: v1.PodRunning, nodeName: "worker-3"},
				{name: "e4", phase: v1.PodUnknown, nodeName: "worker-3"},

				// pods a5, b5, c5, d5 and e5 are on node worker-4
				{name: "a5", nodeName: "worker-3"},
				{name: "b5", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-4"},
				{name: "c5", phase: v1.PodPending, nodeName: "worker-4"},
				{name: "d5", phase: v1.PodRunning, nodeName: "worker-4"},
				{name: "e5", phase: v1.PodUnknown, nodeName: "worker-4"},

				// pods a6, b6, c6, d6 and e6 are on node worker-5
				{name: "a6", nodeName: "worker-5"},
				{name: "b6", deletionTimeStamp: &metav1.Time{}, nodeName: "worker-5"},
				{name: "c6", phase: v1.PodPending, nodeName: "worker-5"},
				{name: "d6", phase: v1.PodRunning, nodeName: "worker-5"},
				{name: "e6", phase: v1.PodUnknown, nodeName: "worker-5"},
			},
			deletedPodNames: sets.New("b1", "b4", "b5", "b6"),
			patchedPodNames: sets.New("b1", "b4", "b5", "b6"),
		},
		{
			name: "pods deleted from node tainted out-of-service",
			nodes: []node{
				{name: "worker", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Effect: v1.TaintEffectNoExecute}}},
			},
			pods: []nameToPodConfig{
				{name: "a", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: "worker"},
				{name: "b", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, nodeName: "worker"},
				{name: "c", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, nodeName: "worker"},
			},
			deletedPodNames: sets.New("a", "b", "c"),
			patchedPodNames: sets.New("a"),
		},
		{
			name: "pods that are only waiting for their finalizers are not deleted again",
			nodes: []node{
				{name: "worker", readyCondition: v1.ConditionFalse, taints: []v1.Taint{{Key: v1.TaintNodeOutOfService,
					Effect: v1.TaintEffectNoExecute}}},
			},
			pods: []nameToPodConfig{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}, nodeName: "worker"},
				{name: "b", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, deletionGracePeriodSeconds: ptr.To[int64](0), finalizers: []string{testFinalizer}, nodeName: "worker"},
			},
			deletedPodNames: sets.New("b"),
			patchedPodNames: sets.New("b"),
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			_, ctx := ktesting.NewTestContext(t)

			creationTime := time.Unix(0, 0)
			nodes := make([]*v1.Node, 0, len(test.nodes))
			for _, node := range test.nodes {
				creationTime = creationTime.Add(2 * time.Hour)
				nodes = append(nodes, &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: node.name, CreationTimestamp: metav1.Time{Time: creationTime}},
					Spec: v1.NodeSpec{
						Taints: node.taints,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:   v1.NodeReady,
								Status: node.readyCondition,
							},
						},
					},
				})
			}
			pods := make([]*v1.Pod, 0, len(test.pods))
			for _, pod := range test.pods {
				creationTime = creationTime.Add(1 * time.Hour)
				pods = append(pods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:                       pod.name,
						Namespace:                  metav1.NamespaceDefault,
						CreationTimestamp:          metav1.Time{Time: creationTime},
						DeletionTimestamp:          pod.deletionTimeStamp,
						DeletionGracePeriodSeconds: pod.deletionGracePeriodSeconds,
						Finalizers:                 pod.finalizers,
					},
					Status: v1.PodStatus{Phase: pod.phase},
					Spec:   v1.PodSpec{NodeName: pod.nodeName},
				})
			}
			client := setupNewSimpleClient(nodes, pods)
			gcc, podInformer, nodeInformer := NewFromClient(ctx, client, -1)

			for _, pod := range pods {
				podInformer.Informer().GetStore().Add(pod)
			}
			for _, node := range nodes {
				nodeInformer.Informer().GetStore().Add(node)
			}

			gcc.gc(ctx)
			verifyDeletedAndPatchedPods(t, client, test.deletedPodNames, test.patchedPodNames)
			testDeletingPodsMetrics(t, len(test.deletedPodNames), metrics.PodGCReasonTerminatingOutOfService)
		})
	}
}

func TestIsPodBlockedOnFinalizers(t *testing.T) {
	testCases := []struct {
		name string
		pod  *v1.Pod
		want bool
	}{
		{
			name: "terminated pod that is not marked for deletion",
			pod:  makePod("a", "node", v1.PodSucceeded),
		},
		{
			name: "terminated pod whose deletion has not been accepted yet",
			pod: func() *v1.Pod {
				pod := makeForceDeletedPod("a", "node", v1.PodSucceeded, testFinalizer)
				pod.DeletionGracePeriodSeconds = nil
				return pod
			}(),
		},
		{
			name: "terminated pod that is still within its grace period",
			pod: func() *v1.Pod {
				pod := makeForceDeletedPod("a", "node", v1.PodSucceeded, testFinalizer)
				pod.DeletionGracePeriodSeconds = ptr.To[int64](30)
				return pod
			}(),
		},
		{
			name: "terminated pod without finalizers",
			pod:  makeForceDeletedPod("a", "node", v1.PodSucceeded),
		},
		{
			name: "running pod",
			pod:  makeForceDeletedPod("a", "node", v1.PodRunning, testFinalizer),
		},
		{
			name: "succeeded pod that is only waiting for its finalizers",
			pod:  makeForceDeletedPod("a", "node", v1.PodSucceeded, testFinalizer),
			want: true,
		},
		{
			name: "failed pod that is only waiting for its finalizers",
			pod:  makeForceDeletedPod("a", "node", v1.PodFailed, testFinalizer),
			want: true,
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			if got := isPodBlockedOnFinalizers(test.pod); got != test.want {
				t.Errorf("isPodBlockedOnFinalizers() = %v, want %v", got, test.want)
			}
		})
	}
}

// TestGCTerminatedConverges verifies that PodGC stops force deleting terminated pods once
// the API server has accepted their deletion, instead of retrying them on every sync.
func TestGCTerminatedConverges(t *testing.T) {
	resetMetrics()
	_, ctx := ktesting.NewTestContext(t)

	creationTime := time.Unix(0, 0)
	pods := make([]*v1.Pod, 0, 3)
	for _, name := range []string{"a", "b", "c"} {
		creationTime = creationTime.Add(1 * time.Hour)
		pod := makePod(name, "node", v1.PodSucceeded)
		pod.CreationTimestamp = metav1.Time{Time: creationTime}
		pod.Finalizers = []string{testFinalizer}
		pods = append(pods, pod)
	}
	nodes := []*v1.Node{testutil.NewNode("node")}

	client := setupNewSimpleClient(nodes, pods)
	gcc, podInformer, nodeInformer := NewFromClient(ctx, client, 1)
	if err := nodeInformer.Informer().GetStore().Add(nodes[0]); err != nil {
		t.Fatalf("Failed to add node to the informer store: %v", err)
	}
	store := podInformer.Informer().GetStore()
	for _, pod := range pods {
		if err := store.Add(pod); err != nil {
			t.Fatalf("Failed to add pod to the informer store: %v", err)
		}
	}

	// Reflect what the informer observes once the API server has accepted the deletion:
	// the pod is marked for immediate deletion but kept around by its finalizer.
	acceptDeletion := func(pod *v1.Pod) {
		t.Helper()
		accepted := pod.DeepCopy()
		accepted.DeletionTimestamp = &metav1.Time{Time: creationTime}
		accepted.DeletionGracePeriodSeconds = ptr.To[int64](0)
		if err := store.Update(accepted); err != nil {
			t.Fatalf("Failed to update pod in the informer store: %v", err)
		}
	}

	// The two oldest pods are force deleted, the threshold keeps the newest one.
	gcc.gc(ctx)
	verifyDeletedAndPatchedPods(t, client, sets.New("a", "b"), nil)
	acceptDeletion(pods[0])
	acceptDeletion(pods[1])

	// a and b are only waiting for their finalizers now, so c is the only candidate left.
	client.ClearActions()
	gcc.gc(ctx)
	verifyDeletedAndPatchedPods(t, client, sets.New("c"), nil)
	acceptDeletion(pods[2])

	// Nothing is left for PodGC to do until the finalizers are removed.
	client.ClearActions()
	gcc.gc(ctx)
	verifyDeletedAndPatchedPods(t, client, nil, nil)
}

func TestGCInspectingPatchedPodBeforeDeletion(t *testing.T) {
	testCases := []struct {
		name                 string
		pod                  *v1.Pod
		expectedPatchedPod   *v1.Pod
		expectedDeleteAction *clienttesting.DeleteActionImpl
	}{
		{
			name: "orphaned pod should have DisruptionTarget condition added before deletion",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "testPod",
				},
				Spec: v1.PodSpec{
					NodeName: "deletedNode",
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
			},
			expectedPatchedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "testPod",
				},
				Spec: v1.PodSpec{
					NodeName: "deletedNode",
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
					Conditions: []v1.PodCondition{
						{
							Type:   v1.PodReady,
							Status: v1.ConditionTrue,
						},
						{
							Type:    v1.DisruptionTarget,
							Status:  v1.ConditionTrue,
							Reason:  "DeletionByPodGC",
							Message: "PodGC: node no longer exists",
						},
					},
				},
			},
			expectedDeleteAction: &clienttesting.DeleteActionImpl{
				Name:          "testPod",
				DeleteOptions: metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)},
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			pods := []*v1.Pod{test.pod}

			client := setupNewSimpleClient(nil, pods)
			gcc, podInformer, _ := NewFromClient(ctx, client, -1)
			gcc.quarantineTime = time.Duration(-1)
			podInformer.Informer().GetStore().Add(test.pod)
			gcc.gc(ctx)

			actions := client.Actions()

			var patchAction clienttesting.PatchAction
			var deleteAction clienttesting.DeleteAction

			for _, action := range actions {
				if action.GetVerb() == "patch" {
					patchAction = action.(clienttesting.PatchAction)
				}

				if action.GetVerb() == "delete" {
					deleteAction = action.(clienttesting.DeleteAction)
				}
			}

			if patchAction != nil && test.expectedPatchedPod == nil {
				t.Fatalf("Pod was pactched but expectedPatchedPod is nil")
			}
			if test.expectedPatchedPod != nil {
				patchedPodBytes := patchAction.GetPatch()
				originalPod, err := json.Marshal(test.pod)
				if err != nil {
					t.Fatalf("Failed to marshal original pod %#v: %v", originalPod, err)
				}
				updated, err := strategicpatch.StrategicMergePatch(originalPod, patchedPodBytes, v1.Pod{})
				if err != nil {
					t.Fatalf("Failed to apply strategic merge patch %q on pod %#v: %v", patchedPodBytes, originalPod, err)
				}

				updatedPod := &v1.Pod{}
				if err := json.Unmarshal(updated, updatedPod); err != nil {
					t.Fatalf("Failed to unmarshal updated pod %q: %v", updated, err)
				}

				if diff := cmp.Diff(test.expectedPatchedPod, updatedPod, cmpopts.IgnoreFields(v1.Pod{}, "TypeMeta"), cmpopts.IgnoreFields(v1.PodCondition{}, "LastTransitionTime")); diff != "" {
					t.Fatalf("Unexpected diff on pod (-want,+got):\n%s", diff)
				}
			}

			if deleteAction != nil && test.expectedDeleteAction == nil {
				t.Fatalf("Pod was deleted but expectedDeleteAction is nil")
			}
			if test.expectedDeleteAction != nil {
				if diff := cmp.Diff(*test.expectedDeleteAction, deleteAction, cmpopts.IgnoreFields(clienttesting.DeleteActionImpl{}, "ActionImpl")); diff != "" {
					t.Fatalf("Unexpected diff on deleteAction (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func verifyDeletedAndPatchedPods(t *testing.T, client *fake.Clientset, wantDeletedPodNames, wantPatchedPodNames sets.Set[string]) {
	t.Helper()
	deletedPodNames := getDeletedPodNames(client)
	if diff := cmp.Diff(wantDeletedPodNames, deletedPodNames, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Deleted pod names (-want,+got):\n%s", diff)
	}
	patchedPodNames := getPatchedPodNames(client)
	if diff := cmp.Diff(wantPatchedPodNames, patchedPodNames, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Patched pod names (-want,+got):\n%s", diff)
	}
}

func testDeletingPodsMetrics(t *testing.T, total int, reason string) {
	t.Helper()

	actualDeletingPodsTotal, err := metricstestutil.GetCounterMetricValue(metrics.DeletingPodsTotal.WithLabelValues(metav1.NamespaceDefault, reason))
	if err != nil {
		t.Errorf("Error getting actualDeletingPodsTotal")
	}
	if actualDeletingPodsTotal != float64(total) {
		t.Errorf("Expected desiredDeletingPodsTotal to be %d, got %v", total, actualDeletingPodsTotal)
	}

	actualDeletingPodsErrorTotal, err := metricstestutil.GetCounterMetricValue(metrics.DeletingPodsErrorTotal.WithLabelValues("", reason))
	if err != nil {
		t.Errorf("Error getting actualDeletingPodsErrorTotal")
	}
	if actualDeletingPodsErrorTotal != float64(0) {
		t.Errorf("Expected desiredDeletingPodsTotal to be %d, got %v", 0, actualDeletingPodsErrorTotal)
	}
}

func setupNewSimpleClient(nodes []*v1.Node, pods []*v1.Pod) *fake.Clientset {
	podList := &v1.PodList{}
	for _, podItem := range pods {
		podList.Items = append(podList.Items, *podItem)
	}
	nodeList := &v1.NodeList{}
	for _, nodeItem := range nodes {
		nodeList.Items = append(nodeList.Items, *nodeItem)
	}
	return fake.NewSimpleClientset(nodeList, podList)
}

func getDeletedPodNames(client *fake.Clientset) sets.Set[string] {
	deletedPodNames := sets.New[string]()
	for _, action := range client.Actions() {
		if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
			deleteAction := action.(clienttesting.DeleteAction)
			deletedPodNames.Insert(deleteAction.GetName())
		}
	}
	return deletedPodNames
}

func getPatchedPodNames(client *fake.Clientset) sets.Set[string] {
	patchedPodNames := sets.New[string]()
	for _, action := range client.Actions() {
		if action.GetVerb() == "patch" && action.GetResource().Resource == "pods" {
			patchAction := action.(clienttesting.PatchAction)
			patchedPodNames.Insert(patchAction.GetName())
		}
	}
	return patchedPodNames
}

func resetMetrics() {
	metrics.DeletingPodsTotal.Reset()
	metrics.DeletingPodsErrorTotal.Reset()
}
