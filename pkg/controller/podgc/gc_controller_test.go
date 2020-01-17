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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

type FakeController struct{}

func (*FakeController) Run(<-chan struct{}) {}

func (*FakeController) HasSynced() bool {
	return true
}

func (*FakeController) LastSyncResourceVersion() string {
	return ""
}

func alwaysReady() bool { return true }

func NewFromClient(kubeClient clientset.Interface, terminatedPodThreshold int) (*PodGCController, coreinformers.PodInformer, coreinformers.NodeInformer) {
	informerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	nodeInformer := informerFactory.Core().V1().Nodes()
	controller := NewPodGC(kubeClient, podInformer, nodeInformer, terminatedPodThreshold)
	controller.podListerSynced = alwaysReady
	return controller, podInformer, nodeInformer
}

func compareStringSetToList(set sets.String, list []string) bool {
	for _, item := range list {
		if !set.Has(item) {
			return false
		}
	}
	if len(list) != len(set) {
		return false
	}
	return true
}

func TestGCTerminated(t *testing.T) {
	type nameToPhase struct {
		name  string
		phase v1.PodPhase
	}

	testCases := []struct {
		pods            []nameToPhase
		threshold       int
		deletedPodNames sets.String
	}{
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold: 0,
			// threshold = 0 disables terminated pod deletion
			deletedPodNames: sets.NewString(),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a", "b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodRunning},
				{name: "b", phase: v1.PodSucceeded},
				{name: "c", phase: v1.PodFailed},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("b"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       1,
			deletedPodNames: sets.NewString("a"),
		},
		{
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed},
				{name: "b", phase: v1.PodSucceeded},
			},
			threshold:       5,
			deletedPodNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset(&v1.NodeList{Items: []v1.Node{*testutil.NewNode("node")}})
		gcc, podInformer, _ := NewFromClient(client, test.threshold)
		deletedPodNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deletePod = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedPodNames = append(deletedPodNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			podInformer.Informer().GetStore().Add(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: pod.name, CreationTimestamp: metav1.Time{Time: creationTime}},
				Status:     v1.PodStatus{Phase: pod.phase},
				Spec:       v1.PodSpec{NodeName: "node"},
			})
		}

		gcc.gc()

		if pass := compareStringSetToList(test.deletedPodNames, deletedPodNames); !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v",
				i, test.deletedPodNames.List(), deletedPodNames)
		}
	}
}

func makePod(name string, nodeName string, phase v1.PodPhase) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec:   v1.PodSpec{NodeName: nodeName},
		Status: v1.PodStatus{Phase: phase},
	}
}

func waitForAdded(q workqueue.DelayingInterface, depth int) error {
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
		deletedPodNames      sets.String
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
			itemsInQueue:    0,
			deletedPodNames: sets.NewString(),
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
			itemsInQueue:    2,
			deletedPodNames: sets.NewString(),
		},
		{
			name:  "no nodes",
			delay: 2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
				makePod("b", "deleted", v1.PodSucceeded),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.NewString("a", "b"),
		},
		{
			name:  "quarantine not finished",
			delay: quarantineTime / 2,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodFailed),
			},
			itemsInQueue:    0,
			deletedPodNames: sets.NewString(),
		},
		{
			name:                 "wrong nodes",
			initialInformerNodes: []*v1.Node{testutil.NewNode("existing")},
			delay:                2 * quarantineTime,
			pods: []*v1.Pod{
				makePod("a", "deleted", v1.PodRunning),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.NewString("a"),
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
			deletedPodNames: sets.NewString("a", "c", "d"),
		},
		{
			name:             "node added to client after quarantine",
			delay:            2 * quarantineTime,
			addedClientNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodRunning),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.NewString(),
		},
		{
			name:               "node added to informer after quarantine",
			delay:              2 * quarantineTime,
			addedInformerNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodFailed),
			},
			itemsInQueue:    1,
			deletedPodNames: sets.NewString(),
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
			deletedPodNames: sets.NewString("a"),
		},
		{
			name:                 "node deleted from informer after quarantine",
			initialInformerNodes: []*v1.Node{testutil.NewNode("node")},
			delay:                2 * quarantineTime,
			deletedInformerNodes: []*v1.Node{testutil.NewNode("node")},
			pods: []*v1.Pod{
				makePod("a", "node", v1.PodSucceeded),
			},
			itemsInQueue:    0,
			deletedPodNames: sets.NewString(),
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			nodeList := &v1.NodeList{}
			for _, node := range test.initialClientNodes {
				nodeList.Items = append(nodeList.Items, *node)
			}
			client := fake.NewSimpleClientset(nodeList)
			gcc, podInformer, nodeInformer := NewFromClient(client, -1)
			for _, node := range test.initialInformerNodes {
				nodeInformer.Informer().GetStore().Add(node)
			}
			for _, pod := range test.pods {
				podInformer.Informer().GetStore().Add(pod)
			}
			// Overwrite queue
			fakeClock := clock.NewFakeClock(time.Now())
			gcc.nodeQueue.ShutDown()
			gcc.nodeQueue = workqueue.NewDelayingQueueWithCustomClock(fakeClock, "podgc_test_queue")

			deletedPodNames := make([]string, 0)
			var lock sync.Mutex
			gcc.deletePod = func(_, name string) error {
				lock.Lock()
				defer lock.Unlock()
				deletedPodNames = append(deletedPodNames, name)
				return nil
			}

			// First GC of orphaned pods
			gcc.gc()
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
				client.CoreV1().Nodes().Create(node)
			}
			for _, node := range test.deletedClientNodes {
				client.CoreV1().Nodes().Delete(node.Name, &metav1.DeleteOptions{})
			}
			for _, node := range test.addedInformerNodes {
				nodeInformer.Informer().GetStore().Add(node)
			}
			for _, node := range test.deletedInformerNodes {
				nodeInformer.Informer().GetStore().Delete(node)
			}

			// Actual pod deletion
			gcc.gc()

			if pass := compareStringSetToList(test.deletedPodNames, deletedPodNames); !pass {
				t.Errorf("pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v",
					test.deletedPodNames.List(), deletedPodNames)
			}
		})
	}
}

func TestGCUnscheduledTerminating(t *testing.T) {
	type nameToPhase struct {
		name              string
		phase             v1.PodPhase
		deletionTimeStamp *metav1.Time
		nodeName          string
	}

	testCases := []struct {
		name            string
		pods            []nameToPhase
		deletedPodNames sets.String
	}{
		{
			name: "Unscheduled pod in any phase must be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: ""},
			},
			deletedPodNames: sets.NewString("a", "b", "c"),
		},
		{
			name: "Scheduled pod in any phase must not be deleted",
			pods: []nameToPhase{
				{name: "a", phase: v1.PodFailed, deletionTimeStamp: nil, nodeName: ""},
				{name: "b", phase: v1.PodSucceeded, deletionTimeStamp: nil, nodeName: "node"},
				{name: "c", phase: v1.PodRunning, deletionTimeStamp: &metav1.Time{}, nodeName: "node"},
			},
			deletedPodNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset()
		gcc, podInformer, _ := NewFromClient(client, -1)
		deletedPodNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deletePod = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedPodNames = append(deletedPodNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, pod := range test.pods {
			creationTime = creationTime.Add(1 * time.Hour)
			podInformer.Informer().GetStore().Add(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: pod.name, CreationTimestamp: metav1.Time{Time: creationTime},
					DeletionTimestamp: pod.deletionTimeStamp},
				Status: v1.PodStatus{Phase: pod.phase},
				Spec:   v1.PodSpec{NodeName: pod.nodeName},
			})
		}

		pods, err := podInformer.Lister().List(labels.Everything())
		if err != nil {
			t.Errorf("Error while listing all Pods: %v", err)
			return
		}
		gcc.gcUnscheduledTerminating(pods)

		if pass := compareStringSetToList(test.deletedPodNames, deletedPodNames); !pass {
			t.Errorf("[%v]pod's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v, test: %v",
				i, test.deletedPodNames.List(), deletedPodNames, test.name)
		}
	}
}
