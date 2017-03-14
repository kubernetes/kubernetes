/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"sort"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/controller/node/testutil"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clienttesting "k8s.io/client-go/testing"
)

func createNoExecuteTaint(index int) v1.Taint {
	return v1.Taint{
		Key:       "testTaint" + fmt.Sprintf("%v", index),
		Value:     "test" + fmt.Sprintf("%v", index),
		Effect:    v1.TaintEffectNoExecute,
		TimeAdded: metav1.Now(),
	}
}

func addToleration(pod *v1.Pod, index int, duration int64) *v1.Pod {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if duration < 0 {
		pod.Spec.Tolerations = []v1.Toleration{{Key: "testTaint" + fmt.Sprintf("%v", index), Value: "test" + fmt.Sprintf("%v", index), Effect: v1.TaintEffectNoExecute}}

	} else {
		pod.Spec.Tolerations = []v1.Toleration{{Key: "testTaint" + fmt.Sprintf("%v", index), Value: "test" + fmt.Sprintf("%v", index), Effect: v1.TaintEffectNoExecute, TolerationSeconds: &duration}}
	}
	return pod
}

func addTaintsToNode(node *v1.Node, key, value string, indices []int) *v1.Node {
	taints := []v1.Taint{}
	for _, index := range indices {
		taints = append(taints, createNoExecuteTaint(index))
	}
	node.Spec.Taints = taints
	return node
}

type timestampedPod struct {
	names     []string
	timestamp time.Duration
}

type durationSlice []timestampedPod

func (a durationSlice) Len() int           { return len(a) }
func (a durationSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a durationSlice) Less(i, j int) bool { return a[i].timestamp < a[j].timestamp }

func TestFilterNoExecuteTaints(t *testing.T) {
	taints := []v1.Taint{
		{
			Key:    "one",
			Value:  "one",
			Effect: v1.TaintEffectNoExecute,
		},
		{
			Key:    "two",
			Value:  "two",
			Effect: v1.TaintEffectNoSchedule,
		},
	}
	taints = getNoExecuteTaints(taints)
	if len(taints) != 1 || taints[0].Key != "one" {
		t.Errorf("Filtering doesn't work. Got %v", taints)
	}
}

func TestCreatePod(t *testing.T) {
	testCases := []struct {
		description  string
		pod          *v1.Pod
		taintedNodes map[string][]v1.Taint
		expectDelete bool
	}{
		{
			description:  "not scheduled - ignore",
			pod:          testutil.NewPod("pod1", ""),
			taintedNodes: map[string][]v1.Taint{},
			expectDelete: false,
		},
		{
			description:  "scheduled on untainted Node",
			pod:          testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]v1.Taint{},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node",
			pod:         testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: true,
		},
		{
			description: "schedule on tainted Node with finite toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node with infinite toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node with infinite ivalid toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 2, -1),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: true,
		},
	}

	for _, item := range testCases {
		stopCh := make(chan struct{})
		fakeClientset := fake.NewSimpleClientset()
		controller := NewNoExecuteTaintManager(fakeClientset)
		controller.recorder = testutil.NewFakeRecorder()
		go controller.Run(stopCh)
		controller.taintedNodes = item.taintedNodes
		controller.PodUpdated(nil, item.pod)
		// wait a bit
		time.Sleep(200 * time.Millisecond)

		podDeleted := false
		for _, action := range fakeClientset.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podDeleted = true
			}
		}
		if podDeleted != item.expectDelete {
			t.Errorf("%v: Unexepected test result. Expected delete %v, got %v", item.description, item.expectDelete, podDeleted)
		}
		close(stopCh)
	}
}

func TestDeletePod(t *testing.T) {
	stopCh := make(chan struct{})
	fakeClientset := fake.NewSimpleClientset()
	controller := NewNoExecuteTaintManager(fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	go controller.Run(stopCh)
	controller.taintedNodes = map[string][]v1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	controller.PodUpdated(testutil.NewPod("pod1", "node1"), nil)
	// wait a bit to see if nothing will panic
	time.Sleep(200 * time.Millisecond)
	close(stopCh)
}

func TestUpdatePod(t *testing.T) {
	testCases := []struct {
		description     string
		prevPod         *v1.Pod
		newPod          *v1.Pod
		taintedNodes    map[string][]v1.Taint
		expectDelete    bool
		additionalSleep time.Duration
	}{
		{
			description: "scheduling onto tainted Node",
			prevPod:     testutil.NewPod("pod1", ""),
			newPod:      testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: true,
		},
		{
			description: "scheduling onto tainted Node with toleration",
			prevPod:     addToleration(testutil.NewPod("pod1", ""), 1, -1),
			newPod:      addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description: "removing toleration",
			prevPod:     addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			newPod:      testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: true,
		},
		{
			description: "lengthening toleration shouldn't work",
			prevPod:     addToleration(testutil.NewPod("pod1", "node1"), 1, 1),
			newPod:      addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			taintedNodes: map[string][]v1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete:    true,
			additionalSleep: 1500 * time.Millisecond,
		},
	}

	for _, item := range testCases {
		stopCh := make(chan struct{})
		fakeClientset := fake.NewSimpleClientset()
		controller := NewNoExecuteTaintManager(fakeClientset)
		controller.recorder = testutil.NewFakeRecorder()
		go controller.Run(stopCh)
		controller.taintedNodes = item.taintedNodes

		controller.PodUpdated(nil, item.prevPod)
		fakeClientset.ClearActions()
		time.Sleep(200 * time.Millisecond)
		controller.PodUpdated(item.prevPod, item.newPod)
		// wait a bit
		time.Sleep(200 * time.Millisecond)
		if item.additionalSleep > 0 {
			time.Sleep(item.additionalSleep)
		}

		podDeleted := false
		for _, action := range fakeClientset.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podDeleted = true
			}
		}
		if podDeleted != item.expectDelete {
			t.Errorf("%v: Unexepected test result. Expected delete %v, got %v", item.description, item.expectDelete, podDeleted)
		}
		close(stopCh)
	}
}

func TestCreateNode(t *testing.T) {
	testCases := []struct {
		description  string
		pods         []v1.Pod
		node         *v1.Node
		expectDelete bool
	}{
		{
			description: "Creating Node maching already assigned Pod",
			pods: []v1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			node:         testutil.NewNode("node1"),
			expectDelete: false,
		},
		{
			description: "Creating tainted Node maching already assigned Pod",
			pods: []v1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			node:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectDelete: true,
		},
		{
			description: "Creating tainted Node maching already assigned tolerating Pod",
			pods: []v1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			},
			node:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectDelete: false,
		},
	}

	for _, item := range testCases {
		stopCh := make(chan struct{})
		fakeClientset := fake.NewSimpleClientset(&v1.PodList{Items: item.pods})
		controller := NewNoExecuteTaintManager(fakeClientset)
		controller.recorder = testutil.NewFakeRecorder()
		go controller.Run(stopCh)
		controller.NodeUpdated(nil, item.node)
		// wait a bit
		time.Sleep(200 * time.Millisecond)

		podDeleted := false
		for _, action := range fakeClientset.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podDeleted = true
			}
		}
		if podDeleted != item.expectDelete {
			t.Errorf("%v: Unexepected test result. Expected delete %v, got %v", item.description, item.expectDelete, podDeleted)
		}
		close(stopCh)
	}
}

func TestDeleteNode(t *testing.T) {
	stopCh := make(chan struct{})
	fakeClientset := fake.NewSimpleClientset()
	controller := NewNoExecuteTaintManager(fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]v1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	go controller.Run(stopCh)
	controller.NodeUpdated(testutil.NewNode("node1"), nil)
	// wait a bit to see if nothing will panic
	time.Sleep(200 * time.Millisecond)
	controller.taintedNodesLock.Lock()
	if _, ok := controller.taintedNodes["node1"]; ok {
		t.Error("Node should have been deleted from taintedNodes list")
	}
	controller.taintedNodesLock.Unlock()
	close(stopCh)
}

func TestUpdateNode(t *testing.T) {
	testCases := []struct {
		description     string
		pods            []v1.Pod
		oldNode         *v1.Node
		newNode         *v1.Node
		expectDelete    bool
		additionalSleep time.Duration
	}{
		{
			description: "Added taint",
			pods: []v1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectDelete: true,
		},
		{
			description: "Added tolerated taint",
			pods: []v1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectDelete: false,
		},
		{
			description: "Only one added taint tolerated",
			pods: []v1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectDelete: true,
		},
		{
			description: "Taint removed",
			pods: []v1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 1),
			},
			oldNode:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			newNode:         testutil.NewNode("node1"),
			expectDelete:    false,
			additionalSleep: 1500 * time.Millisecond,
		},
		{
			description: "Pod with multiple tolerations are evicted when first one runs out",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "pod1",
					},
					Spec: v1.PodSpec{
						NodeName: "node1",
						Tolerations: []v1.Toleration{
							{Key: "testTaint1", Value: "test1", Effect: v1.TaintEffectNoExecute, TolerationSeconds: &[]int64{1}[0]},
							{Key: "testTaint2", Value: "test2", Effect: v1.TaintEffectNoExecute, TolerationSeconds: &[]int64{100}[0]},
						},
					},
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			oldNode:         testutil.NewNode("node1"),
			newNode:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectDelete:    true,
			additionalSleep: 1500 * time.Millisecond,
		},
	}

	for _, item := range testCases {
		stopCh := make(chan struct{})
		fakeClientset := fake.NewSimpleClientset(&v1.PodList{Items: item.pods})
		controller := NewNoExecuteTaintManager(fakeClientset)
		controller.recorder = testutil.NewFakeRecorder()
		go controller.Run(stopCh)
		controller.NodeUpdated(item.oldNode, item.newNode)
		// wait a bit
		time.Sleep(200 * time.Millisecond)
		if item.additionalSleep > 0 {
			time.Sleep(item.additionalSleep)
		}

		podDeleted := false
		for _, action := range fakeClientset.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podDeleted = true
			}
		}
		if podDeleted != item.expectDelete {
			t.Errorf("%v: Unexepected test result. Expected delete %v, got %v", item.description, item.expectDelete, podDeleted)
		}
		close(stopCh)
	}
}

func TestUpdateNodeWithMultiplePods(t *testing.T) {
	testCases := []struct {
		description         string
		pods                []v1.Pod
		oldNode             *v1.Node
		newNode             *v1.Node
		expectedDeleteTimes durationSlice
	}{
		{
			description: "Pods with different toleration times are evicted appropriately",
			pods: []v1.Pod{
				*testutil.NewPod("pod1", "node1"),
				*addToleration(testutil.NewPod("pod2", "node1"), 1, 1),
				*addToleration(testutil.NewPod("pod3", "node1"), 1, -1),
			},
			oldNode: testutil.NewNode("node1"),
			newNode: addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectedDeleteTimes: durationSlice{
				{[]string{"pod1"}, 0},
				{[]string{"pod2"}, time.Second},
			},
		},
		{
			description: "Evict all pods not maching all taints instantly",
			pods: []v1.Pod{
				*testutil.NewPod("pod1", "node1"),
				*addToleration(testutil.NewPod("pod2", "node1"), 1, 1),
				*addToleration(testutil.NewPod("pod3", "node1"), 1, -1),
			},
			oldNode: testutil.NewNode("node1"),
			newNode: addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectedDeleteTimes: durationSlice{
				{[]string{"pod1", "pod2", "pod3"}, 0},
			},
		},
	}

	for _, item := range testCases {
		stopCh := make(chan struct{})
		fakeClientset := fake.NewSimpleClientset(&v1.PodList{Items: item.pods})
		sort.Sort(item.expectedDeleteTimes)
		controller := NewNoExecuteTaintManager(fakeClientset)
		controller.recorder = testutil.NewFakeRecorder()
		go controller.Run(stopCh)
		controller.NodeUpdated(item.oldNode, item.newNode)

		sleptAlready := time.Duration(0)
		for i := range item.expectedDeleteTimes {
			var increment time.Duration
			if i == 0 || item.expectedDeleteTimes[i-1].timestamp != item.expectedDeleteTimes[i].timestamp {
				if i == len(item.expectedDeleteTimes)-1 || item.expectedDeleteTimes[i+1].timestamp == item.expectedDeleteTimes[i].timestamp {
					increment = 200 * time.Millisecond
				} else {
					increment = ((item.expectedDeleteTimes[i+1].timestamp - item.expectedDeleteTimes[i].timestamp) / time.Duration(2))
				}
				sleepTime := item.expectedDeleteTimes[i].timestamp - sleptAlready + increment
				glog.Infof("Sleeping for %v", sleepTime)
				time.Sleep(sleepTime)
				sleptAlready = item.expectedDeleteTimes[i].timestamp + increment
			}

			for delay, podName := range item.expectedDeleteTimes[i].names {
				deleted := false
				for _, action := range fakeClientset.Actions() {
					deleteAction, ok := action.(clienttesting.DeleteActionImpl)
					if !ok {
						glog.Infof("Found not-delete action with verb %v. Ignoring.", action.GetVerb())
						continue
					}
					if deleteAction.GetResource().Resource != "pods" {
						continue
					}
					if podName == deleteAction.GetName() {
						deleted = true
					}
				}
				if !deleted {
					t.Errorf("Failed to deleted pod %v after %v", podName, delay)
				}
			}
			for _, action := range fakeClientset.Actions() {
				deleteAction, ok := action.(clienttesting.DeleteActionImpl)
				if !ok {
					glog.Infof("Found not-delete action with verb %v. Ignoring.", action.GetVerb())
					continue
				}
				if deleteAction.GetResource().Resource != "pods" {
					continue
				}
				deletedPodName := deleteAction.GetName()
				expected := false
				for _, podName := range item.expectedDeleteTimes[i].names {
					if podName == deletedPodName {
						expected = true
					}
				}
				if !expected {
					t.Errorf("Pod %v was deleted even though it shouldn't have", deletedPodName)
				}
			}
			fakeClientset.ClearActions()
		}

		close(stopCh)
	}
}

func TestGetMinTolerationTime(t *testing.T) {
	one := int64(1)
	oneSec := 1 * time.Second

	tests := []struct {
		tolerations []v1.Toleration
		expected    time.Duration
	}{
		{
			tolerations: []v1.Toleration{},
			expected:    0,
		},
		{
			tolerations: []v1.Toleration{
				{
					TolerationSeconds: &one,
				},
				{
					TolerationSeconds: nil,
				},
			},
			expected: oneSec,
		},
		{
			tolerations: []v1.Toleration{
				{
					TolerationSeconds: nil,
				},
				{
					TolerationSeconds: &one,
				},
			},
			expected: oneSec,
		},
	}

	for _, test := range tests {
		got := getMinTolerationTime(test.tolerations)
		if got != test.expected {
			t.Errorf("Incorrect min toleration time: got %v, expected %v", got, test.expected)
		}
	}
}
