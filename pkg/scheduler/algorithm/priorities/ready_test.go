/*
Copyright 2019 The Kubernetes Authors.

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

package priorities

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestReadyPodPriority(t *testing.T) {

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			/*
			   Test case: 3 of 3 nodes with equal non ready pods
			   Total non ready pods across cluster: 3

			   Node1 scores on 0-10 scale
			   Node1 non ready pods: 1
			   Node1 score: 10 - ( 1 / 3 ) * 10 = math.Round(6.67) = 7

			   Node2 scores on 0-10 scale
			   Node2 non ready pods: 1
			   Node2 score: 10 - ( 1 / 3 ) * 10 = math.Round(6.67) = 7

			   Node3 scores on 0-10 scale
			   Node3 non ready pods: 1
			   Node3 score: 10 - ( 1 / 3 ) * 10 = math.Round(6.67) = 7
			*/
			name: "3 of 3 nodes with equal non ready pods",
			pod:  &v1.Pod{Spec: v1.PodSpec{NodeName: ""}},
			pods: []*v1.Pod{
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node1"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node2"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node3"}},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node3"}},
			},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 7}, {Name: "node2", Score: 7}, {Name: "node3", Score: 7}},
		},
		{
			/*
			   Test case: 3 of 3 nodes with zero non ready pods
			   Total non ready pods across cluster: 0

			   When all nodes are w/o non running pods, they are all scored with 10.
			*/
			name: "3 of 3 nodes with zero non ready pods",
			pod:  &v1.Pod{Spec: v1.PodSpec{NodeName: ""}},
			pods: []*v1.Pod{
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node1"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node2"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node3"}},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node3"}},
			},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 10}, {Name: "node2", Score: 10}, {Name: "node3", Score: 10}},
		},
		{
			/*
			   Test case: 3 nodes with varying non ready pods
			   Total non ready pods across cluster: 5

			   Node1 scores on 0-10 scale
			   Node1 non ready pods: 0
			   Node1 score: 10 - ( 0 / 5 ) * 10 = math.Round(10) = 10

			   Node2 scores on 0-10 scale
			   Node2 non ready pods: 1
			   Node2 score: 10 - ( 1 / 5 ) * 10 = math.Round(8) = 8

			   Node3 scores on 0-10 scale
			   Node3 non ready pods: 4
			   Node3 score: 10 - ( 4 / 5 ) * 10 = math.Round(2) = 2
			*/
			name: "3 nodes with varying non ready pods",
			pod:  &v1.Pod{Spec: v1.PodSpec{NodeName: ""}},
			pods: []*v1.Pod{
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node1"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node1"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node2"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node2"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}}, Spec: v1.PodSpec{NodeName: "node3"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node3"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node3"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node3"}},
				{Status: v1.PodStatus{Conditions: []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionFalse}}}, Spec: v1.PodSpec{NodeName: "node3"}},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node3"}},
			},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 10}, {Name: "node2", Score: 8}, {Name: "node3", Score: 2}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, test.nodes)
			readyPod := ReadyPod{}
			list, err := readyPod.CalculateReadyPodPriority(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected \n\t%#v, \ngot \n\t%#v\n", test.expectedList, list)
			}
		})
	}
}
