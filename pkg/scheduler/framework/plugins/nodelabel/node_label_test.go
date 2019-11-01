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

package nodelabel

import (
	"context"
	"reflect"
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
)

func TestNodeLabelPresence(t *testing.T) {
	label := map[string]string{"foo": "bar", "bar": "foo"}
	tests := []struct {
		name    string
		pod     *v1.Pod
		rawArgs string
		res     framework.Code
	}{
		{
			name:    "label does not match, presence true",
			rawArgs: `{"labels" : ["baz"], "presence" : true}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "label does not match, presence false",
			rawArgs: `{"labels" : ["baz"], "presence" : false}`,
			res:     framework.Success,
		},
		{
			name:    "one label matches, presence true",
			rawArgs: `{"labels" : ["foo", "baz"], "presence" : true}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "one label matches, presence false",
			rawArgs: `{"labels" : ["foo", "baz"], "presence" : false}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "all labels match, presence true",
			rawArgs: `{"labels" : ["foo", "bar"], "presence" : true}`,
			res:     framework.Success,
		},
		{
			name:    "all labels match, presence false",
			rawArgs: `{"labels" : ["foo", "bar"], "presence" : false}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: label}}
			nodeInfo := schedulernodeinfo.NewNodeInfo()
			nodeInfo.SetNode(&node)

			args := &runtime.Unknown{Raw: []byte(test.rawArgs)}
			p, err := New(args, nil)
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}

			status := p.(framework.FilterPlugin).Filter(context.TODO(), nil, test.pod, nodeInfo)
			if status.Code() != test.res {
				t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), test.res)
			}
		})
	}
}

func TestNodeLabelScore(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"bar": "foo"}
	label3 := map[string]string{"bar": "baz"}
	tests := []struct {
		nodes        []*v1.Node
		rawArgs      string
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}, {Name: "machine3", Score: 0}},
			rawArgs:      `{"preferenceLabel" : "baz", "preferenceLabelPresence" : true}`,
			name:         "no match found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}, {Name: "machine3", Score: framework.MaxNodeScore}},
			rawArgs:      `{"preferenceLabel" : "baz", "preferenceLabelPresence" : false}`,
			name:         "no match found, presence false",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}, {Name: "machine3", Score: 0}},
			rawArgs:      `{"preferenceLabel" : "foo", "preferenceLabelPresence" : true}`,
			name:         "one match found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: framework.MaxNodeScore}, {Name: "machine3", Score: framework.MaxNodeScore}},
			rawArgs:      `{"preferenceLabel" : "foo", "preferenceLabelPresence" : false}`,
			name:         "one match found, presence false",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: framework.MaxNodeScore}, {Name: "machine3", Score: framework.MaxNodeScore}},
			rawArgs:      `{"preferenceLabel" : "bar", "preferenceLabelPresence" : true}`,
			name:         "two matches found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}, {Name: "machine3", Score: 0}},
			rawArgs:      `{"preferenceLabel" : "bar", "preferenceLabelPresence" : false}`,
			name:         "two matches found, presence false",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			fh, _ := framework.NewFramework(nil, nil, nil, framework.WithNodeInfoSnapshot(nodeinfosnapshot.NewSnapshot(nil, test.nodes)))
			args := &runtime.Unknown{Raw: []byte(test.rawArgs)}
			p, _ := New(args, fh)

			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				nodeName := n.ObjectMeta.Name
				score, status := p.(framework.ScorePlugin).Score(context.Background(), state, nil, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}
			// sort the two lists to avoid failures on account of different ordering
			sortNodeScoreList(test.expectedList)
			sortNodeScoreList(gotList)
			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected %#v, got %#v", test.expectedList, gotList)
			}
		})
	}
}

func sortNodeScoreList(out framework.NodeScoreList) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Name < out[j].Name
		}
		return out[i].Score < out[j].Score
	})
}
