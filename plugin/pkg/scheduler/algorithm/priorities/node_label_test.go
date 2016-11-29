/*
Copyright 2016 The Kubernetes Authors.

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
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func TestNewNodeLabelPriority(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"bar": "foo"}
	label3 := map[string]string{"bar": "baz"}
	tests := []struct {
		nodes        []*v1.Node
		label        string
		presence     bool
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			label:        "baz",
			presence:     true,
			test:         "no match found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "baz",
			presence:     false,
			test:         "no match found, presence false",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			label:        "foo",
			presence:     true,
			test:         "one match found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "foo",
			presence:     false,
			test:         "one match found, presence false",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "bar",
			presence:     true,
			test:         "two matches found, presence true",
		},
		{
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			label:        "bar",
			presence:     false,
			test:         "two matches found, presence false",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(nil, test.nodes)
		list, err := priorityFunction(NewNodeLabelPriority(test.label, test.presence))(nil, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// sort the two lists to avoid failures on account of different ordering
		sort.Sort(test.expectedList)
		sort.Sort(list)
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}
