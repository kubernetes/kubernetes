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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
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
