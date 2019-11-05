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

func TestValidateNodeLabelArgs(t *testing.T) {
	// "bar" exists in both present and absent labels therefore validatio should fail.
	args := &runtime.Unknown{Raw: []byte(`{"presentLabels" : ["foo", "bar"], "absentLabels" : ["bar", "baz"]}`)}
	_, err := New(args, nil)
	if err == nil {
		t.Fatal("Plugin initialization should fail.")
	}
}

func TestNodeLabelFilter(t *testing.T) {
	label := map[string]string{"foo": "any value", "bar": "any value"}
	var pod *v1.Pod
	tests := []struct {
		name    string
		rawArgs string
		res     framework.Code
	}{
		{
			name:    "present label does not match",
			rawArgs: `{"presentLabels" : ["baz"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "absent label does not match",
			rawArgs: `{"absentLabels" : ["baz"]}`,
			res:     framework.Success,
		},
		{
			name:    "one of two present labels matches",
			rawArgs: `{"presentLabels" : ["foo", "baz"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "one of two absent labels matches",
			rawArgs: `{"absentLabels" : ["foo", "baz"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "all present abels match",
			rawArgs: `{"presentLabels" : ["foo", "bar"]}`,
			res:     framework.Success,
		},
		{
			name:    "all absent labels match",
			rawArgs: `{"absentLabels" : ["foo", "bar"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "both present and absent label matches",
			rawArgs: `{"presentLabels" : ["foo"], "absentLabels" : ["bar"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "neither present nor absent label matches",
			rawArgs: `{"presentLabels" : ["foz"], "absentLabels" : ["baz"]}`,
			res:     framework.UnschedulableAndUnresolvable,
		},
		{
			name:    "present label matches and absent label doesn't match",
			rawArgs: `{"presentLabels" : ["foo"], "absentLabels" : ["baz"]}`,
			res:     framework.Success,
		},
		{
			name:    "present label doesn't match and absent label matches",
			rawArgs: `{"presentLabels" : ["foz"], "absentLabels" : ["bar"]}`,
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

			status := p.(framework.FilterPlugin).Filter(context.TODO(), nil, pod, nodeInfo)
			if status.Code() != test.res {
				t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), test.res)
			}
		})
	}
}
