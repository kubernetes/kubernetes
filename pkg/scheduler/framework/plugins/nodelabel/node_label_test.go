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
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestValidateNodeLabelArgs(t *testing.T) {
	tests := []struct {
		name string
		args string
		err  bool
	}{
		{
			name: "happy case",
			args: `{"presentLabels" : ["foo", "bar"], "absentLabels" : ["baz"], "presentLabelsPreference" : ["foo", "bar"], "absentLabelsPreference" : ["baz"]}`,
		},
		{
			name: "label presence conflict",
			// "bar" exists in both present and absent labels therefore validation should fail.
			args: `{"presentLabels" : ["foo", "bar"], "absentLabels" : ["bar", "baz"], "presentLabelsPreference" : ["foo", "bar"], "absentLabelsPreference" : ["baz"]}`,
			err:  true,
		},
		{
			name: "label preference conflict",
			// "bar" exists in both present and absent labels preferences therefore validation should fail.
			args: `{"presentLabels" : ["foo", "bar"], "absentLabels" : ["baz"], "presentLabelsPreference" : ["foo", "bar"], "absentLabelsPreference" : ["bar", "baz"]}`,
			err:  true,
		},
		{
			name: "both label presence and preference conflict",
			args: `{"presentLabels" : ["foo", "bar"], "absentLabels" : ["bar", "baz"], "presentLabelsPreference" : ["foo", "bar"], "absentLabelsPreference" : ["bar", "baz"]}`,
			err:  true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			args := &runtime.Unknown{Raw: []byte(test.args)}
			_, err := New(args, nil)
			if test.err && err == nil {
				t.Fatal("Plugin initialization should fail.")
			}
			if !test.err && err != nil {
				t.Fatalf("Plugin initialization shouldn't fail: %v", err)
			}
		})
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

func TestNodeLabelScore(t *testing.T) {
	tests := []struct {
		rawArgs string
		want    int64
		name    string
	}{
		{
			want:    framework.MaxNodeScore,
			rawArgs: `{"presentLabelsPreference" : ["foo"]}`,
			name:    "one present label match",
		},
		{
			want:    0,
			rawArgs: `{"presentLabelsPreference" : ["somelabel"]}`,
			name:    "one present label mismatch",
		},
		{
			want:    framework.MaxNodeScore,
			rawArgs: `{"presentLabelsPreference" : ["foo", "bar"]}`,
			name:    "two present labels match",
		},
		{
			want:    0,
			rawArgs: `{"presentLabelsPreference" : ["somelabel1", "somelabel2"]}`,
			name:    "two present labels mismatch",
		},
		{
			want:    framework.MaxNodeScore / 2,
			rawArgs: `{"presentLabelsPreference" : ["foo", "somelabel"]}`,
			name:    "two present labels only one matches",
		},
		{
			want:    0,
			rawArgs: `{"absentLabelsPreference" : ["foo"]}`,
			name:    "one absent label match",
		},
		{
			want:    framework.MaxNodeScore,
			rawArgs: `{"absentLabelsPreference" : ["somelabel"]}`,
			name:    "one absent label mismatch",
		},
		{
			want:    0,
			rawArgs: `{"absentLabelsPreference" : ["foo", "bar"]}`,
			name:    "two absent labels match",
		},
		{
			want:    framework.MaxNodeScore,
			rawArgs: `{"absentLabelsPreference" : ["somelabel1", "somelabel2"]}`,
			name:    "two absent labels mismatch",
		},
		{
			want:    framework.MaxNodeScore / 2,
			rawArgs: `{"absentLabelsPreference" : ["foo", "somelabel"]}`,
			name:    "two absent labels only one matches",
		},
		{
			want:    framework.MaxNodeScore,
			rawArgs: `{"presentLabelsPreference" : ["foo", "bar"], "absentLabelsPreference" : ["somelabel1", "somelabel2"]}`,
			name:    "two present labels match, two absent labels mismatch",
		},
		{
			want:    0,
			rawArgs: `{"absentLabelsPreference" : ["foo", "bar"], "presentLabelsPreference" : ["somelabel1", "somelabel2"]}`,
			name:    "two present labels both mismatch, two absent labels both match",
		},
		{
			want:    3 * framework.MaxNodeScore / 4,
			rawArgs: `{"presentLabelsPreference" : ["foo", "somelabel"], "absentLabelsPreference" : ["somelabel1", "somelabel2"]}`,
			name:    "two present labels one matches, two absent labels mismatch",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: map[string]string{"foo": "", "bar": ""}}}
			fh, _ := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(cache.NewSnapshot(nil, []*v1.Node{node})))
			args := &runtime.Unknown{Raw: []byte(test.rawArgs)}
			p, err := New(args, fh)
			if err != nil {
				t.Fatalf("Failed to create plugin: %+v", err)
			}
			nodeName := node.ObjectMeta.Name
			score, status := p.(framework.ScorePlugin).Score(context.Background(), state, nil, nodeName)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}
			if test.want != score {
				t.Errorf("Wrong score. got %#v, want %#v", score, test.want)
			}
		})
	}
}

func TestNodeLabelFilterWithoutNode(t *testing.T) {
	var pod *v1.Pod
	t.Run("node does not exist", func(t *testing.T) {
		nodeInfo := schedulernodeinfo.NewNodeInfo()
		p, err := New(nil, nil)
		if err != nil {
			t.Fatalf("Failed to create plugin: %v", err)
		}
		status := p.(framework.FilterPlugin).Filter(context.TODO(), nil, pod, nodeInfo)
		if status.Code() != framework.Error {
			t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), framework.Error)
		}
	})
}

func TestNodeLabelScoreWithoutNode(t *testing.T) {
	t.Run("node does not exist", func(t *testing.T) {
		fh, _ := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(cache.NewEmptySnapshot()))
		p, err := New(nil, fh)
		if err != nil {
			t.Fatalf("Failed to create plugin: %+v", err)
		}
		_, status := p.(framework.ScorePlugin).Score(context.Background(), nil, nil, "")
		if status.Code() != framework.Error {
			t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), framework.Error)
		}
	})

}
