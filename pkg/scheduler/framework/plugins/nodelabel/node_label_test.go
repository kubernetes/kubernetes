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
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func TestNodeLabelFilter(t *testing.T) {
	label := map[string]string{"foo": "any value", "bar": "any value"}
	var pod *v1.Pod
	tests := []struct {
		name string
		args config.NodeLabelArgs
		res  framework.Code
	}{
		{
			name: "present label does not match",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"baz"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "absent label does not match",
			args: config.NodeLabelArgs{
				AbsentLabels: []string{"baz"},
			},
			res: framework.Success,
		},
		{
			name: "one of two present labels matches",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foo", "baz"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "one of two absent labels matches",
			args: config.NodeLabelArgs{
				AbsentLabels: []string{"foo", "baz"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "all present labels match",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foo", "bar"},
			},
			res: framework.Success,
		},
		{
			name: "all absent labels match",
			args: config.NodeLabelArgs{
				AbsentLabels: []string{"foo", "bar"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "both present and absent label matches",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foo"},
				AbsentLabels:  []string{"bar"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "neither present nor absent label matches",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foz"},
				AbsentLabels:  []string{"baz"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
		{
			name: "present label matches and absent label doesn't match",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foo"},
				AbsentLabels:  []string{"baz"},
			},
			res: framework.Success,
		},
		{
			name: "present label doesn't match and absent label matches",
			args: config.NodeLabelArgs{
				PresentLabels: []string{"foz"},
				AbsentLabels:  []string{"bar"},
			},
			res: framework.UnschedulableAndUnresolvable,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: label}}
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(&node)

			p, err := New(&test.args, nil)
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
		args config.NodeLabelArgs
		want int64
		name string
	}{
		{
			want: framework.MaxNodeScore,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"foo"},
			},
			name: "one present label match",
		},
		{
			want: 0,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"somelabel"},
			},
			name: "one present label mismatch",
		},
		{
			want: framework.MaxNodeScore,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"foo", "bar"},
			},
			name: "two present labels match",
		},
		{
			want: 0,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"somelabel1", "somelabel2"},
			},
			name: "two present labels mismatch",
		},
		{
			want: framework.MaxNodeScore / 2,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"foo", "somelabel"},
			},
			name: "two present labels only one matches",
		},
		{
			want: 0,
			args: config.NodeLabelArgs{
				AbsentLabelsPreference: []string{"foo"},
			},
			name: "one absent label match",
		},
		{
			want: framework.MaxNodeScore,
			args: config.NodeLabelArgs{
				AbsentLabelsPreference: []string{"somelabel"},
			},
			name: "one absent label mismatch",
		},
		{
			want: 0,
			args: config.NodeLabelArgs{
				AbsentLabelsPreference: []string{"foo", "bar"},
			},
			name: "two absent labels match",
		},
		{
			want: framework.MaxNodeScore,
			args: config.NodeLabelArgs{
				AbsentLabelsPreference: []string{"somelabel1", "somelabel2"},
			},
			name: "two absent labels mismatch",
		},
		{
			want: framework.MaxNodeScore / 2,
			args: config.NodeLabelArgs{
				AbsentLabelsPreference: []string{"foo", "somelabel"},
			},
			name: "two absent labels only one matches",
		},
		{
			want: framework.MaxNodeScore,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"foo", "bar"},
				AbsentLabelsPreference:  []string{"somelabel1", "somelabel2"},
			},
			name: "two present labels match, two absent labels mismatch",
		},
		{
			want: 0,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"somelabel1", "somelabel2"},
				AbsentLabelsPreference:  []string{"foo", "bar"},
			},
			name: "two present labels both mismatch, two absent labels both match",
		},
		{
			want: 3 * framework.MaxNodeScore / 4,
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"foo", "somelabel"},
				AbsentLabelsPreference:  []string{"somelabel1", "somelabel2"},
			},
			name: "two present labels one matches, two absent labels mismatch",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: map[string]string{"foo": "", "bar": ""}}}
			fh, _ := runtime.NewFramework(nil, nil, nil, runtime.WithSnapshotSharedLister(cache.NewSnapshot(nil, []*v1.Node{node})))
			p, err := New(&test.args, fh)
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
		nodeInfo := framework.NewNodeInfo()
		p, err := New(&config.NodeLabelArgs{}, nil)
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
		fh, _ := runtime.NewFramework(nil, nil, nil, runtime.WithSnapshotSharedLister(cache.NewEmptySnapshot()))
		p, err := New(&config.NodeLabelArgs{}, fh)
		if err != nil {
			t.Fatalf("Failed to create plugin: %+v", err)
		}
		_, status := p.(framework.ScorePlugin).Score(context.Background(), nil, nil, "")
		if status.Code() != framework.Error {
			t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), framework.Error)
		}
	})
}
