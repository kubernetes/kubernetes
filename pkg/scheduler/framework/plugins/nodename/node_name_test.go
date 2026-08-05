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

package nodename

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNodeName(t *testing.T) {
	tests := []struct {
		pod        *v1.Pod
		node       *v1.Node
		name       string
		wantStatus *fwk.Status
	}{
		{
			pod:  &v1.Pod{},
			node: &v1.Node{},
			name: "no host specified",
		},
		{
			pod:  st.MakePod().Node("foo").Obj(),
			node: st.MakeNode().Name("foo").Obj(),
			name: "host matches",
		},
		{
			pod:        st.MakePod().Node("bar").Obj(),
			node:       st.MakeNode().Name("foo").Obj(),
			name:       "host doesn't match",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReason),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(test.node)
			_, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			gotStatus := p.(fwk.FilterPlugin).Filter(ctx, nil, test.pod, nodeInfo)
			if diff := cmp.Diff(test.wantStatus, gotStatus); diff != "" {
				t.Errorf("status does not match (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestNodeName_PreFilter(t *testing.T) {
	tests := []struct {
		name       string
		pod        *v1.Pod
		wantResult *fwk.PreFilterResult
	}{
		{
			name: "assigned pod with nodeName set, prefilter restricts to assigned node",
			pod:  st.MakePod().Node("foo").Obj(),
			wantResult: &fwk.PreFilterResult{
				NodeNames: sets.New("foo"),
			},
		},
		{
			name: "unassigned pod without nodeName set, prefilter returns nil",
			pod:  st.MakePod().Obj(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			gotResult, gotStatus := p.(fwk.PreFilterPlugin).PreFilter(ctx, nil, tt.pod, nil)
			if gotStatus != nil && !gotStatus.IsSuccess() {
				t.Errorf("status does not match (-want,+got):\n-nil\n+%v", gotStatus)
			}
			if diff := cmp.Diff(tt.wantResult, gotResult); diff != "" {
				t.Errorf("result does not match (-want,+got):\n%s", diff)
			}
		})
	}
}
