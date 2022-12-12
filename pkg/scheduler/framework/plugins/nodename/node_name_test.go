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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestNodeName(t *testing.T) {
	tests := []struct {
		pod                 *v1.Pod
		node                *v1.Node
		name                string
		wantStatus          *framework.Status
		wantPreFilterStatus *framework.Status
		wantPreFilterResult *framework.PreFilterResult
	}{
		{
			pod:                 &v1.Pod{},
			node:                &v1.Node{},
			name:                "no host specified",
			wantPreFilterStatus: framework.NewStatus(framework.Skip),
		},
		{
			pod:                 st.MakePod().Node("foo").Obj(),
			node:                st.MakeNode().Name("foo").Obj(),
			name:                "host matches",
			wantPreFilterStatus: framework.NewStatus(framework.Skip),
			wantPreFilterResult: &framework.PreFilterResult{NodeNames: sets.NewString("foo")},
		},
		{
			pod:                 st.MakePod().Node("bar").Obj(),
			node:                st.MakeNode().Name("foo").Obj(),
			name:                "host doesn't match",
			wantStatus:          framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReason),
			wantPreFilterStatus: framework.NewStatus(framework.Skip),
			wantPreFilterResult: &framework.PreFilterResult{NodeNames: sets.NewString("bar")},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(test.node)

			p, _ := New(nil, nil)
			gotPreFilterResult, gotStatus := p.(framework.PreFilterPlugin).PreFilter(context.Background(), nil, test.pod)
			if diff := cmp.Diff(test.wantPreFilterStatus, gotStatus); diff != "" {
				t.Errorf("unexpected PreFilter Status (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(test.wantPreFilterResult, gotPreFilterResult); diff != "" {
				t.Errorf("unexpected PreFilterResult (-want,+got):\n%s", diff)
			}
			gotStatus = p.(framework.FilterPlugin).Filter(context.Background(), nil, test.pod, nodeInfo)
			if diff := cmp.Diff(test.wantPreFilterStatus, gotStatus); diff != "" {
				t.Errorf("unexpected Filter Status (-want,+got):\n%s", diff)
			}
		})
	}
}
