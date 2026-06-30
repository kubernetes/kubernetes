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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
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
		gate       bool
		wantResult *fwk.PreFilterResult
		wantStatus *fwk.Status
	}{
		{
			name: "normal pod, prefilter returns nil result",
			pod:  st.MakePod().Node("foo").Obj(),
			gate: true,
		},
		{
			name: "deferred resize pod, prefilter returns result with node name",
			pod: st.MakePod().Node("foo").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gate: true,
			wantResult: &fwk.PreFilterResult{
				NodeNames: sets.New("foo"),
			},
		},
		{
			name: "deferred resize pod, gate disabled, prefilter returns nil",
			pod: st.MakePod().Node("foo").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gate: false,
		},
		{
			name: "deferred resize pod with empty node name (edge case), returns nil",
			pod: st.MakePod().
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingSchedulerPreemption, tt.gate)
			_, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			gotResult, gotStatus := p.(fwk.PreFilterPlugin).PreFilter(ctx, nil, tt.pod, nil)
			if diff := cmp.Diff(tt.wantStatus, gotStatus); diff != "" {
				t.Errorf("status does not match (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantResult, gotResult); diff != "" {
				t.Errorf("result does not match (-want,+got):\n%s", diff)
			}
		})
	}
}
