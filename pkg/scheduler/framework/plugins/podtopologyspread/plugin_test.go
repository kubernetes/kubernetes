/*
Copyright 2023 The Kubernetes Authors.

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

package podtopologyspread

import (
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_isSchedulableAfterNodeChange(t *testing.T) {
	testcases := []struct {
		name             string
		pod              *v1.Pod
		oldNode, newNode *v1.Node
		expectedHint     framework.QueueingHint
		expectedErr      bool
	}{
		{
			name: "node updates label which matches topologyKey",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone2").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "node that doesn't match topologySpreadConstraints updates non-related label",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("foo", "bar1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("foo", "bar2").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "node that match topologySpreadConstraints adds non-related label",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Label("foo", "bar").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "create node with non-related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("foo", "bar").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "create node with related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "delete node with non-related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("foo", "bar").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "delete node with related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "add node with related labels that only match one of topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "add node with related labels that match all topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "update node with related labels that only match one of topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "update node with related labels that match all topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node2").Obj(),
			expectedHint: framework.Queue,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPlugin(ctx, t, topologySpreadFunc, &config.PodTopologySpreadArgs{DefaultingType: config.ListDefaulting}, snapshot)
			p := pl.(*PodTopologySpread)
			actualHint, err := p.isSchedulableAfterNodeChange(logger, tc.pod, tc.oldNode, tc.newNode)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}

func Test_isSchedulableAfterPodChange(t *testing.T) {
	testcases := []struct {
		name           string
		pod            *v1.Pod
		oldPod, newPod *v1.Pod
		expectedHint   framework.QueueingHint
		expectedErr    bool
	}{
		{
			name: "add pod with labels match topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "add un-scheduled pod",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().Label("foo", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "update un-scheduled pod",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().Label("foo", "").Obj(),
			oldPod:       st.MakePod().Label("bar", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "delete un-scheduled pod",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Label("foo", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "add pod with different namespace",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().Node("fake-node").Namespace("fake-namespace").Label("foo", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "add pod with labels don't match topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("bar", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "delete pod with labels that match topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "delete pod with labels that don't match topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("bar", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "update pod's non-related label",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar1").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "add pod's label that matches topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "delete pod label that matches topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().Node("fake-node").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "change pod's label that matches topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "foo1").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "foo2").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "change pod's label that doesn't match topologySpreadConstraints selector",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar1").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "add pod's label that matches topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: framework.Queue,
		},
		{
			name: "change pod's label that doesn't match topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("baz", "").Obj(),
			expectedHint: framework.QueueSkip,
		},
		{
			name: "change pod's label that match topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "").Obj(),
			newPod:       st.MakePod().Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: framework.QueueSkip,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPlugin(ctx, t, topologySpreadFunc, &config.PodTopologySpreadArgs{DefaultingType: config.ListDefaulting}, snapshot)
			p := pl.(*PodTopologySpread)
			actualHint, err := p.isSchedulableAfterPodChange(logger, tc.pod, tc.oldPod, tc.newPod)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}
