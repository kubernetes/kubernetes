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
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func Test_isSchedulableAfterNodeChange(t *testing.T) {
	testcases := []struct {
		name             string
		pod              *v1.Pod
		oldNode, newNode *v1.Node
		expectedHint     fwk.QueueingHint
		expectedErr      bool
	}{
		{
			name: "node updates label which matches topologyKey",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone2").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "node that doesn't match topologySpreadConstraints updates non-related label",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("foo", "bar1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("foo", "bar2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "node that match topologySpreadConstraints adds non-related label",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Label("foo", "bar").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "create node with non-related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("foo", "bar").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "create node with related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "delete node with non-related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("foo", "bar").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "delete node with related labels",
			pod: st.MakePod().Name("p").SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "add node with related labels that only match one of topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "add node with related labels that match all topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "update node with related labels that only match one of topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "update node with related labels that match all topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node2").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "update node with different taints that match all topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Taints([]v1.Taint{{Key: "aaa", Value: "bbb", Effect: v1.TaintEffectNoSchedule}}).Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node1").Taints([]v1.Taint{{Key: "ccc", Value: "bbb", Effect: v1.TaintEffectNoSchedule}}).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "update node with different taints that only match one of topologySpreadConstraints",
			pod: st.MakePod().Name("p").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("node", "node1").Taints([]v1.Taint{{Key: "aaa", Value: "bbb", Effect: v1.TaintEffectNoSchedule}}).Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("node", "node1").Taints([]v1.Taint{{Key: "ccc", Value: "bbb", Effect: v1.TaintEffectNoSchedule}}).Obj(),
			expectedHint: fwk.QueueSkip,
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
		name                                         string
		pod                                          *v1.Pod
		oldPod, newPod                               *v1.Pod
		expectedHint                                 fwk.QueueingHint
		expectedErr                                  bool
		enableNodeInclusionPolicyInPodTopologySpread bool
	}{
		{
			name: "add pod with labels match topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "add un-scheduled pod",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().UID("p2").Label("foo", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "update un-scheduled pod",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().UID("p2").Label("foo", "").Obj(),
			oldPod:       st.MakePod().UID("p2").Label("bar", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "delete un-scheduled pod",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Label("foo", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "add pod with different namespace",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Namespace("fake-namespace").Label("foo", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "add pod with labels don't match topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("bar", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "delete pod with labels that match topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "delete pod with labels that don't match topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("bar", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "update pod's non-related label",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar1").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "add pod's label that matches topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "delete pod label that matches topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "change pod's label that matches topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "foo1").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "foo2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "change pod's label that doesn't match topologySpreadConstraints selector",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar1").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "add pod's label that matches topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "change pod's label that doesn't match topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("baz", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "change pod's label that match topologySpreadConstraints selector with multi topologySpreadConstraints",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, nil, nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "").Obj(),
			newPod:       st.MakePod().UID("p2").Node("fake-node").Label("foo", "").Label("bar", "bar2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "the unschedulable Pod has topologySpreadConstraint with NodeTaintsPolicy:Honor and has got a new toleration",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, ptr.To(v1.NodeInclusionPolicyHonor), nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p").Name("p").Label("foo", "").Obj(),
			newPod:       st.MakePod().UID("p").Name("p").Label("foo", "").Toleration(v1.TaintNodeUnschedulable).Obj(),
			expectedHint: fwk.Queue,
			enableNodeInclusionPolicyInPodTopologySpread: true,
		},
		{
			name: "the unschedulable Pod has topologySpreadConstraint without NodeTaintsPolicy:Honor and has got a new toleration",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, ptr.To(v1.NodeInclusionPolicyIgnore), nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p").Name("p").Label("foo", "").Obj(),
			newPod:       st.MakePod().UID("p").Name("p").Label("foo", "").Toleration(v1.TaintNodeUnschedulable).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "the unschedulable Pod has topologySpreadConstraint and has got a new label matching the selector of the constraint",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, ptr.To(v1.NodeInclusionPolicyIgnore), nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p").Name("p").Obj(),
			newPod:       st.MakePod().UID("p").Name("p").Label("foo", "").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "the unschedulable Pod has topologySpreadConstraint and has got a new unrelated label",
			pod: st.MakePod().UID("p").Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector, nil, nil, ptr.To(v1.NodeInclusionPolicyIgnore), nil).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector, nil, nil, nil, nil).
				Obj(),
			oldPod:       st.MakePod().UID("p").Name("p").Obj(),
			newPod:       st.MakePod().UID("p").Name("p").Label("unrelated", "").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPlugin(ctx, t, topologySpreadFunc, &config.PodTopologySpreadArgs{DefaultingType: config.ListDefaulting}, snapshot)
			p := pl.(*PodTopologySpread)
			p.enableNodeInclusionPolicyInPodTopologySpread = tc.enableNodeInclusionPolicyInPodTopologySpread

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
