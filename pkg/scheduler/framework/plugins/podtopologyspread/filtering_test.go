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

package podtopologyspread

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/pointer"
)

var cmpOpts = []cmp.Option{
	cmp.Comparer(func(s1 labels.Selector, s2 labels.Selector) bool {
		return reflect.DeepEqual(s1, s2)
	}),
	cmp.Comparer(func(p1, p2 criticalPaths) bool {
		p1.sort()
		p2.sort()
		return p1[0] == p2[0] && p1[1] == p2[1]
	}),
}

func (p *criticalPaths) sort() {
	if p[0].MatchNum == p[1].MatchNum && p[0].TopologyValue > p[1].TopologyValue {
		// Swap TopologyValue to make them sorted alphabetically.
		p[0].TopologyValue, p[1].TopologyValue = p[1].TopologyValue, p[0].TopologyValue
	}
}

func TestPreFilterState(t *testing.T) {
	fooSelector := st.MakeLabelSelector().Exists("foo").Obj()
	barSelector := st.MakeLabelSelector().Exists("bar").Obj()
	tests := []struct {
		name               string
		pod                *v1.Pod
		nodes              []*v1.Node
		existingPods       []*v1.Pod
		objs               []runtime.Object
		defaultConstraints []v1.TopologySpreadConstraint
		want               *preFilterState
	}{
		{
			name: "clean cluster with one spreadConstraint",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				5, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Label("foo", "bar").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     5,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Label("foo", "bar").Obj()),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 0}, {"zone2", 0}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(0),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(0),
				},
			},
		},
		{
			name: "normal case with one spreadConstraint",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, fooSelector,
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 2}, {"zone1", 3}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "normal case with one spreadConstraint, on a 3-zone cluster",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
				st.MakeNode().Name("node-o").Label("zone", "zone3").Label("node", "node-o").Obj(),
				st.MakeNode().Name("node-p").Label("zone", "zone3").Label("node", "node-p").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone3", 0}, {"zone2", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(2),
					{key: "zone", value: "zone3"}: pointer.Int32Ptr(0),
				},
			},
		},
		{
			name: "namespace mismatch doesn't count",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, fooSelector,
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Namespace("ns1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Namespace("ns2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 1}, {"zone1", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(2),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "normal case with two spreadConstraints",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 3}, {"zone2", 4}},
					"node": {{"node-x", 0}, {"node-b", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(4),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(2),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-y"}: pointer.Int32Ptr(4),
				},
			},
		},
		{
			name: "soft spreadConstraints should be bypassed",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.ScheduleAnyway, fooSelector).
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector).
				SpreadConstraint(1, "node", v1.ScheduleAnyway, fooSelector).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 3}, {"zone2", 4}},
					"node": {{"node-b", 1}, {"node-a", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(4),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(2),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-y"}: pointer.Int32Ptr(4),
				},
			},
		},
		{
			name: "different labelSelectors - simple version",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b").Node("node-b").Label("bar", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, barSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 0}, {"zone1", 1}},
					"node": {{"node-a", 0}, {"node-y", 0}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(0),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-y"}: pointer.Int32Ptr(0),
				},
			},
		},
		{
			name: "different labelSelectors - complex pods",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector).
				SpreadConstraint(1, "node", v1.DoNotSchedule, barSelector).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, barSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 3}, {"zone2", 4}},
					"node": {{"node-b", 0}, {"node-a", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(4),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-y"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "two spreadConstraints, and with podAffinity",
			pod: st.MakePod().Name("p").Label("foo", "").
				NodeAffinityNotIn("node", []string{"node-x"}). // exclude node-x
				SpreadConstraint(1, "zone", v1.DoNotSchedule, fooSelector).
				SpreadConstraint(1, "node", v1.DoNotSchedule, fooSelector).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, fooSelector),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 3}, {"zone2", 4}},
					"node": {{"node-b", 1}, {"node-a", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(4),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(2),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-y"}: pointer.Int32Ptr(4),
				},
			},
		},
		{
			name: "default constraints and a service",
			pod:  st.MakePod().Name("p").Label("foo", "bar").Label("baz", "kar").Obj(),
			defaultConstraints: []v1.TopologySpreadConstraint{
				{MaxSkew: 3, TopologyKey: "node", WhenUnsatisfiable: v1.DoNotSchedule},
				{MaxSkew: 2, TopologyKey: "node", WhenUnsatisfiable: v1.ScheduleAnyway},
				{MaxSkew: 5, TopologyKey: "rack", WhenUnsatisfiable: v1.DoNotSchedule},
			},
			objs: []runtime.Object{
				&v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"foo": "bar"}}},
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     3,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Label("foo", "bar").Obj()),
					},
					{
						MaxSkew:     5,
						TopologyKey: "rack",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Label("foo", "bar").Obj()),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"node": newCriticalPaths(),
					"rack": newCriticalPaths(),
				},
				TpPairToMatchNum: make(map[topologyPair]*int32),
			},
		},
		{
			name: "default constraints and a service that doesn't match",
			pod:  st.MakePod().Name("p").Label("foo", "bar").Obj(),
			defaultConstraints: []v1.TopologySpreadConstraint{
				{MaxSkew: 3, TopologyKey: "node", WhenUnsatisfiable: v1.DoNotSchedule},
			},
			objs: []runtime.Object{
				&v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "kep"}}},
			},
			want: &preFilterState{},
		},
		{
			name: "default constraints and a service, but pod has constraints",
			pod: st.MakePod().Name("p").Label("foo", "bar").Label("baz", "tar").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Label("baz", "tar").Obj()).
				SpreadConstraint(2, "planet", v1.ScheduleAnyway, st.MakeLabelSelector().Label("fot", "rok").Obj()).Obj(),
			defaultConstraints: []v1.TopologySpreadConstraint{
				{MaxSkew: 2, TopologyKey: "node", WhenUnsatisfiable: v1.DoNotSchedule},
			},
			objs: []runtime.Object{
				&v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"foo": "bar"}}},
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					{
						MaxSkew:     1,
						TopologyKey: "zone",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Label("baz", "tar").Obj()),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": newCriticalPaths(),
				},
				TpPairToMatchNum: make(map[topologyPair]*int32),
			},
		},
		{
			name: "default soft constraints and a service",
			pod:  st.MakePod().Name("p").Label("foo", "bar").Obj(),
			defaultConstraints: []v1.TopologySpreadConstraint{
				{MaxSkew: 2, TopologyKey: "node", WhenUnsatisfiable: v1.ScheduleAnyway},
			},
			objs: []runtime.Object{
				&v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"foo": "bar"}}},
			},
			want: &preFilterState{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			informerFactory := informers.NewSharedInformerFactory(fake.NewSimpleClientset(tt.objs...), 0)
			pl := PodTopologySpread{
				sharedLister:       cache.NewSnapshot(tt.existingPods, tt.nodes),
				defaultConstraints: tt.defaultConstraints,
			}
			pl.setListers(informerFactory)
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			cs := framework.NewCycleState()
			if s := pl.PreFilter(ctx, cs, tt.pod); !s.IsSuccess() {
				t.Fatal(s.AsError())
			}
			got, err := getPreFilterState(cs)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, got, cmpOpts...); diff != "" {
				t.Errorf("PodTopologySpread#PreFilter() returned diff (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPreFilterStateAddPod(t *testing.T) {
	nodeConstraint := topologySpreadConstraint{
		MaxSkew:     1,
		TopologyKey: "node",
		Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Exists("foo").Obj()),
	}
	zoneConstraint := nodeConstraint
	zoneConstraint.TopologyKey = "zone"
	tests := []struct {
		name         string
		preemptor    *v1.Pod
		addedPod     *v1.Pod
		existingPods []*v1.Pod
		nodeIdx      int // denotes which node 'addedPod' belongs to
		nodes        []*v1.Node
		want         *preFilterState
	}{
		{
			name: "node a and b both impact current min match",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod:     st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: nil, // it's an empty cluster
			nodeIdx:      0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"node": {{"node-b", 0}, {"node-a", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(0),
				},
			},
		},
		{
			name: "only node a impacts current min match",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"node": {{"node-a", 1}, {"node-b", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "add a pod in a different namespace doesn't change topologyKeyToMinPodsMap",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Namespace("ns1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"node": {{"node-a", 0}, {"node-b", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "node", value: "node-a"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "add pod on non-critical node won't trigger re-calculation",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 1,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"node": {{"node-a", 0}, {"node-b", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "node", value: "node-a"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "node a and x both impact topologyKeyToMinPodsMap on zone and node",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod:     st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: nil, // it's an empty cluster
			nodeIdx:      0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint, nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 0}, {"zone1", 1}},
					"node": {{"node-x", 0}, {"node-a", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(0),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(0),
				},
			},
		},
		{
			name: "only node a impacts topologyKeyToMinPodsMap on zone and node",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint, nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 1}, {"zone2", 1}},
					"node": {{"node-a", 1}, {"node-x", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(1),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "node a impacts topologyKeyToMinPodsMap on node, node x impacts topologyKeyToMinPodsMap on zone",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint, nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 1}, {"zone1", 3}},
					"node": {{"node-a", 1}, {"node-x", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(1),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(2),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "Constraints hold different labelSelectors, node a impacts topologyKeyToMinPodsMap on zone",
			preemptor: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("bar", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					zoneConstraint,
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Exists("bar").Obj()),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 1}, {"zone1", 2}},
					"node": {{"node-a", 0}, {"node-b", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(2),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(1),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(0),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "Constraints hold different labelSelectors, node a impacts topologyKeyToMinPodsMap on both zone and node",
			preemptor: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Label("bar", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("bar", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("bar", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{
					zoneConstraint,
					{
						MaxSkew:     1,
						TopologyKey: "node",
						Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Exists("bar").Obj()),
					},
				},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 1}, {"zone2", 1}},
					"node": {{"node-a", 1}, {"node-b", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(1),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(2),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)
			pl := PodTopologySpread{
				sharedLister: snapshot,
			}
			cs := framework.NewCycleState()
			ctx := context.Background()
			if s := pl.PreFilter(ctx, cs, tt.preemptor); !s.IsSuccess() {
				t.Fatal(s.AsError())
			}
			nodeInfo, err := snapshot.Get(tt.nodes[tt.nodeIdx].Name)
			if err != nil {
				t.Fatal(err)
			}
			if s := pl.AddPod(ctx, cs, tt.preemptor, tt.addedPod, nodeInfo); !s.IsSuccess() {
				t.Fatal(s.AsError())
			}
			state, err := getPreFilterState(cs)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(state, tt.want, cmpOpts...); diff != "" {
				t.Errorf("PodTopologySpread.AddPod() returned diff (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPreFilterStateRemovePod(t *testing.T) {
	nodeConstraint := topologySpreadConstraint{
		MaxSkew:     1,
		TopologyKey: "node",
		Selector:    mustConvertLabelSelectorAsSelector(t, st.MakeLabelSelector().Exists("foo").Obj()),
	}
	zoneConstraint := nodeConstraint
	zoneConstraint.TopologyKey = "zone"
	tests := []struct {
		name          string
		preemptor     *v1.Pod // preemptor pod
		nodes         []*v1.Node
		existingPods  []*v1.Pod
		deletedPodIdx int     // need to reuse *Pod of existingPods[i]
		deletedPod    *v1.Pod // this field is used only when deletedPodIdx is -1
		nodeIdx       int     // denotes which node "deletedPod" belongs to
		want          *preFilterState
	}{
		{
			// A high priority pod may not be scheduled due to node taints or resource shortage.
			// So preemption is triggered.
			name: "one spreadConstraint on zone, topologyKeyToMinPodsMap unchanged",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a1"
			nodeIdx:       0, // node-a
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 1}, {"zone2", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(1),
				},
			},
		},
		{
			name: "one spreadConstraint on node, topologyKeyToMinPodsMap changed",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a1"
			nodeIdx:       0, // node-a
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 1}, {"zone2", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(1),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "delete an irrelevant pod won't help",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a0").Node("node-a").Label("bar", "").Obj(),
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a0"
			nodeIdx:       0, // node-a
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 2}, {"zone2", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(2),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "delete a non-existing pod won't help",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: -1,
			deletedPod:    st.MakePod().Name("p-a0").Node("node-a").Label("bar", "").Obj(),
			nodeIdx:       0, // node-a
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone1", 2}, {"zone2", 2}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}: pointer.Int32Ptr(2),
					{key: "zone", value: "zone2"}: pointer.Int32Ptr(2),
				},
			},
		},
		{
			name: "two spreadConstraints",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("foo", "").Obj(),
			},
			deletedPodIdx: 3, // remove pod "p-x1"
			nodeIdx:       2, // node-x
			want: &preFilterState{
				Constraints: []topologySpreadConstraint{zoneConstraint, nodeConstraint},
				TpKeyToCriticalPaths: map[string]*criticalPaths{
					"zone": {{"zone2", 1}, {"zone1", 3}},
					"node": {{"node-b", 1}, {"node-x", 1}},
				},
				TpPairToMatchNum: map[topologyPair]*int32{
					{key: "zone", value: "zone1"}:  pointer.Int32Ptr(3),
					{key: "zone", value: "zone2"}:  pointer.Int32Ptr(1),
					{key: "node", value: "node-a"}: pointer.Int32Ptr(2),
					{key: "node", value: "node-b"}: pointer.Int32Ptr(1),
					{key: "node", value: "node-x"}: pointer.Int32Ptr(1),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)
			pl := PodTopologySpread{
				sharedLister: snapshot,
			}
			cs := framework.NewCycleState()
			ctx := context.Background()
			s := pl.PreFilter(ctx, cs, tt.preemptor)
			if !s.IsSuccess() {
				t.Fatal(s.AsError())
			}

			deletedPod := tt.deletedPod
			if tt.deletedPodIdx < len(tt.existingPods) && tt.deletedPodIdx >= 0 {
				deletedPod = tt.existingPods[tt.deletedPodIdx]
			}

			nodeInfo, err := snapshot.Get(tt.nodes[tt.nodeIdx].Name)
			if err != nil {
				t.Fatal(err)
			}
			if s := pl.RemovePod(ctx, cs, tt.preemptor, deletedPod, nodeInfo); !s.IsSuccess() {
				t.Fatal(s.AsError())
			}

			state, err := getPreFilterState(cs)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(state, tt.want, cmpOpts...); diff != "" {
				t.Errorf("PodTopologySpread.RemovePod() returned diff (-want,+got):\n%s", diff)
			}
		})
	}
}

func BenchmarkFilter(b *testing.B) {
	tests := []struct {
		name             string
		pod              *v1.Pod
		existingPodsNum  int
		allNodesNum      int
		filteredNodesNum int
	}{
		{
			name: "1000nodes/single-constraint-zone",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, v1.LabelTopologyZone, v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
		{
			name: "1000nodes/single-constraint-node",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, v1.LabelHostname, v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
		{
			name: "1000nodes/two-Constraints-zone-node",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, v1.LabelTopologyZone, v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, v1.LabelHostname, v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
	}
	for _, tt := range tests {
		var state *framework.CycleState
		b.Run(tt.name, func(b *testing.B) {
			existingPods, allNodes, _ := st.MakeNodesAndPodsForEvenPodsSpread(tt.pod.Labels, tt.existingPodsNum, tt.allNodesNum, tt.filteredNodesNum)
			pl := PodTopologySpread{
				sharedLister: cache.NewSnapshot(existingPods, allNodes),
			}
			ctx := context.Background()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				state = framework.NewCycleState()
				s := pl.PreFilter(ctx, state, tt.pod)
				if !s.IsSuccess() {
					b.Fatal(s.AsError())
				}
				filterNode := func(i int) {
					n, _ := pl.sharedLister.NodeInfos().Get(allNodes[i].Name)
					pl.Filter(ctx, state, tt.pod, n)
				}
				parallelize.Until(ctx, len(allNodes), filterNode)
			}
		})
		b.Run(tt.name+"/Clone", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				state.Clone()
			}
		})
	}
}

func mustConvertLabelSelectorAsSelector(t *testing.T, ls *metav1.LabelSelector) labels.Selector {
	t.Helper()
	s, err := metav1.LabelSelectorAsSelector(ls)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestSingleConstraint(t *testing.T) {
	tests := []struct {
		name           string
		pod            *v1.Pod
		nodes          []*v1.Node
		existingPods   []*v1.Pod
		wantStatusCode map[string]framework.Code
	}{
		{
			name: "no existing pods",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success,
				"node-x": framework.Success,
				"node-y": framework.Success,
			},
		},
		{
			name: "no existing pods, incoming pod doesn't match itself",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success,
				"node-x": framework.Success,
				"node-y": framework.Success,
			},
		},
		{
			name: "existing pods in a different namespace do not count",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Namespace("ns1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Namespace("ns2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			name: "pods spread across zones as 3/3, all nodes fit",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success,
				"node-x": framework.Success,
				"node-y": framework.Success,
			},
		},
		{
			// TODO(Huang-Wei): maybe document this to remind users that typos on node labels
			// can cause unexpected behavior
			name: "pods spread across zones as 1/2 due to absence of label 'zone' on node-b",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zon", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.UnschedulableAndUnresolvable,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			name: "pod cannot be scheduled as all nodes don't have label 'rack'",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "rack", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.UnschedulableAndUnresolvable,
				"node-x": framework.UnschedulableAndUnresolvable,
			},
		},
		{
			name: "pods spread across nodes as 2/1/0/3, only node-x fits",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Unschedulable,
				"node-x": framework.Success,
				"node-y": framework.Unschedulable,
			},
		},
		{
			name: "pods spread across nodes as 2/1/0/3, maxSkew is 2, node-b and node-x fit",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				2, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Success,
				"node-x": framework.Success,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// not a desired case, but it can happen
			// TODO(Huang-Wei): document this "pod-not-match-itself" case
			// in this case, placement of the new pod doesn't change pod distribution of the cluster
			// as the incoming pod doesn't have label "foo"
			name: "pods spread across nodes as 2/1/0/3, but pod doesn't match itself",
			pod: st.MakePod().Name("p").Label("bar", "").SpreadConstraint(
				1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Success,
				"node-x": framework.Success,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// only node-a and node-y are considered, so pods spread as 2/~1~/~0~/3
			// ps: '~num~' is a markdown symbol to denote a crossline through 'num'
			// but in this unit test, we don't run NodeAffinity Predicate, so node-b and node-x are
			// still expected to be fits;
			// the fact that node-a fits can prove the underlying logic works
			name: "incoming pod has nodeAffinity, pods spread as 2/~1~/~0~/3, hence node-a fits",
			pod: st.MakePod().Name("p").Label("foo", "").
				NodeAffinityIn("node", []string{"node-a", "node-y"}).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success, // in real case, it's false
				"node-x": framework.Success, // in real case, it's false
				"node-y": framework.Unschedulable,
			},
		},
		{
			name: "terminating Pods should be excluded",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a").Node("node-a").Label("foo", "").Terminating().Obj(),
				st.MakePod().Name("p-b").Node("node-b").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Unschedulable,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)
			p := &PodTopologySpread{sharedLister: snapshot}
			state := framework.NewCycleState()
			preFilterStatus := p.PreFilter(context.Background(), state, tt.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("preFilter failed with status: %v", preFilterStatus)
			}

			for _, node := range tt.nodes {
				nodeInfo, _ := snapshot.NodeInfos().Get(node.Name)
				status := p.Filter(context.Background(), state, tt.pod, nodeInfo)
				if len(tt.wantStatusCode) != 0 && status.Code() != tt.wantStatusCode[node.Name] {
					t.Errorf("[%s]: expected status code %v got %v", node.Name, tt.wantStatusCode[node.Name], status.Code())
				}
			}
		})
	}
}

func TestMultipleConstraints(t *testing.T) {
	tests := []struct {
		name           string
		pod            *v1.Pod
		nodes          []*v1.Node
		existingPods   []*v1.Pod
		wantStatusCode map[string]framework.Code
	}{
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on any zone (hence any node)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-x
			// intersection of (1) and (2) returns node-x
			name: "two Constraints on zone and node, spreads = [3/3, 2/1/0/3]",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Unschedulable,
				"node-x": framework.Success,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on zone1 (node-a or node-b)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-x
			// intersection of (1) and (2) returns no node
			name: "two Constraints on zone and node, spreads = [3/4, 2/1/0/4]",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Unschedulable,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on zone2 (node-x or node-y)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-a, node-b or node-x
			// intersection of (1) and (2) returns node-x
			name: "Constraints hold different labelSelectors, spreads = [1/0, 1/0/0/1]",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("bar", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Unschedulable,
				"node-x": framework.Success,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on zone2 (node-x or node-y)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-a or node-b
			// intersection of (1) and (2) returns no node
			name: "Constraints hold different labelSelectors, spreads = [1/0, 0/0/1/1]",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("bar", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("bar", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Unschedulable,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on zone1 (node-a or node-b)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-b or node-x
			// intersection of (1) and (2) returns node-b
			name: "Constraints hold different labelSelectors, spreads = [2/3, 1/0/0/1]",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Success,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. pod doesn't match itself on "zone" constraint, so it can be put onto any zone
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-a or node-b
			// intersection of (1) and (2) returns node-a and node-b
			name: "Constraints hold different labelSelectors but pod doesn't match itself on 'zone' constraint",
			pod: st.MakePod().Name("p").Label("bar", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("bar", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("bar", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Success,
				"node-b": framework.Success,
				"node-x": framework.Unschedulable,
				"node-y": framework.Unschedulable,
			},
		},
		{
			// 1. to fulfil "zone" constraint, incoming pod can be placed on any zone (hence any node)
			// 2. to fulfil "node" constraint, incoming pod can be placed on node-b (node-x doesn't have the required label)
			// intersection of (1) and (2) returns node-b
			name: "two Constraints on zone and node, absence of label 'node' on node-x, spreads = [1/1, 1/0/0/1]",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			wantStatusCode: map[string]framework.Code{
				"node-a": framework.Unschedulable,
				"node-b": framework.Success,
				"node-x": framework.UnschedulableAndUnresolvable,
				"node-y": framework.Unschedulable,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)
			p := &PodTopologySpread{sharedLister: snapshot}
			state := framework.NewCycleState()
			preFilterStatus := p.PreFilter(context.Background(), state, tt.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("preFilter failed with status: %v", preFilterStatus)
			}

			for _, node := range tt.nodes {
				nodeInfo, _ := snapshot.NodeInfos().Get(node.Name)
				status := p.Filter(context.Background(), state, tt.pod, nodeInfo)
				if len(tt.wantStatusCode) != 0 && status.Code() != tt.wantStatusCode[node.Name] {
					t.Errorf("[%s]: expected error code %v got %v", node.Name, tt.wantStatusCode[node.Name], status.Code())
				}
			}
		})
	}
}

func TestPreFilterDisabled(t *testing.T) {
	pod := &v1.Pod{}
	nodeInfo := framework.NewNodeInfo()
	node := v1.Node{}
	nodeInfo.SetNode(&node)
	p := &PodTopologySpread{}
	cycleState := framework.NewCycleState()
	gotStatus := p.Filter(context.Background(), cycleState, pod, nodeInfo)
	wantStatus := framework.AsStatus(fmt.Errorf(`reading "PreFilterPodTopologySpread" from cycleState: not found`))
	if !reflect.DeepEqual(gotStatus, wantStatus) {
		t.Errorf("status does not match: %v, want: %v", gotStatus, wantStatus)
	}
}
