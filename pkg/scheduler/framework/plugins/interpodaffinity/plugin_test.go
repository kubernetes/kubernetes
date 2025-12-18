/*
Copyright 2024 The Kubernetes Authors.

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

package interpodaffinity

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	schedruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_isSchedulableAfterPodChange(t *testing.T) {
	tests := []struct {
		name           string
		pod            *v1.Pod
		oldPod, newPod *v1.Pod
		expectedHint   fwk.QueueingHint
	}{
		{
			name:         "add a pod which matches the pod affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add an un-scheduled pod",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Label("service", "securityscan").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a pod which doesn't match the pod affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update a pod from non-match to match pod affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "the updating pod matches target pod's affinity both before and after label changes",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "value2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update an un-scheduled pod",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().UID("other").Label("aaa", "a").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update a pod from match to non-match the pod affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "update a pod from match to non-match pod's affinity - multiple terms case",
			pod: st.MakePod().UID("p").Name("p").PodAffinityExists("aaa", "hostname", st.PodAffinityWithRequiredReq).
				PodAffinityExists("bbb", "hostname", st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "update a pod from non-match to match pod's affinity - multiple terms case",
			pod: st.MakePod().UID("p").Name("p").PodAffinityExists("aaa", "hostname", st.PodAffinityWithRequiredReq).
				PodAffinityExists("bbb", "hostname", st.PodAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "").Label("bbb", "").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "updated pod is the target pod",
			pod:          st.MakePod().UID("p").Name("p").Label("foo", "baz").Obj(),
			newPod:       st.MakePod().UID("p").Name("p").Label("foo", "baz").Obj(),
			oldPod:       st.MakePod().UID("p").Name("p").Label("foo", "bar").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "modify pod label to change it from satisfying pod anti-affinity to not satisfying anti-affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "aaaa").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "modify pod label to change it from not satisfying pod anti-affinity to satisfying anti-affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "bbb").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a pod which doesn't match pod's anti-affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a pod which matches pod's anti-affinity",
			pod:          st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "delete a pod with anti-affinity that matches pending pod",
			pod:          st.MakePod().Name("p").Label("service", "securityscan").Obj(),
			oldPod:       st.MakePod().Node("fake-node").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "delete a pod with anti-affinity that doesn't match pending pod",
			pod:          st.MakePod().Name("p").Label("service", "foo").Obj(),
			oldPod:       st.MakePod().Node("fake-node").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a pod which doesn't match pending pod's anti-affinity and has anti-affinity that doesn't match pending pod",
			pod:          st.MakePod().Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Label("service", "foo").Obj(),
			oldPod:       st.MakePod().Node("fake-node").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Label("service", "foo").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPluginWithInformers(ctx, t, schedruntime.FactoryAdapter(feature.Features{}, New), &config.InterPodAffinityArgs{}, snapshot, namespaces)
			p := pl.(*InterPodAffinity)
			actualHint, err := p.isSchedulableAfterPodChange(logger, tc.pod, tc.oldPod, tc.newPod)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("expected QueuingHint doesn't match (-want,+got): \n %s", diff)
			}
		})
	}
}

func Test_isSchedulableAfterNodeChange(t *testing.T) {
	testcases := []struct {
		name             string
		pod              *v1.Pod
		oldNode, newNode *v1.Node
		expectedHint     fwk.QueueingHint
	}{
		// affinity
		{
			name:         "add a new node with matched pod affinity topologyKey",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "zone", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a new node without matched topologyKey",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update node label but not topologyKey",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "zone", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("aaa", "a").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("aaa", "b").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update node label that isn't related to the pod affinity",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "zone", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("unrelated-label", "unrelated").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update node with different affinity topologyKey value",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "zone", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone2").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update node to have the affinity topology label",
			pod:          st.MakePod().Name("p").PodAffinityIn("service", "zone", []string{"securityscan", "value2"}, st.PodAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("aaa", "a").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.Queue,
		},
		// anti-affinity
		{
			name:         "add a new node with matched pod anti-affinity topologyKey",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a new node without matched pod anti-affinity topologyKey",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newNode:      st.MakeNode().Name("node-a").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update node label that isn't related to the pod anti-affinity",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Label("unrelated-label", "unrelated").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update node with different anti-affinity topologyKey value",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone2").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update node to have the anti-affinity topology label",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("aaa", "a").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update node label not to have anti-affinity topology label",
			pod:          st.MakePod().Name("p").PodAntiAffinity("zone", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldNode:      st.MakeNode().Name("node-a").Label("zone", "zone1").Obj(),
			newNode:      st.MakeNode().Name("node-a").Label("aaa", "a").Obj(),
			expectedHint: fwk.Queue,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPluginWithInformers(ctx, t, schedruntime.FactoryAdapter(feature.Features{}, New), &config.InterPodAffinityArgs{}, snapshot, namespaces)
			p := pl.(*InterPodAffinity)
			actualHint, err := p.isSchedulableAfterNodeChange(logger, tc.pod, tc.oldNode, tc.newNode)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("expected QueuingHint doesn't match (-want,+got): \n %s", diff)
			}
		})
	}
}

func TestPodAffinitySignature(t *testing.T) {
	tests := []struct {
		name              string
		pod               *v1.Pod
		expectedSignature []fwk.SignFragment
		schedulable       bool
		config            config.InterPodAffinityArgs
	}{
		{
			name: "no affinity, default settings",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: v1.PodSpec{},
			},
			expectedSignature: []fwk.SignFragment{
				{
					Key:   fwk.LabelsSignerName,
					Value: map[string]string{"foo": "bar"},
				},
			},
			schedulable: true,
		},
		{
			name: "no affinity, ignore setting set",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: v1.PodSpec{},
			},
			schedulable: true,
			config: config.InterPodAffinityArgs{
				IgnorePreferredTermsOfExistingPods: true,
			},
		},
		{
			name: "affinity set",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{},
							},
						},
					},
				},
			},
			schedulable: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			snapshot := cache.NewSnapshot(nil, nil)
			pl := plugintesting.SetupPluginWithInformers(ctx, t, schedruntime.FactoryAdapter(feature.Features{}, New), &test.config, snapshot, namespaces)
			p := pl.(*InterPodAffinity)
			signature, status := p.SignPod(ctx, test.pod)

			if !status.IsSuccess() && test.schedulable {
				t.Fatalf("Expected success, got %v", status)
			}

			if diff := cmp.Diff(test.expectedSignature, signature); diff != "" {
				t.Fatalf("Diff %s", diff)
			}
		})
	}
}
