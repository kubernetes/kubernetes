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
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	schedruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func withAffinityNamespaceSelector(p *v1.Pod, team string) *v1.Pod {
	term := &p.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0]
	term.NamespaceSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"team": team}}
	return p
}

func withAntiAffinityNamespaceSelector(p *v1.Pod, team string) *v1.Pod {
	term := &p.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0]
	term.NamespaceSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"team": team}}
	return p
}

func Test_isSchedulableAfterAssignedPodChange(t *testing.T) {
	tests := []struct {
		name             string
		pod              *v1.Pod
		oldPod, newPod   *v1.Pod
		namespaceObjects []runtime.Object
		expectedHint     fwk.QueueingHint
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
		{
			name:         "delete a nominated pod which doesn't match pod's anti-affinity",
			pod:          st.MakePod().Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldPod:       st.MakePod().NominatedNodeName("fake-node").Label("aaa", "a").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a nominated pod which matches pod's anti-affinity",
			pod:          st.MakePod().Name("p").PodAntiAffinityIn("service", "region", []string{"securityscan", "value2"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldPod:       st.MakePod().NominatedNodeName("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "modify pod label to change it from matching only one of target pod's anti-affinity terms to matching none",
			pod: st.MakePod().UID("p").Name("p").
				PodAntiAffinityIn("service", "region", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).
				PodAntiAffinityIn("app", "region", []string{"web"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "foo").Label("app", "bar").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Label("app", "bar").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "modify pod label but it still matches one of target pod's anti-affinity terms",
			pod: st.MakePod().UID("p").Name("p").
				PodAntiAffinityIn("service", "region", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).
				PodAntiAffinityIn("app", "region", []string{"web"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "foo").Label("app", "web").Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Label("app", "bar").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name: "delete a pod which matches only one of target pod's anti-affinity terms",
			pod: st.MakePod().UID("p").Name("p").
				PodAntiAffinityIn("service", "region", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).
				PodAntiAffinityIn("app", "region", []string{"web"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "delete a pod with anti-affinity that matches pending pod on only one term",
			pod:  st.MakePod().Name("p").Label("service", "securityscan").Obj(),
			oldPod: st.MakePod().Node("fake-node").UID("other").
				PodAntiAffinityIn("service", "region", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).
				PodAntiAffinityIn("app", "region", []string{"web"}, st.PodAntiAffinityWithRequiredReq).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a pod matching the pending pod's affinity with namespaceSelector",
			pod:          withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a pod in a namespace not matching the namespaceSelector of the pending pod's affinity",
			pod:          withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a pod matching the namespaceSelector but not the labelSelector of the pending pod's affinity",
			pod:          withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update a pod to match the pending pod's affinity with namespaceSelector",
			pod:          withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update a pod to not match the pending pod's anti-affinity with namespaceSelector",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update a pod to match the pending pod's anti-affinity with namespaceSelector",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a pod matching the namespaceSelector but not the labelSelector of the pending pod's anti-affinity",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete a pod matching the pending pod's anti-affinity with namespaceSelector",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "delete a pod whose anti-affinity with namespaceSelector matches the pending pod",
			pod:  st.MakePod().Name("p").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			oldPod: withAntiAffinityNamespaceSelector(st.MakePod().Node("fake-node").UID("other").Namespace("subteam1.team1").
				PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			expectedHint: fwk.Queue,
		},
		{
			name:             "namespace cache miss requeues conservatively",
			pod:              withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:           st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("missing-team").Obj(),
			namespaceObjects: []runtime.Object{&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "other-team", Labels: map[string]string{"team": "team2"}}}},
			expectedHint:     fwk.Queue,
		},
		{
			name:         "update affinity does not match a pod in a non-matching namespace",
			pod:          withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team2").Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "update anti-affinity does not match a pod in a non-matching namespace",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team2").Obj(),
			newPod:       st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "delete does not match anti-affinity in a non-matching namespace",
			pod:          withAntiAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			oldPod:       st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "reverse delete does not match anti-affinity in a non-matching namespace",
			pod:          st.MakePod().Name("p").Label("service", "securityscan").Namespace("subteam1.team2").Obj(),
			oldPod:       withAntiAffinityNamespaceSelector(st.MakePod().Node("fake-node").UID("other").Namespace("subteam1.team1").PodAntiAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAntiAffinityWithRequiredReq).Obj(), "team1"),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			snapshot := cache.NewSnapshot(nil, nil)
			namespaceObjects := namespaces
			if tc.namespaceObjects != nil {
				namespaceObjects = tc.namespaceObjects
			}
			pl := plugintesting.SetupPluginWithInformers(ctx, t, schedruntime.FactoryAdapter(feature.Features{}, New), &config.InterPodAffinityArgs{}, snapshot, namespaceObjects)
			p := pl.(*InterPodAffinity)
			actualHint, err := p.isSchedulableAfterAssignedPodChange(logger, tc.pod, tc.oldPod, tc.newPod)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("unexpected QueueingHint (-want, +got):\n%s", diff)
			}
		})
	}
}

type countingNamespaceLister struct {
	getCalls   int
	listCalls  int
	namespaces map[string]*v1.Namespace
	getErr     error
}

func (l *countingNamespaceLister) List(selector labels.Selector) ([]*v1.Namespace, error) {
	l.listCalls++
	return nil, nil
}

func (l *countingNamespaceLister) Get(name string) (*v1.Namespace, error) {
	l.getCalls++
	if l.getErr != nil {
		return nil, l.getErr
	}
	ns, ok := l.namespaces[name]
	if !ok {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "namespaces"}, name)
	}
	return ns, nil
}

func TestTermsNeedNamespaceLabels(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("team1").Label("service", "securityscan").Obj()
	tests := []struct {
		name string
		term fwk.AffinityTerm
		want bool
	}{
		{
			name: "empty namespace selector",
			term: fwk.AffinityTerm{
				NamespaceSelector: labels.Everything(),
				Selector:          labels.Everything(),
			},
			want: false,
		},
		{
			name: "namespace selector and Namespaces both cover the pod",
			term: fwk.AffinityTerm{
				NamespaceSelector: labels.SelectorFromSet(labels.Set{"team": "team1"}),
				Namespaces:        sets.New[string]("team1"),
				Selector:          labels.Everything(),
			},
			want: false,
		},
		{
			name: "namespace labels are needed",
			term: fwk.AffinityTerm{
				NamespaceSelector: labels.SelectorFromSet(labels.Set{"team": "team1"}),
				Selector:          labels.SelectorFromSet(labels.Set{"service": "securityscan"}),
			},
			want: true,
		},
		{
			name: "label selector cannot match",
			term: fwk.AffinityTerm{
				NamespaceSelector: labels.SelectorFromSet(labels.Set{"team": "team1"}),
				Selector:          labels.SelectorFromSet(labels.Set{"service": "other"}),
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := termsNeedNamespaceLabels([]fwk.AffinityTerm{tc.term}, pod); got != tc.want {
				t.Errorf("termsNeedNamespaceLabels() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNamespaceLabelsForNamespace(t *testing.T) {
	wantLabels := labels.Set{"team": "team1"}
	tests := []struct {
		name       string
		needed     bool
		lister     *countingNamespaceLister
		wantLabels labels.Set
		wantFound  bool
		wantErr    bool
	}{
		{
			name:      "does not look up when not needed",
			needed:    false,
			lister:    &countingNamespaceLister{},
			wantFound: true,
		},
		{
			name:   "returns labels when found",
			needed: true,
			lister: &countingNamespaceLister{namespaces: map[string]*v1.Namespace{
				"team1": {ObjectMeta: metav1.ObjectMeta{Name: "team1", Labels: map[string]string{"team": "team1"}}},
			}},
			wantLabels: wantLabels,
			wantFound:  true,
		},
		{
			name:      "reports missing namespace without error",
			needed:    true,
			lister:    &countingNamespaceLister{},
			wantFound: false,
		},
		{
			name:      "returns lister error",
			needed:    true,
			lister:    &countingNamespaceLister{getErr: errors.New("lister error")},
			wantFound: false,
			wantErr:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &InterPodAffinity{nsLister: tc.lister}
			gotLabels, found, err := p.namespaceLabelsForNamespace("team1", tc.needed)
			if (err != nil) != tc.wantErr {
				t.Fatalf("namespaceLabelsForNamespace() error = %v, wantErr %v", err, tc.wantErr)
			}
			if found != tc.wantFound {
				t.Errorf("namespaceLabelsForNamespace() found = %v, want %v", found, tc.wantFound)
			}
			if diff := cmp.Diff(tc.wantLabels, gotLabels); diff != "" {
				t.Errorf("namespaceLabelsForNamespace() labels (-want, +got):\n%s", diff)
			}
		})
	}
}

func Test_isSchedulableAfterAssignedPodChangeNamespaceLookup(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		newPod   *v1.Pod
		lister   *countingNamespaceLister
		wantHint fwk.QueueingHint
		wantGet  int
		wantList int
	}{
		{
			name: "uses Get when namespace selector is needed",
			pod: withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").
				PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod: st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			lister: &countingNamespaceLister{namespaces: map[string]*v1.Namespace{
				"subteam1.team1": {ObjectMeta: metav1.ObjectMeta{Name: "subteam1.team1", Labels: map[string]string{"team": "team1"}}},
			}},
			wantHint: fwk.Queue,
			wantGet:  1,
		},
		{
			name: "does not lookup when Namespaces already matches",
			pod: func() *v1.Pod {
				p := st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj()
				p.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].Namespaces = []string{"subteam1.team1"}
				return p
			}(),
			newPod:   st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("subteam1.team1").Obj(),
			lister:   &countingNamespaceLister{},
			wantHint: fwk.Queue,
		},
		{
			name:     "requeues on namespace cache miss",
			pod:      withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:   st.MakePod().UID("other").Node("fake-node").Label("service", "securityscan").Namespace("missing-team").Obj(),
			lister:   &countingNamespaceLister{},
			wantHint: fwk.Queue,
			wantGet:  1,
		},
		{
			name:     "does not lookup when the label selector cannot match",
			pod:      withAffinityNamespaceSelector(st.MakePod().UID("p").Name("p").PodAffinityIn("service", "hostname", []string{"securityscan"}, st.PodAffinityWithRequiredReq).Obj(), "team1"),
			newPod:   st.MakePod().UID("other").Node("fake-node").Label("aaa", "a").Namespace("subteam1.team1").Obj(),
			lister:   &countingNamespaceLister{},
			wantHint: fwk.QueueSkip,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			p := &InterPodAffinity{nsLister: tc.lister}
			hint, err := p.isSchedulableAfterAssignedPodChange(logger, tc.pod, nil, tc.newPod)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.wantHint, hint); diff != "" {
				t.Errorf("unexpected QueueingHint (-want, +got):\n%s", diff)
			}
			if tc.lister.getCalls != tc.wantGet || tc.lister.listCalls != tc.wantList {
				t.Errorf("namespace lookup calls = Get:%d List:%d, want Get:%d List:%d", tc.lister.getCalls, tc.lister.listCalls, tc.wantGet, tc.wantList)
			}
		})
	}
}

func Test_isSchedulableAfterTargetPodUpdateLabel(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	pod := st.MakePod().UID("p").Name("p").Label("foo", "baz").Obj()

	snapshot := cache.NewSnapshot(nil, nil)
	pl := plugintesting.SetupPluginWithInformers(ctx, t, schedruntime.FactoryAdapter(feature.Features{}, New), &config.InterPodAffinityArgs{}, snapshot, namespaces)
	p := pl.(*InterPodAffinity)
	actualHint, err := p.isSchedulableAfterTargetPodUpdateLabel(logger, pod, pod.DeepCopy(), pod.DeepCopy())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff(fwk.Queue, actualHint); diff != "" {
		t.Errorf("unexpected QueueingHint (-want, +got):\n%s", diff)
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
