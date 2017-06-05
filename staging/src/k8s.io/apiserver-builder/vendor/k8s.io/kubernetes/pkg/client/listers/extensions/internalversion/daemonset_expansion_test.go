/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestDaemonSetLister(t *testing.T) {
	store := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	lister := NewDaemonSetLister(store)
	testCases := []struct {
		inDSs             []*extensions.DaemonSet
		list              func() ([]*extensions.DaemonSet, error)
		outDaemonSetNames sets.String
		expectErr         bool
	}{
		// Basic listing
		{
			inDSs: []*extensions.DaemonSet{
				{ObjectMeta: metav1.ObjectMeta{Name: "basic"}},
			},
			list: func() ([]*extensions.DaemonSet, error) {
				return lister.List(labels.Everything())
			},
			outDaemonSetNames: sets.NewString("basic"),
		},
		// Listing multiple daemon sets
		{
			inDSs: []*extensions.DaemonSet{
				{ObjectMeta: metav1.ObjectMeta{Name: "basic"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "complex"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "complex2"}},
			},
			list: func() ([]*extensions.DaemonSet, error) {
				return lister.List(labels.Everything())
			},
			outDaemonSetNames: sets.NewString("basic", "complex", "complex2"),
		},
		// No pod labels
		{
			inDSs: []*extensions.DaemonSet{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "basic", Namespace: "ns"},
					Spec: extensions.DaemonSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "baz"}},
					},
				},
			},
			list: func() ([]*extensions.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "ns"},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString(),
			expectErr:         true,
		},
		// No DS selectors
		{
			inDSs: []*extensions.DaemonSet{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "basic", Namespace: "ns"},
				},
			},
			list: func() ([]*extensions.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "ns",
						Labels:    map[string]string{"foo": "bar"},
					},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString(),
			expectErr:         true,
		},
		// Matching labels to selectors and namespace
		{
			inDSs: []*extensions.DaemonSet{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "foo"},
					Spec: extensions.DaemonSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: extensions.DaemonSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			list: func() ([]*extensions.DaemonSet, error) {
				pod := &api.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "ns",
					},
				}
				return lister.GetPodDaemonSets(pod)
			},
			outDaemonSetNames: sets.NewString("bar"),
		},
	}
	for _, c := range testCases {
		for _, r := range c.inDSs {
			store.Add(r)
		}

		daemonSets, err := c.list()
		if err != nil && c.expectErr {
			continue
		} else if c.expectErr {
			t.Error("Expected error, got none")
			continue
		} else if err != nil {
			t.Errorf("Unexpected error %#v", err)
			continue
		}
		daemonSetNames := make([]string, len(daemonSets))
		for ix := range daemonSets {
			daemonSetNames[ix] = daemonSets[ix].Name
		}
		if !c.outDaemonSetNames.HasAll(daemonSetNames...) || len(daemonSetNames) != len(c.outDaemonSetNames) {
			t.Errorf("Unexpected got controllers %+v expected %+v", daemonSetNames, c.outDaemonSetNames)
		}
	}
}
