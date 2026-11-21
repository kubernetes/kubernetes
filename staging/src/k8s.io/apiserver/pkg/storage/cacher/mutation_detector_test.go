/*
Copyright The Kubernetes Authors.

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

package cacher

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWatchCacheMutationDetector(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	const listKey = "/pods/ns"
	makePodFn := func(name, rv string) *example.Pod {
		pod := makeExamplePod(name, "ns", "", map[string]string{"team": "blue"})
		pod.ResourceVersion = rv
		pod.Spec.NodeName = name + "-node"
		return pod
	}
	makePodListFn := func(rv string, pods ...*example.Pod) *example.PodList {
		items := make([]example.Pod, len(pods))
		for i, pod := range pods {
			items[i] = *pod
		}
		return &example.PodList{
			ListMeta: metav1.ListMeta{ResourceVersion: rv},
			Items:    items,
		}
	}
	makeEventFn := func(eventType watch.EventType, obj *example.Pod) watch.Event {
		return watch.Event{Type: eventType, Object: obj}
	}
	listAndMutateResponseFn := func(opts storage.ListOptions, want *example.PodList) func(*testing.T, *Cacher) {
		opts.Recursive = true
		return func(t *testing.T, c *Cacher) {
			var list example.PodList
			require.NoError(t, c.GetList(t.Context(), listKey, opts, &list))
			require.Equal(t, want, &list)
			list.Items[0].Labels["team"] = "red"
		}
	}
	mutationPanicMessage := func(name, rv string) string {
		return fmt.Sprintf("watch cache mutation detected for *example.Pod %q, admitted at RV %q", "ns/"+name, rv)
	}

	for _, test := range []struct {
		name                                string
		initialStorageObjects               *example.PodList
		subsequentStorageEvents             []watch.Event
		sendRequestAndMaybeMutateResponseFn func(*testing.T, *Cacher)
		wantPanicMessage                    string
	}{
		{
			name:                  "LIST mutation of a Pod loaded by the initial LIST",
			initialStorageObjects: makePodListFn("42", makePodFn("target", "42")),
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(
				storage.ListOptions{ResourceVersion: "0"},
				makePodListFn("42", makePodFn("target", "42")),
			),
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name:                    "LIST mutation of a Pod inserted by an Added event",
			subsequentStorageEvents: []watch.Event{makeEventFn(watch.Added, makePodFn("target", "42"))},
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(
				storage.ListOptions{ResourceVersion: "0"},
				makePodListFn("42", makePodFn("target", "42")),
			),
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name: "LIST mutation of the new Pod version after a Modified event",
			subsequentStorageEvents: []watch.Event{
				makeEventFn(watch.Added, makePodFn("target", "42")),
				makeEventFn(watch.Modified, makePodFn("target", "43")),
			},
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(
				storage.ListOptions{ResourceVersion: "0"},
				makePodListFn("43", makePodFn("target", "43")),
			),
			wantPanicMessage: mutationPanicMessage("target", "43"),
		},
		{
			name: "indexed LIST mutation of a Pod selected by node name",
			subsequentStorageEvents: []watch.Event{
				makeEventFn(watch.Added, makePodFn("first", "41")),
				makeEventFn(watch.Added, makePodFn("target", "42")),
			},
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(storage.ListOptions{
				ResourceVersion: "0",
				Predicate: storage.SelectionPredicate{
					Label:       labels.Everything(),
					Field:       fields.OneTermEqualSelector("spec.nodeName", "target-node"),
					IndexFields: []string{"spec.nodeName"},
				},
			}, makePodListFn("42", makePodFn("target", "42"))),
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name: "exact LIST mutation of the old Pod version after an update",
			subsequentStorageEvents: []watch.Event{
				makeEventFn(watch.Added, makePodFn("target", "42")),
				makeEventFn(watch.Modified, makePodFn("target", "43")),
			},
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(storage.ListOptions{
				ResourceVersion:      "42",
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			}, makePodListFn("42", makePodFn("target", "42"))),
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name: "exact LIST mutation of a Pod retained after deletion",
			subsequentStorageEvents: []watch.Event{
				makeEventFn(watch.Added, makePodFn("target", "42")),
				makeEventFn(watch.Deleted, makePodFn("target", "43")),
			},
			sendRequestAndMaybeMutateResponseFn: listAndMutateResponseFn(storage.ListOptions{
				ResourceVersion:      "42",
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			}, makePodListFn("42", makePodFn("target", "42"))),
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name: "continued LIST mutation of a Pod retained after deletion",
			subsequentStorageEvents: []watch.Event{
				makeEventFn(watch.Added, makePodFn("first", "41")),
				makeEventFn(watch.Added, makePodFn("target", "42")),
				makeEventFn(watch.Deleted, makePodFn("target", "43")),
			},
			sendRequestAndMaybeMutateResponseFn: func(t *testing.T, c *Cacher) {
				opts := storage.ListOptions{Predicate: storage.SelectionPredicate{Limit: 1}}
				continueToken, _, err := storage.PrepareContinueToken(listKey+"/first", listKey+"/", 42, 2, true, opts)
				require.NoError(t, err)
				opts.Predicate.Continue = continueToken
				listAndMutateResponseFn(opts, makePodListFn("42", makePodFn("target", "42")))(t, c)
			},
			wantPanicMessage: mutationPanicMessage("target", "42"),
		},
		{
			name:                    "mutating a GET response leaves the cache intact",
			subsequentStorageEvents: []watch.Event{makeEventFn(watch.Added, makePodFn("target", "42"))},
			sendRequestAndMaybeMutateResponseFn: func(t *testing.T, c *Cacher) {
				want := makePodFn("target", "42")
				key, err := c.watchCache.config.keyFunc(want)
				require.NoError(t, err)
				var pod example.Pod
				require.NoError(t, c.Get(t.Context(), key, storage.GetOptions{ResourceVersion: "0"}, &pod))
				require.Equal(t, want, &pod)
				pod.Labels["team"] = "red"
			},
			wantPanicMessage: "",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := newTestCacherConfig(nil)
			wc, err := newWatchCache(
				config.KeyFunc,
				nil,
				config.GetAttrsFunc,
				config.Versioner,
				podIndexers(),
				config.Clock,
				config.EventsHistoryWindow,
				config.WatchCacheMaxEvents,
				config.GroupResource,
				nil,
				nil,
				true,
			)
			require.NoError(t, err)
			c := &Cacher{watchCache: wc, ready: newReady(config.Clock), resourcePrefix: config.ResourcePrefix, versioner: config.Versioner}
			c.ready.setReady()

			if initialList := test.initialStorageObjects; initialList != nil {
				objects := make([]any, len(initialList.Items))
				for i := range initialList.Items {
					objects[i] = &initialList.Items[i]
				}
				require.NoError(t, wc.Replace(objects, initialList.ResourceVersion))
			}
			for _, event := range test.subsequentStorageEvents {
				rv, err := config.Versioner.ObjectResourceVersion(event.Object)
				require.NoError(t, err)
				require.NoError(t, wc.processEvent(event, rv))
			}

			checkMutations := func() { wc.storage.CheckMutations(t.Context()) }
			require.NotPanics(t, checkMutations)
			test.sendRequestAndMaybeMutateResponseFn(t, c)
			if test.wantPanicMessage == "" {
				require.NotPanics(t, checkMutations)
			} else {
				require.PanicsWithValue(t, test.wantPanicMessage, checkMutations)
			}
			runtime.KeepAlive(wc)
		})
	}
}
