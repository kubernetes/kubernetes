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

package rest

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"regexp"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
)

func TestWatchListResult(t *testing.T) {
	scenarios := []struct {
		name   string
		target WatchListResult
		result runtime.Object

		expectedResult *v1.PodList
		expectedErr    error
	}{
		{
			name:        "not a pointer",
			result:      fakeObj{},
			expectedErr: fmt.Errorf("rest.fakeObj is not a list: expected pointer, but got rest.fakeObj type"),
		},
		{
			name:        "nil input won't panic",
			result:      nil,
			expectedErr: fmt.Errorf("<nil> is not a list: expected pointer, but got invalid kind"),
		},
		{
			name:        "not a list",
			result:      &v1.Pod{},
			expectedErr: fmt.Errorf("*v1.Pod is not a list: no Items field in this object"),
		},
		{
			name:   "invalid base64EncodedInitialEventsListBlueprint",
			result: &v1.PodList{},
			target: WatchListResult{
				base64EncodedInitialEventsListBlueprint: "invalid",
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedErr: fmt.Errorf("failed to decode the received blueprint list, err illegal base64 data at input byte 4"),
		},
		{
			name:   "empty list",
			result: &v1.PodList{},
			target: WatchListResult{
				base64EncodedInitialEventsListBlueprint: encodeObjectToBase64String(makeEmptyPodList(), t),
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedResult: &v1.PodList{
				TypeMeta: metav1.TypeMeta{Kind: "PodList"},
				Items:    []v1.Pod{},
			},
		},
		{
			name:   "rv is applied",
			result: &v1.PodList{},
			target: WatchListResult{
				initialEventsEndBookmarkRV:              "100",
				base64EncodedInitialEventsListBlueprint: encodeObjectToBase64String(makeEmptyPodList(), t),
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedResult: &v1.PodList{
				TypeMeta: metav1.TypeMeta{Kind: "PodList"},
				ListMeta: metav1.ListMeta{ResourceVersion: "100"},
				Items:    []v1.Pod{},
			},
		},
		{
			name:   "items are applied",
			result: &v1.PodList{},
			target: WatchListResult{
				items:                                   []runtime.Object{makePod(1), makePod(2)},
				base64EncodedInitialEventsListBlueprint: encodeObjectToBase64String(makeEmptyPodList(), t),
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedResult: &v1.PodList{
				TypeMeta: metav1.TypeMeta{Kind: "PodList"},
				Items:    []v1.Pod{*makePod(1), *makePod(2)},
			},
		},
		{
			name:   "list's object type mismatch",
			result: &v1.PodList{},
			target: WatchListResult{
				items:                                   []runtime.Object{makeNamespace("1")},
				base64EncodedInitialEventsListBlueprint: encodeObjectToBase64String(makeEmptyPodList(), t),
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedErr: fmt.Errorf("received object type = v1.Namespace at index = 0, doesn't match the list item type = v1.Pod"),
		},
		{
			name:   "list type mismatch",
			result: &v1.SecretList{},
			target: WatchListResult{
				items:                                   []runtime.Object{makePod(1), makePod(2)},
				base64EncodedInitialEventsListBlueprint: encodeObjectToBase64String(makeEmptyPodList(), t),
				negotiatedObjectDecoder:                 newJSONSerializer(),
			},
			expectedErr: fmt.Errorf("unable to decode /v1, Kind=PodList into *v1.SecretList"),
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			err := scenario.target.Into(scenario.result)
			if scenario.expectedErr != nil && err == nil {
				t.Fatalf("expected an error = %v, got nil", scenario.expectedErr)
			}
			if scenario.expectedErr == nil && err != nil {
				t.Fatalf("didn't expect an error, got =  %v", err)
			}
			if err != nil {
				if scenario.expectedErr.Error() != err.Error() {
					t.Fatalf("unexpected err = %v, expected = %v", err, scenario.expectedErr)
				}
				return
			}
			if !apiequality.Semantic.DeepEqual(scenario.expectedResult, scenario.result) {
				t.Errorf("diff: %v", cmp.Diff(scenario.expectedResult, scenario.result))
			}
		})
	}
}

func TestWatchListSuccess(t *testing.T) {
	scenarios := []struct {
		name                    string
		watchEvents             []watch.Event
		negotiatedObjectDecoder runtime.Serializer

		expectedResult *v1.PodList
	}{
		{
			name:                    "happy path",
			negotiatedObjectDecoder: newJSONSerializer(),
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod(1)},
				{Type: watch.Added, Object: makePod(2)},
				{Type: watch.Bookmark, Object: makeBookmarkEvent(5, t)},
			},
			expectedResult: &v1.PodList{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "",
					Kind:       "PodList",
				},
				ListMeta: metav1.ListMeta{ResourceVersion: "5"},
				Items:    []v1.Pod{*makePod(1), *makePod(2)},
			},
		},
		{
			name:                    "only the bookmark",
			negotiatedObjectDecoder: newJSONSerializer(),
			watchEvents: []watch.Event{
				{Type: watch.Bookmark, Object: makeBookmarkEvent(5, t)},
			},
			expectedResult: &v1.PodList{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "",
					Kind:       "PodList",
				},
				ListMeta: metav1.ListMeta{ResourceVersion: "5"},
				Items:    []v1.Pod{},
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			ctx := context.Background()
			fakeWatcher := watch.NewFake()
			target := &Request{
				c: &RESTClient{
					content: ClientContentConfig{},
				},
			}

			go func(watchEvents []watch.Event) {
				for _, watchEvent := range watchEvents {
					fakeWatcher.Action(watchEvent.Type, watchEvent.Object)
				}
			}(scenario.watchEvents)

			res := target.handleWatchList(ctx, fakeWatcher, scenario.negotiatedObjectDecoder)
			if res.err != nil {
				t.Fatal(res.err)
			}

			result := &v1.PodList{}
			if err := res.Into(result); err != nil {
				t.Fatal(err)
			}
			if !apiequality.Semantic.DeepEqual(scenario.expectedResult, result) {
				t.Errorf("diff: %v", cmp.Diff(scenario.expectedResult, result))
			}
			if !fakeWatcher.IsStopped() {
				t.Fatalf("the watcher wasn't stopped")
			}
		})
	}
}

func TestWatchListFailure(t *testing.T) {
	scenarios := []struct {
		name        string
		ctx         context.Context
		watcher     *watch.FakeWatcher
		watchEvents []watch.Event

		expectedError error
	}{
		{
			name: "request stop",
			ctx: func() context.Context {
				ctx, ctxCancel := context.WithCancel(context.TODO())
				ctxCancel()
				return ctx
			}(),
			watcher:       watch.NewFake(),
			expectedError: fmt.Errorf("context canceled"),
		},
		{
			name: "stop watcher",
			ctx:  context.TODO(),
			watcher: func() *watch.FakeWatcher {
				w := watch.NewFake()
				w.Stop()
				return w
			}(),
			expectedError: fmt.Errorf("unexpected watch close"),
		},
		{
			name:          "stop on watch.Error",
			ctx:           context.TODO(),
			watcher:       watch.NewFake(),
			watchEvents:   []watch.Event{{Type: watch.Error, Object: &apierrors.NewInternalError(fmt.Errorf("dummy errror")).ErrStatus}},
			expectedError: fmt.Errorf("Internal error occurred: dummy errror"),
		},
		{
			name:          "incorrect watch type (Deleted)",
			ctx:           context.TODO(),
			watcher:       watch.NewFake(),
			watchEvents:   []watch.Event{{Type: watch.Deleted, Object: makePod(1)}},
			expectedError: fmt.Errorf("unexpected watch event .*, expected to only receive watch.Added and watch.Bookmark events"),
		},
		{
			name:          "incorrect watch type (Modified)",
			ctx:           context.TODO(),
			watcher:       watch.NewFake(),
			watchEvents:   []watch.Event{{Type: watch.Modified, Object: makePod(1)}},
			expectedError: fmt.Errorf("unexpected watch event .*, expected to only receive watch.Added and watch.Bookmark events"),
		},
		{
			name:          "unordered input returns an error",
			ctx:           context.TODO(),
			watcher:       watch.NewFake(),
			watchEvents:   []watch.Event{{Type: watch.Added, Object: makePod(3)}, {Type: watch.Added, Object: makePod(1)}},
			expectedError: fmt.Errorf("cannot add the obj .* with the key = ns/pod-1, as it violates the ordering guarantees provided by the watchlist feature in beta phase, lastInsertedKey was = ns/pod-3"),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			target := &Request{}
			go func(w *watch.FakeWatcher, watchEvents []watch.Event) {
				for _, event := range watchEvents {
					w.Action(event.Type, event.Object)
				}
			}(scenario.watcher, scenario.watchEvents)

			res := target.handleWatchList(scenario.ctx, scenario.watcher, nil /*TODO*/)
			resErr := res.Into(nil)
			if resErr == nil {
				t.Fatal("expected to get an error, got nil")
			}
			matched, err := regexp.MatchString(scenario.expectedError.Error(), resErr.Error())
			if err != nil {
				t.Fatal(err)
			}
			if !matched {
				t.Fatalf("unexpected err = %v, expected = %v", resErr, scenario.expectedError)
			}
			if !scenario.watcher.IsStopped() {
				t.Fatalf("the watcher wasn't stopped")
			}
		})
	}
}

func TestWatchListWhenFeatureGateDisabled(t *testing.T) {
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, false)
	expectedError := fmt.Errorf("%q feature gate is not enabled", clientfeatures.WatchListClient)
	target := &Request{}

	res := target.WatchList(context.TODO())

	resErr := res.Into(nil)
	if resErr == nil {
		t.Fatal("expected to get an error, got nil")
	}
	if resErr.Error() != expectedError.Error() {
		t.Fatalf("unexpected error: %v, expected: %v", resErr, expectedError)
	}
}

func makeEmptyPodList() *v1.PodList {
	return &v1.PodList{
		TypeMeta: metav1.TypeMeta{Kind: "PodList"},
		Items:    []v1.Pod{},
	}
}

func makePod(rv uint64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            fmt.Sprintf("pod-%d", rv),
			Namespace:       "ns",
			ResourceVersion: fmt.Sprintf("%d", rv),
			Annotations:     map[string]string{},
		},
	}
}

func makeNamespace(name string) *v1.Namespace {
	return &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func makeBookmarkEvent(rv uint64, t *testing.T) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			ResourceVersion: fmt.Sprintf("%d", rv),
			Annotations: map[string]string{
				metav1.InitialEventsAnnotationKey:              "true",
				metav1.InitialEventsListBlueprintAnnotationKey: encodeObjectToBase64String(makeEmptyPodList(), t),
			},
		},
	}
}

type fakeObj struct {
}

func (f fakeObj) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}

func (f fakeObj) DeepCopyObject() runtime.Object {
	return fakeObj{}
}

func newJSONSerializer() runtime.Serializer {
	return runtimejson.NewSerializerWithOptions(
		runtimejson.DefaultMetaFactory,
		clientgoscheme.Scheme,
		clientgoscheme.Scheme,
		runtimejson.SerializerOptions{},
	)
}

func encodeObjectToBase64String(obj runtime.Object, t *testing.T) string {
	e := newJSONSerializer()

	var buf bytes.Buffer
	err := e.Encode(obj, &buf)
	if err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}
