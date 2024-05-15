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

package cache

import (
	"fmt"
	"sort"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

func TestWatchList(t *testing.T) {
	scenarios := []struct {
		name                string
		disableUseWatchList bool

		// closes listWatcher after sending the specified number of watch events
		closeAfterWatchEvents int
		// closes listWatcher after getting the specified number of watch requests
		closeAfterWatchRequests int
		// closes listWatcher after getting the specified number of list requests
		closeAfterListRequests int

		// stops Watcher after sending the specified number of watch events
		stopAfterWatchEvents int

		watchOptionsPredicate func(options metav1.ListOptions) error
		watchEvents           []watch.Event
		podList               *v1.PodList

		expectedRequestOptions []metav1.ListOptions
		expectedWatchRequests  int
		expectedListRequests   int
		expectedStoreContent   []v1.Pod
		expectedError          error
	}{
		{
			name:                  "the reflector won't be synced if the bookmark event has been received",
			watchEvents:           []watch.Event{{Type: watch.Added, Object: makePod("p1", "1")}},
			closeAfterWatchEvents: 1,
			expectedWatchRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{{
				SendInitialEvents:    pointer.Bool(true),
				AllowWatchBookmarks:  true,
				ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
				TimeoutSeconds:       pointer.Int64(1),
			}},
		},
		{
			name:                    "the reflector uses the old LIST/WATCH semantics if the UseWatchList is turned off",
			disableUseWatchList:     true,
			closeAfterWatchRequests: 1,
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "1"},
				Items:    []v1.Pod{*makePod("p1", "1")},
			},
			expectedWatchRequests: 1,
			expectedListRequests:  1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion: "0",
					Limit:           500,
				},
				{
					AllowWatchBookmarks: true,
					ResourceVersion:     "1",
					TimeoutSeconds:      pointer.Int64(1),
				}},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1")},
		},
		{
			name: "returning any other error than apierrors.NewInvalid forces fallback",
			watchOptionsPredicate: func(options metav1.ListOptions) error {
				if options.SendInitialEvents != nil && *options.SendInitialEvents {
					return fmt.Errorf("dummy error")
				}
				return nil
			},
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "1"},
				Items:    []v1.Pod{*makePod("p1", "1")},
			},
			closeAfterWatchEvents: 1,
			watchEvents:           []watch.Event{{Type: watch.Added, Object: makePod("p2", "2")}},
			expectedWatchRequests: 2,
			expectedListRequests:  1,
			expectedStoreContent:  []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					ResourceVersion: "0",
					Limit:           500,
				},
				{
					AllowWatchBookmarks: true,
					ResourceVersion:     "1",
					TimeoutSeconds:      pointer.Int64(1),
				},
			},
		},
		{
			name: "the reflector can fall back to old LIST/WATCH semantics when a server doesn't support streaming",
			watchOptionsPredicate: func(options metav1.ListOptions) error {
				if options.SendInitialEvents != nil && *options.SendInitialEvents {
					return apierrors.NewInvalid(schema.GroupKind{}, "streaming is not allowed", nil)
				}
				return nil
			},
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "1"},
				Items:    []v1.Pod{*makePod("p1", "1")},
			},
			closeAfterWatchEvents: 1,
			watchEvents:           []watch.Event{{Type: watch.Added, Object: makePod("p2", "2")}},
			expectedWatchRequests: 2,
			expectedListRequests:  1,
			expectedStoreContent:  []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					ResourceVersion: "0",
					Limit:           500,
				},
				{
					AllowWatchBookmarks: true,
					ResourceVersion:     "1",
					TimeoutSeconds:      pointer.Int64(1),
				},
			},
		},
		{
			name:                  "prove that the reflector is synced after receiving a bookmark event",
			closeAfterWatchEvents: 3,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Added, Object: makePod("p2", "2")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "2",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
			},
			expectedWatchRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{{
				SendInitialEvents:    pointer.Bool(true),
				AllowWatchBookmarks:  true,
				ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
				TimeoutSeconds:       pointer.Int64(1),
			}},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
		},
		{
			name:                  "check if Updates and Deletes events are propagated during streaming (until the bookmark is received)",
			closeAfterWatchEvents: 6,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Added, Object: makePod("p2", "2")},
				{Type: watch.Modified, Object: func() runtime.Object {
					p1 := makePod("p1", "3")
					p1.Spec.ActiveDeadlineSeconds = pointer.Int64(12)
					return p1
				}()},
				{Type: watch.Added, Object: makePod("p3", "4")},
				{Type: watch.Deleted, Object: makePod("p3", "5")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "5",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
			},
			expectedWatchRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{{
				SendInitialEvents:    pointer.Bool(true),
				AllowWatchBookmarks:  true,
				ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
				TimeoutSeconds:       pointer.Int64(1),
			}},
			expectedStoreContent: []v1.Pod{
				*makePod("p2", "2"),
				func() v1.Pod {
					p1 := *makePod("p1", "3")
					p1.Spec.ActiveDeadlineSeconds = pointer.Int64(12)
					return p1
				}(),
			},
		},
		{
			name: "checks if the reflector retries 429",
			watchOptionsPredicate: func() func(options metav1.ListOptions) error {
				counter := 1
				return func(options metav1.ListOptions) error {
					if counter < 3 {
						counter++
						return apierrors.NewTooManyRequests("busy, check again later", 1)
					}
					return nil
				}
			}(),
			closeAfterWatchEvents: 2,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "2",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
			},
			expectedWatchRequests: 3,
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
			},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1")},
		},
		{
			name:                  "check if stopping a watcher before sync results in creating a new watch-list request",
			stopAfterWatchEvents:  1,
			closeAfterWatchEvents: 3,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				// second request
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "1",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
			},
			expectedWatchRequests: 2,
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
			},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1")},
		},
		{
			name:                  "stopping a watcher after synchronization results in creating a new watch request",
			stopAfterWatchEvents:  4,
			closeAfterWatchEvents: 5,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Added, Object: makePod("p2", "2")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "2",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
				{Type: watch.Added, Object: makePod("p3", "3")},
				// second request
				{Type: watch.Added, Object: makePod("p4", "4")},
			},
			expectedWatchRequests: 2,
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					AllowWatchBookmarks: true,
					ResourceVersion:     "3",
					TimeoutSeconds:      pointer.Int64(1),
				},
			},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3"), *makePod("p4", "4")},
		},
		{
			name: "expiring an established watcher results in returning an error from the reflector",
			watchOptionsPredicate: func() func(options metav1.ListOptions) error {
				counter := 0
				return func(options metav1.ListOptions) error {
					counter++
					if counter == 2 {
						return apierrors.NewResourceExpired("rv already expired")
					}
					return nil
				}
			}(),
			stopAfterWatchEvents: 3,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "2",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
					},
				}},
				{Type: watch.Added, Object: makePod("p3", "3")},
			},
			expectedWatchRequests: 2,
			expectedRequestOptions: []metav1.ListOptions{
				{
					SendInitialEvents:    pointer.Bool(true),
					AllowWatchBookmarks:  true,
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					TimeoutSeconds:       pointer.Int64(1),
				},
				{
					AllowWatchBookmarks: true,
					ResourceVersion:     "3",
					TimeoutSeconds:      pointer.Int64(1),
				},
			},
			expectedStoreContent: []v1.Pod{*makePod("p1", "1"), *makePod("p3", "3")},
			expectedError:        apierrors.NewResourceExpired("rv already expired"),
		},
		{
			name:                  "prove that the reflector is checking the value of the initialEventsEnd annotation",
			closeAfterWatchEvents: 3,
			watchEvents: []watch.Event{
				{Type: watch.Added, Object: makePod("p1", "1")},
				{Type: watch.Added, Object: makePod("p2", "2")},
				{Type: watch.Bookmark, Object: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						ResourceVersion: "2",
						Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "false"},
					},
				}},
			},
			expectedWatchRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{{
				SendInitialEvents:    pointer.Bool(true),
				AllowWatchBookmarks:  true,
				ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
				TimeoutSeconds:       pointer.Int64(1),
			}},
		},
	}
	for _, s := range scenarios {
		t.Run(s.name, func(t *testing.T) {
			scenario := s // capture as local variable
			listWatcher, store, reflector, stopCh := testData()
			go func() {
				for i, e := range scenario.watchEvents {
					listWatcher.fakeWatcher.Action(e.Type, e.Object)
					if i+1 == scenario.stopAfterWatchEvents {
						listWatcher.StopAndRecreateWatch()
						continue
					}
					if i+1 == scenario.closeAfterWatchEvents {
						close(stopCh)
					}
				}
			}()
			listWatcher.watchOptionsPredicate = scenario.watchOptionsPredicate
			listWatcher.closeAfterWatchRequests = scenario.closeAfterWatchRequests
			listWatcher.customListResponse = scenario.podList
			listWatcher.closeAfterListRequests = scenario.closeAfterListRequests
			if scenario.disableUseWatchList {
				reflector.UseWatchList = ptr.To(false)
			}

			err := reflector.ListAndWatch(stopCh)
			if scenario.expectedError != nil && err == nil {
				t.Fatalf("expected error %q, got nil", scenario.expectedError)
			}
			if scenario.expectedError == nil && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if scenario.expectedError != nil && err.Error() != scenario.expectedError.Error() {
				t.Fatalf("expected error %q, got %q", scenario.expectedError, err.Error())
			}

			verifyWatchCounter(t, listWatcher, scenario.expectedWatchRequests)
			verifyListCounter(t, listWatcher, scenario.expectedListRequests)
			verifyRequestOptions(t, listWatcher, scenario.expectedRequestOptions)
			verifyStore(t, store, scenario.expectedStoreContent)
		})
	}
}

func verifyRequestOptions(t *testing.T, lw *fakeListWatcher, expectedRequestOptions []metav1.ListOptions) {
	if len(lw.requestOptions) != len(expectedRequestOptions) {
		t.Fatalf("expected to receive exactly %v requests, got %v", len(expectedRequestOptions), len(lw.requestOptions))
	}

	for index, expectedRequestOption := range expectedRequestOptions {
		actualRequestOption := lw.requestOptions[index]
		if actualRequestOption.TimeoutSeconds == nil && expectedRequestOption.TimeoutSeconds != nil {
			t.Fatalf("expected the request to specify TimeoutSeconds option but it didn't, actual = %#v, expected = %#v", actualRequestOption, expectedRequestOption)
		}
		if actualRequestOption.TimeoutSeconds != nil && expectedRequestOption.TimeoutSeconds == nil {
			t.Fatalf("unexpected TimeoutSeconds option specified, actual = %#v, expected = %#v", actualRequestOption, expectedRequestOption)
		}
		// ignore actual values
		actualRequestOption.TimeoutSeconds = nil
		expectedRequestOption.TimeoutSeconds = nil
		if !cmp.Equal(actualRequestOption, expectedRequestOption) {
			t.Fatalf("expected %#v, got %#v", expectedRequestOption, actualRequestOption)
		}
	}
}

func verifyListCounter(t *testing.T, lw *fakeListWatcher, expectedListCounter int) {
	if lw.listCounter != expectedListCounter {
		t.Fatalf("unexpected number of LIST requests, got: %v, expected: %v", lw.listCounter, expectedListCounter)
	}
}

func verifyWatchCounter(t *testing.T, lw *fakeListWatcher, expectedWatchCounter int) {
	if lw.watchCounter != expectedWatchCounter {
		t.Fatalf("unexpected number of WATCH requests, got: %v, expected: %v", lw.watchCounter, expectedWatchCounter)
	}
}

type byName []v1.Pod

func (a byName) Len() int           { return len(a) }
func (a byName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func verifyStore(t *testing.T, s Store, expectedPods []v1.Pod) {
	rawPods := s.List()
	actualPods := []v1.Pod{}
	for _, p := range rawPods {
		actualPods = append(actualPods, *p.(*v1.Pod))
	}

	sort.Sort(byName(actualPods))
	sort.Sort(byName(expectedPods))
	if !cmp.Equal(actualPods, expectedPods, cmpopts.EquateEmpty()) {
		t.Fatalf("unexpected store content, diff: %s", cmp.Diff(actualPods, expectedPods))
	}
}

func makePod(name, rv string) *v1.Pod {
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, ResourceVersion: rv, UID: types.UID(name)}}
}

func testData() (*fakeListWatcher, Store, *Reflector, chan struct{}) {
	s := NewStore(MetaNamespaceKeyFunc)
	stopCh := make(chan struct{})
	lw := &fakeListWatcher{
		fakeWatcher: watch.NewFake(),
		stop: func() {
			close(stopCh)
		},
	}
	r := NewReflector(lw, &v1.Pod{}, s, 0)
	r.UseWatchList = ptr.To(true)

	return lw, s, r, stopCh
}

type fakeListWatcher struct {
	lock                    sync.Mutex
	fakeWatcher             *watch.FakeWatcher
	listCounter             int
	watchCounter            int
	closeAfterWatchRequests int
	closeAfterListRequests  int
	stop                    func()

	requestOptions []metav1.ListOptions

	customListResponse    *v1.PodList
	watchOptionsPredicate func(options metav1.ListOptions) error
}

func (lw *fakeListWatcher) List(options metav1.ListOptions) (runtime.Object, error) {
	lw.listCounter++
	lw.requestOptions = append(lw.requestOptions, options)
	if lw.listCounter == lw.closeAfterListRequests {
		lw.stop()
	}
	if lw.customListResponse != nil {
		return lw.customListResponse, nil
	}
	return nil, fmt.Errorf("not implemented")
}

func (lw *fakeListWatcher) Watch(options metav1.ListOptions) (watch.Interface, error) {
	lw.watchCounter++
	lw.requestOptions = append(lw.requestOptions, options)
	if lw.watchCounter == lw.closeAfterWatchRequests {
		lw.stop()
	}
	if lw.watchOptionsPredicate != nil {
		if err := lw.watchOptionsPredicate(options); err != nil {
			return nil, err
		}
	}
	lw.lock.Lock()
	defer lw.lock.Unlock()
	return lw.fakeWatcher, nil
}

func (lw *fakeListWatcher) StopAndRecreateWatch() {
	lw.lock.Lock()
	defer lw.lock.Unlock()
	lw.fakeWatcher.Stop()
	lw.fakeWatcher = watch.NewFake()
}
