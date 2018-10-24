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

package watch

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/watch"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

type byEventTypeAndName []watch.Event

func (a byEventTypeAndName) Len() int      { return len(a) }
func (a byEventTypeAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byEventTypeAndName) Less(i, j int) bool {
	if a[i].Type < a[j].Type {
		return true
	}

	if a[i].Type > a[j].Type {
		return false
	}

	return a[i].Object.(*corev1.Secret).Name < a[j].Object.(*corev1.Secret).Name
}

func TestTicketer(t *testing.T) {
	tg := newTicketer()

	const numTickets = 100 // current golang limit for race detector is 8192 simultaneously alive goroutines
	var tickets []uint64
	for i := 0; i < numTickets; i++ {
		ticket := tg.GetTicket()
		tickets = append(tickets, ticket)

		exp, got := uint64(i), ticket
		if got != exp {
			t.Fatalf("expected ticket %d, got %d", exp, got)
		}
	}

	// shuffle tickets
	rand.Shuffle(len(tickets), func(i, j int) {
		tickets[i], tickets[j] = tickets[j], tickets[i]
	})

	res := make(chan uint64, len(tickets))
	for _, ticket := range tickets {
		go func(ticket uint64) {
			time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
			tg.WaitForTicket(ticket, func() {
				res <- ticket
			})
		}(ticket)
	}

	for i := 0; i < numTickets; i++ {
		exp, got := uint64(i), <-res
		if got != exp {
			t.Fatalf("expected ticket %d, got %d", exp, got)
		}
	}
}

func TestNewInformerWatcher(t *testing.T) {
	// Make sure there are no 2 same types of events on a secret with the same name or that might be flaky.
	tt := []struct {
		name    string
		objects []runtime.Object
		events  []watch.Event
	}{
		{
			name: "basic test",
			objects: []runtime.Object{
				&corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-1",
					},
					StringData: map[string]string{
						"foo-1": "initial",
					},
				},
				&corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-2",
					},
					StringData: map[string]string{
						"foo-2": "initial",
					},
				},
				&corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-3",
					},
					StringData: map[string]string{
						"foo-3": "initial",
					},
				},
			},
			events: []watch.Event{
				{
					Type: watch.Added,
					Object: &corev1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name: "pod-4",
						},
						StringData: map[string]string{
							"foo-4": "initial",
						},
					},
				},
				{
					Type: watch.Modified,
					Object: &corev1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name: "pod-2",
						},
						StringData: map[string]string{
							"foo-2": "new",
						},
					},
				},
				{
					Type: watch.Deleted,
					Object: &corev1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name: "pod-3",
						},
					},
				},
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			var expected []watch.Event
			for _, o := range tc.objects {
				expected = append(expected, watch.Event{
					Type:   watch.Added,
					Object: o.DeepCopyObject(),
				})
			}
			for _, e := range tc.events {
				expected = append(expected, *e.DeepCopy())
			}

			fake := fakeclientset.NewSimpleClientset(tc.objects...)
			fakeWatch := watch.NewFakeWithChanSize(len(tc.events), false)
			fake.PrependWatchReactor("secrets", testcore.DefaultWatchReactor(fakeWatch, nil))

			for _, e := range tc.events {
				fakeWatch.Action(e.Type, e.Object)
			}

			lw := &cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					return fake.Core().Secrets("").List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return fake.Core().Secrets("").Watch(options)
				},
			}
			_, _, w := NewIndexerInformerWatcher(lw, &corev1.Secret{})

			var result []watch.Event
		loop:
			for {
				var event watch.Event
				var ok bool
				select {
				case event, ok = <-w.ResultChan():
					if !ok {
						t.Errorf("Failed to read event: channel is already closed!")
						return
					}

					result = append(result, *event.DeepCopy())
				case <-time.After(time.Second * 1):
					// All the events are buffered -> this means we are done
					// Also the one sec will make sure that we would detect RetryWatcher's incorrect behaviour after last event
					break loop
				}
			}

			// Informers don't guarantee event order so we need to sort these arrays to compare them
			sort.Sort(byEventTypeAndName(expected))
			sort.Sort(byEventTypeAndName(result))

			if !reflect.DeepEqual(expected, result) {
				t.Error(spew.Errorf("\nexpected: %#v,\ngot:      %#v,\ndiff: %s", expected, result, diff.ObjectReflectDiff(expected, result)))
				return
			}

			// Fill in some data to test watch closing while there are some events to be read
			for _, e := range tc.events {
				fakeWatch.Action(e.Type, e.Object)
			}

			// Stop before reading all the data to make sure the informer can deal with closed channel
			w.Stop()

			// Wait a bit to see if the informer won't panic
			// TODO: Try to figure out a more reliable mechanism than time.Sleep (https://github.com/kubernetes/kubernetes/pull/50102/files#r184716591)
			time.Sleep(1 * time.Second)
		})
	}

}
