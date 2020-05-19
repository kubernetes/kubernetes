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
	"context"
	"reflect"
	goruntime "runtime"
	"sort"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/watch"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

// TestEventProcessorExit is expected to timeout if the event processor fails
// to exit when stopped.
func TestEventProcessorExit(t *testing.T) {
	event := watch.Event{}

	tests := []struct {
		name  string
		write func(e *eventProcessor)
	}{
		{
			name: "exit on blocked read",
			write: func(e *eventProcessor) {
				e.push(event)
			},
		},
		{
			name: "exit on blocked write",
			write: func(e *eventProcessor) {
				e.push(event)
				e.push(event)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			out := make(chan watch.Event)
			e := newEventProcessor(out)

			test.write(e)

			exited := make(chan struct{})
			go func() {
				e.run()
				close(exited)
			}()

			<-out
			e.stop()
			goruntime.Gosched()
			<-exited
		})
	}
}

type apiInt int

func (apiInt) GetObjectKind() schema.ObjectKind { return nil }
func (apiInt) DeepCopyObject() runtime.Object   { return nil }

func TestEventProcessorOrdersEvents(t *testing.T) {
	out := make(chan watch.Event)
	e := newEventProcessor(out)
	go e.run()

	numProcessed := 0
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	go func() {
		for i := 0; i < 1000; i++ {
			e := <-out
			if got, want := int(e.Object.(apiInt)), i; got != want {
				t.Errorf("unexpected event: got=%d, want=%d", got, want)
			}
			numProcessed++
		}
		cancel()
	}()

	for i := 0; i < 1000; i++ {
		e.push(watch.Event{Object: apiInt(i)})
	}

	<-ctx.Done()
	e.stop()

	if numProcessed != 1000 {
		t.Errorf("unexpected number of events processed: %d", numProcessed)
	}

}

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
					return fake.CoreV1().Secrets("").List(context.TODO(), options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return fake.CoreV1().Secrets("").Watch(context.TODO(), options)
				},
			}
			_, _, w, done := NewIndexerInformerWatcher(lw, &corev1.Secret{})

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

			<-done
		})
	}

}
