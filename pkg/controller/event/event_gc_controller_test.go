package event

import (
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestGC(t *testing.T) {
	timea, err := time.Parse(time.RFC3339, "2006-01-02T15:04:05Z")
	if err != nil {
		t.Fatal(err)
	}
	timeb, _ := time.Parse(time.RFC3339, "2016-01-02T15:04:05Z")
	testCases := []struct {
		events            []api.Event
		shouldGC          func(api.Event) bool
		deletedEventNames sets.String
	}{
		{
			events: []api.Event{
				{
					ObjectMeta: api.ObjectMeta{Name: "a", CreationTimestamp: unversioned.Time{timea}},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "b", CreationTimestamp: unversioned.Time{timeb}},
				},
			},
			shouldGC: func(event api.Event) bool {
				timegc, _ := time.Parse(time.RFC3339, "2020-01-02T15:04:05Z")
				return event.CreationTimestamp.Time.Before(timegc)
			},
			deletedEventNames: sets.NewString("a", "b"),
		},
		{
			events: []api.Event{
				{
					ObjectMeta: api.ObjectMeta{Name: "a", CreationTimestamp: unversioned.Time{timea}},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "b", CreationTimestamp: unversioned.Time{timeb}},
				},
			},
			shouldGC: func(event api.Event) bool {
				timegc, _ := time.Parse(time.RFC3339, "2010-01-02T15:04:05Z")
				return event.CreationTimestamp.Time.Before(timegc)
			},
			deletedEventNames: sets.NewString("a"),
		},
		{
			events: []api.Event{
				{
					ObjectMeta: api.ObjectMeta{Name: "a", CreationTimestamp: unversioned.Time{timea}},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "b", CreationTimestamp: unversioned.Time{timeb}},
				},
			},
			shouldGC: func(event api.Event) bool {
				timegc, _ := time.Parse(time.RFC3339, "2000-01-02T15:04:05Z")
				return event.CreationTimestamp.Time.Before(timegc)
			},
			deletedEventNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := testclient.NewSimpleFake()
		gcc := New(client, controller.NoResyncPeriodFunc, time.Duration(0))
		deletedEventNames := make([]string, 0)

		var lock sync.Mutex
		gcc.deleteEvent = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedEventNames = append(deletedEventNames, name)
			return nil
		}
		gcc.shouldGC = test.shouldGC

		for i := range test.events {
			gcc.eventStore.Store.Add(&test.events[i])
		}

		gcc.gc()

		pass := true
		for _, event := range deletedEventNames {
			if !test.deletedEventNames.Has(event) {
				pass = false
			}
		}
		if len(deletedEventNames) != len(test.deletedEventNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]event's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedEventNames, deletedEventNames)
		}
	}
}
