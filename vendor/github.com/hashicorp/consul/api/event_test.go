package api

import (
	"testing"

	"github.com/hashicorp/consul/testutil"
)

func TestEvent_FireList(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	event := c.Event()

	params := &UserEvent{Name: "foo"}
	id, meta, err := event.Fire(params, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	var events []*UserEvent
	var qm *QueryMeta
	testutil.WaitForResult(func() (bool, error) {
		events, qm, err = event.List("", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return len(events) > 0, err
	}, func(err error) {
		t.Fatalf("err: %#v", err)
	})

	if events[len(events)-1].ID != id {
		t.Fatalf("bad: %#v", events)
	}

	if qm.LastIndex != event.IDToIndex(id) {
		t.Fatalf("Bad: %#v", qm)
	}
}
