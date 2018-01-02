package events

import (
	"fmt"
	"testing"
	"time"

	"github.com/docker/docker/api/types/events"
	timetypes "github.com/docker/docker/api/types/time"
	eventstestutils "github.com/docker/docker/daemon/events/testutils"
)

func TestEventsLog(t *testing.T) {
	e := New()
	_, l1, _ := e.Subscribe()
	_, l2, _ := e.Subscribe()
	defer e.Evict(l1)
	defer e.Evict(l2)
	count := e.SubscribersCount()
	if count != 2 {
		t.Fatalf("Must be 2 subscribers, got %d", count)
	}
	actor := events.Actor{
		ID:         "cont",
		Attributes: map[string]string{"image": "image"},
	}
	e.Log("test", events.ContainerEventType, actor)
	select {
	case msg := <-l1:
		jmsg, ok := msg.(events.Message)
		if !ok {
			t.Fatalf("Unexpected type %T", msg)
		}
		if len(e.events) != 1 {
			t.Fatalf("Must be only one event, got %d", len(e.events))
		}
		if jmsg.Status != "test" {
			t.Fatalf("Status should be test, got %s", jmsg.Status)
		}
		if jmsg.ID != "cont" {
			t.Fatalf("ID should be cont, got %s", jmsg.ID)
		}
		if jmsg.From != "image" {
			t.Fatalf("From should be image, got %s", jmsg.From)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for broadcasted message")
	}
	select {
	case msg := <-l2:
		jmsg, ok := msg.(events.Message)
		if !ok {
			t.Fatalf("Unexpected type %T", msg)
		}
		if len(e.events) != 1 {
			t.Fatalf("Must be only one event, got %d", len(e.events))
		}
		if jmsg.Status != "test" {
			t.Fatalf("Status should be test, got %s", jmsg.Status)
		}
		if jmsg.ID != "cont" {
			t.Fatalf("ID should be cont, got %s", jmsg.ID)
		}
		if jmsg.From != "image" {
			t.Fatalf("From should be image, got %s", jmsg.From)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for broadcasted message")
	}
}

func TestEventsLogTimeout(t *testing.T) {
	e := New()
	_, l, _ := e.Subscribe()
	defer e.Evict(l)

	c := make(chan struct{})
	go func() {
		actor := events.Actor{
			ID: "image",
		}
		e.Log("test", events.ImageEventType, actor)
		close(c)
	}()

	select {
	case <-c:
	case <-time.After(time.Second):
		t.Fatal("Timeout publishing message")
	}
}

func TestLogEvents(t *testing.T) {
	e := New()

	for i := 0; i < eventsLimit+16; i++ {
		action := fmt.Sprintf("action_%d", i)
		id := fmt.Sprintf("cont_%d", i)
		from := fmt.Sprintf("image_%d", i)

		actor := events.Actor{
			ID:         id,
			Attributes: map[string]string{"image": from},
		}
		e.Log(action, events.ContainerEventType, actor)
	}
	time.Sleep(50 * time.Millisecond)
	current, l, _ := e.Subscribe()
	for i := 0; i < 10; i++ {
		num := i + eventsLimit + 16
		action := fmt.Sprintf("action_%d", num)
		id := fmt.Sprintf("cont_%d", num)
		from := fmt.Sprintf("image_%d", num)

		actor := events.Actor{
			ID:         id,
			Attributes: map[string]string{"image": from},
		}
		e.Log(action, events.ContainerEventType, actor)
	}
	if len(e.events) != eventsLimit {
		t.Fatalf("Must be %d events, got %d", eventsLimit, len(e.events))
	}

	var msgs []events.Message
	for len(msgs) < 10 {
		m := <-l
		jm, ok := (m).(events.Message)
		if !ok {
			t.Fatalf("Unexpected type %T", m)
		}
		msgs = append(msgs, jm)
	}
	if len(current) != eventsLimit {
		t.Fatalf("Must be %d events, got %d", eventsLimit, len(current))
	}
	first := current[0]
	if first.Status != "action_16" {
		t.Fatalf("First action is %s, must be action_16", first.Status)
	}
	last := current[len(current)-1]
	if last.Status != "action_271" {
		t.Fatalf("Last action is %s, must be action_271", last.Status)
	}

	firstC := msgs[0]
	if firstC.Status != "action_272" {
		t.Fatalf("First action is %s, must be action_272", firstC.Status)
	}
	lastC := msgs[len(msgs)-1]
	if lastC.Status != "action_281" {
		t.Fatalf("Last action is %s, must be action_281", lastC.Status)
	}
}

// https://github.com/docker/docker/issues/20999
// Fixtures:
//
//2016-03-07T17:28:03.022433271+02:00 container die 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)
//2016-03-07T17:28:03.091719377+02:00 network disconnect 19c5ed41acb798f26b751e0035cd7821741ab79e2bbd59a66b5fd8abf954eaa0 (type=bridge, container=0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079, name=bridge)
//2016-03-07T17:28:03.129014751+02:00 container destroy 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)
func TestLoadBufferedEvents(t *testing.T) {
	now := time.Now()
	f, err := timetypes.GetTimestamp("2016-03-07T17:28:03.100000000+02:00", now)
	if err != nil {
		t.Fatal(err)
	}
	s, sNano, err := timetypes.ParseTimestamps(f, -1)
	if err != nil {
		t.Fatal(err)
	}

	m1, err := eventstestutils.Scan("2016-03-07T17:28:03.022433271+02:00 container die 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}
	m2, err := eventstestutils.Scan("2016-03-07T17:28:03.091719377+02:00 network disconnect 19c5ed41acb798f26b751e0035cd7821741ab79e2bbd59a66b5fd8abf954eaa0 (type=bridge, container=0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079, name=bridge)")
	if err != nil {
		t.Fatal(err)
	}
	m3, err := eventstestutils.Scan("2016-03-07T17:28:03.129014751+02:00 container destroy 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}

	events := &Events{
		events: []events.Message{*m1, *m2, *m3},
	}

	since := time.Unix(s, sNano)
	until := time.Time{}

	out := events.loadBufferedEvents(since, until, nil)
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d: %v", len(out), out)
	}
}

func TestLoadBufferedEventsOnlyFromPast(t *testing.T) {
	now := time.Now()
	f, err := timetypes.GetTimestamp("2016-03-07T17:28:03.090000000+02:00", now)
	if err != nil {
		t.Fatal(err)
	}
	s, sNano, err := timetypes.ParseTimestamps(f, 0)
	if err != nil {
		t.Fatal(err)
	}

	f, err = timetypes.GetTimestamp("2016-03-07T17:28:03.100000000+02:00", now)
	if err != nil {
		t.Fatal(err)
	}
	u, uNano, err := timetypes.ParseTimestamps(f, 0)
	if err != nil {
		t.Fatal(err)
	}

	m1, err := eventstestutils.Scan("2016-03-07T17:28:03.022433271+02:00 container die 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}
	m2, err := eventstestutils.Scan("2016-03-07T17:28:03.091719377+02:00 network disconnect 19c5ed41acb798f26b751e0035cd7821741ab79e2bbd59a66b5fd8abf954eaa0 (type=bridge, container=0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079, name=bridge)")
	if err != nil {
		t.Fatal(err)
	}
	m3, err := eventstestutils.Scan("2016-03-07T17:28:03.129014751+02:00 container destroy 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}

	events := &Events{
		events: []events.Message{*m1, *m2, *m3},
	}

	since := time.Unix(s, sNano)
	until := time.Unix(u, uNano)

	out := events.loadBufferedEvents(since, until, nil)
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d: %v", len(out), out)
	}

	if out[0].Type != "network" {
		t.Fatalf("expected network event, got %s", out[0].Type)
	}
}

// #13753
func TestIgnoreBufferedWhenNoTimes(t *testing.T) {
	m1, err := eventstestutils.Scan("2016-03-07T17:28:03.022433271+02:00 container die 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}
	m2, err := eventstestutils.Scan("2016-03-07T17:28:03.091719377+02:00 network disconnect 19c5ed41acb798f26b751e0035cd7821741ab79e2bbd59a66b5fd8abf954eaa0 (type=bridge, container=0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079, name=bridge)")
	if err != nil {
		t.Fatal(err)
	}
	m3, err := eventstestutils.Scan("2016-03-07T17:28:03.129014751+02:00 container destroy 0b863f2a26c18557fc6cdadda007c459f9ec81b874780808138aea78a3595079 (image=ubuntu, name=small_hoover)")
	if err != nil {
		t.Fatal(err)
	}

	events := &Events{
		events: []events.Message{*m1, *m2, *m3},
	}

	since := time.Time{}
	until := time.Time{}

	out := events.loadBufferedEvents(since, until, nil)
	if len(out) != 0 {
		t.Fatalf("expected 0 buffered events, got %q", out)
	}
}
