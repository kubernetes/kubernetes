// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestActivityService_ListEvents(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListEvents(opt)
	if err != nil {
		t.Errorf("Activities.ListEvents returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListEvents returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListRepositoryEvents(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListRepositoryEvents("o", "r", opt)
	if err != nil {
		t.Errorf("Activities.ListRepositoryEvents returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListRepositoryEvents returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListRepositoryEvents_invalidOwner(t *testing.T) {
	_, _, err := client.Activity.ListRepositoryEvents("%", "%", nil)
	testURLParseError(t, err)
}

func TestActivityService_ListIssueEventsForRepository(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/issues/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListIssueEventsForRepository("o", "r", opt)
	if err != nil {
		t.Errorf("Activities.ListIssueEventsForRepository returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListIssueEventsForRepository returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListIssueEventsForRepository_invalidOwner(t *testing.T) {
	_, _, err := client.Activity.ListIssueEventsForRepository("%", "%", nil)
	testURLParseError(t, err)
}

func TestActivityService_ListEventsForRepoNetwork(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/networks/o/r/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListEventsForRepoNetwork("o", "r", opt)
	if err != nil {
		t.Errorf("Activities.ListEventsForRepoNetwork returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListEventsForRepoNetwork returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsForRepoNetwork_invalidOwner(t *testing.T) {
	_, _, err := client.Activity.ListEventsForRepoNetwork("%", "%", nil)
	testURLParseError(t, err)
}

func TestActivityService_ListEventsForOrganization(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListEventsForOrganization("o", opt)
	if err != nil {
		t.Errorf("Activities.ListEventsForOrganization returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListEventsForOrganization returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsForOrganization_invalidOrg(t *testing.T) {
	_, _, err := client.Activity.ListEventsForOrganization("%", nil)
	testURLParseError(t, err)
}

func TestActivityService_ListEventsPerformedByUser_all(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListEventsPerformedByUser("u", false, opt)
	if err != nil {
		t.Errorf("Events.ListPerformedByUser returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Events.ListPerformedByUser returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsPerformedByUser_publicOnly(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/events/public", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	events, _, err := client.Activity.ListEventsPerformedByUser("u", true, nil)
	if err != nil {
		t.Errorf("Events.ListPerformedByUser returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Events.ListPerformedByUser returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsPerformedByUser_invalidUser(t *testing.T) {
	_, _, err := client.Activity.ListEventsPerformedByUser("%", false, nil)
	testURLParseError(t, err)
}

func TestActivityService_ListEventsRecievedByUser_all(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/received_events", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListEventsRecievedByUser("u", false, opt)
	if err != nil {
		t.Errorf("Events.ListRecievedByUser returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Events.ListRecievedUser returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsRecievedByUser_publicOnly(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/received_events/public", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	events, _, err := client.Activity.ListEventsRecievedByUser("u", true, nil)
	if err != nil {
		t.Errorf("Events.ListRecievedByUser returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Events.ListRecievedByUser returned %+v, want %+v", events, want)
	}
}

func TestActivityService_ListEventsRecievedByUser_invalidUser(t *testing.T) {
	_, _, err := client.Activity.ListEventsRecievedByUser("%", false, nil)
	testURLParseError(t, err)
}

func TestActivityService_ListUserEventsForOrganization(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/events/orgs/o", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})
		fmt.Fprint(w, `[{"id":"1"},{"id":"2"}]`)
	})

	opt := &ListOptions{Page: 2}
	events, _, err := client.Activity.ListUserEventsForOrganization("o", "u", opt)
	if err != nil {
		t.Errorf("Activities.ListUserEventsForOrganization returned error: %v", err)
	}

	want := []Event{{ID: String("1")}, {ID: String("2")}}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("Activities.ListUserEventsForOrganization returned %+v, want %+v", events, want)
	}
}

func TestActivity_EventPayload_typed(t *testing.T) {
	raw := []byte(`{"type": "PushEvent","payload":{"push_id": 1}}`)
	var event *Event
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatalf("Unmarshal Event returned error: %v", err)
	}

	want := &PushEvent{PushID: Int(1)}
	if !reflect.DeepEqual(event.Payload(), want) {
		t.Errorf("Event Payload returned %+v, want %+v", event.Payload(), want)
	}
}

// TestEvent_Payload_untyped checks that unrecognized events are parsed to an
// interface{} value (instead of being discarded or throwing an error), for
// forward compatibility with new event types.
func TestActivity_EventPayload_untyped(t *testing.T) {
	raw := []byte(`{"type": "UnrecognizedEvent","payload":{"field": "val"}}`)
	var event *Event
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatalf("Unmarshal Event returned error: %v", err)
	}

	want := map[string]interface{}{"field": "val"}
	if !reflect.DeepEqual(event.Payload(), want) {
		t.Errorf("Event Payload returned %+v, want %+v", event.Payload(), want)
	}
}
