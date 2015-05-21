// Copyright 2014 The go-github AUTHORS. All rights reserved.
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
	"time"
)

func TestActivityService_ListNotification(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"all":           "true",
			"participating": "true",
			"since":         "2006-01-02T15:04:05Z",
		})

		fmt.Fprint(w, `[{"id":"1", "subject":{"title":"t"}}]`)
	})

	opt := &NotificationListOptions{
		All:           true,
		Participating: true,
		Since:         time.Date(2006, 01, 02, 15, 04, 05, 0, time.UTC),
	}
	notifications, _, err := client.Activity.ListNotifications(opt)
	if err != nil {
		t.Errorf("Activity.ListNotifications returned error: %v", err)
	}

	want := []Notification{{ID: String("1"), Subject: &NotificationSubject{Title: String("t")}}}
	if !reflect.DeepEqual(notifications, want) {
		t.Errorf("Activity.ListNotifications returned %+v, want %+v", notifications, want)
	}
}

func TestActivityService_ListRepositoryNotification(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/notifications", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":"1"}]`)
	})

	notifications, _, err := client.Activity.ListRepositoryNotifications("o", "r", nil)
	if err != nil {
		t.Errorf("Activity.ListRepositoryNotifications returned error: %v", err)
	}

	want := []Notification{{ID: String("1")}}
	if !reflect.DeepEqual(notifications, want) {
		t.Errorf("Activity.ListRepositoryNotifications returned %+v, want %+v", notifications, want)
	}
}

func TestActivityService_MarkNotificationsRead(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		testFormValues(t, r, values{
			"last_read_at": "2006-01-02T15:04:05Z",
		})

		w.WriteHeader(http.StatusResetContent)
	})

	_, err := client.Activity.MarkNotificationsRead(time.Date(2006, 01, 02, 15, 04, 05, 0, time.UTC))
	if err != nil {
		t.Errorf("Activity.MarkNotificationsRead returned error: %v", err)
	}
}

func TestActivityService_MarkRepositoryNotificationsRead(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/notifications", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		testFormValues(t, r, values{
			"last_read_at": "2006-01-02T15:04:05Z",
		})

		w.WriteHeader(http.StatusResetContent)
	})

	_, err := client.Activity.MarkRepositoryNotificationsRead("o", "r", time.Date(2006, 01, 02, 15, 04, 05, 0, time.UTC))
	if err != nil {
		t.Errorf("Activity.MarkRepositoryNotificationsRead returned error: %v", err)
	}
}

func TestActivityService_GetThread(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications/threads/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":"1"}`)
	})

	notification, _, err := client.Activity.GetThread("1")
	if err != nil {
		t.Errorf("Activity.GetThread returned error: %v", err)
	}

	want := &Notification{ID: String("1")}
	if !reflect.DeepEqual(notification, want) {
		t.Errorf("Activity.GetThread returned %+v, want %+v", notification, want)
	}
}

func TestActivityService_MarkThreadRead(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications/threads/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PATCH")
		w.WriteHeader(http.StatusResetContent)
	})

	_, err := client.Activity.MarkThreadRead("1")
	if err != nil {
		t.Errorf("Activity.MarkThreadRead returned error: %v", err)
	}
}

func TestActivityService_GetThreadSubscription(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications/threads/1/subscription", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"subscribed":true}`)
	})

	sub, _, err := client.Activity.GetThreadSubscription("1")
	if err != nil {
		t.Errorf("Activity.GetThreadSubscription returned error: %v", err)
	}

	want := &Subscription{Subscribed: Bool(true)}
	if !reflect.DeepEqual(sub, want) {
		t.Errorf("Activity.GetThreadSubscription returned %+v, want %+v", sub, want)
	}
}

func TestActivityService_SetThreadSubscription(t *testing.T) {
	setup()
	defer teardown()

	input := &Subscription{Subscribed: Bool(true)}

	mux.HandleFunc("/notifications/threads/1/subscription", func(w http.ResponseWriter, r *http.Request) {
		v := new(Subscription)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PUT")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"ignored":true}`)
	})

	sub, _, err := client.Activity.SetThreadSubscription("1", input)
	if err != nil {
		t.Errorf("Activity.SetThreadSubscription returned error: %v", err)
	}

	want := &Subscription{Ignored: Bool(true)}
	if !reflect.DeepEqual(sub, want) {
		t.Errorf("Activity.SetThreadSubscription returned %+v, want %+v", sub, want)
	}
}

func TestActivityService_DeleteThreadSubscription(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/notifications/threads/1/subscription", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Activity.DeleteThreadSubscription("1")
	if err != nil {
		t.Errorf("Activity.DeleteThreadSubscription returned error: %v", err)
	}
}
