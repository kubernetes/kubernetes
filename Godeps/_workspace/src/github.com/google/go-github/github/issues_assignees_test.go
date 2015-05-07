// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestIssuesService_ListAssignees(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/assignees", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	assignees, _, err := client.Issues.ListAssignees("o", "r", opt)
	if err != nil {
		t.Errorf("Issues.List returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(assignees, want) {
		t.Errorf("Issues.ListAssignees returned %+v, want %+v", assignees, want)
	}
}

func TestIssuesService_ListAssignees_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.ListAssignees("%", "r", nil)
	testURLParseError(t, err)
}

func TestIssuesService_IsAssignee_true(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/assignees/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
	})

	assignee, _, err := client.Issues.IsAssignee("o", "r", "u")
	if err != nil {
		t.Errorf("Issues.IsAssignee returned error: %v", err)
	}
	if want := true; assignee != want {
		t.Errorf("Issues.IsAssignee returned %+v, want %+v", assignee, want)
	}
}

func TestIssuesService_IsAssignee_false(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/assignees/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	assignee, _, err := client.Issues.IsAssignee("o", "r", "u")
	if err != nil {
		t.Errorf("Issues.IsAssignee returned error: %v", err)
	}
	if want := false; assignee != want {
		t.Errorf("Issues.IsAssignee returned %+v, want %+v", assignee, want)
	}
}

func TestIssuesService_IsAssignee_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/assignees/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	assignee, _, err := client.Issues.IsAssignee("o", "r", "u")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; assignee != want {
		t.Errorf("Issues.IsAssignee returned %+v, want %+v", assignee, want)
	}
}

func TestIssuesService_IsAssignee_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.IsAssignee("%", "r", "u")
	testURLParseError(t, err)
}
