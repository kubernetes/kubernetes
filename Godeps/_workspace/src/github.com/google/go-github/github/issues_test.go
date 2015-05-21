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
	"time"
)

func TestIssuesService_List_all(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/issues", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"filter":    "all",
			"state":     "closed",
			"labels":    "a,b",
			"sort":      "updated",
			"direction": "asc",
			"since":     "2002-02-10T15:30:00Z",
			"page":      "1",
			"per_page":  "2",
		})
		fmt.Fprint(w, `[{"number":1}]`)
	})

	opt := &IssueListOptions{
		"all", "closed", []string{"a", "b"}, "updated", "asc",
		time.Date(2002, time.February, 10, 15, 30, 0, 0, time.UTC),
		ListOptions{Page: 1, PerPage: 2},
	}
	issues, _, err := client.Issues.List(true, opt)

	if err != nil {
		t.Errorf("Issues.List returned error: %v", err)
	}

	want := []Issue{{Number: Int(1)}}
	if !reflect.DeepEqual(issues, want) {
		t.Errorf("Issues.List returned %+v, want %+v", issues, want)
	}
}

func TestIssuesService_List_owned(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/issues", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"number":1}]`)
	})

	issues, _, err := client.Issues.List(false, nil)
	if err != nil {
		t.Errorf("Issues.List returned error: %v", err)
	}

	want := []Issue{{Number: Int(1)}}
	if !reflect.DeepEqual(issues, want) {
		t.Errorf("Issues.List returned %+v, want %+v", issues, want)
	}
}

func TestIssuesService_ListByOrg(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/issues", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"number":1}]`)
	})

	issues, _, err := client.Issues.ListByOrg("o", nil)
	if err != nil {
		t.Errorf("Issues.ListByOrg returned error: %v", err)
	}

	want := []Issue{{Number: Int(1)}}
	if !reflect.DeepEqual(issues, want) {
		t.Errorf("Issues.List returned %+v, want %+v", issues, want)
	}
}

func TestIssuesService_ListByOrg_invalidOrg(t *testing.T) {
	_, _, err := client.Issues.ListByOrg("%", nil)
	testURLParseError(t, err)
}

func TestIssuesService_ListByRepo(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/issues", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"milestone": "*",
			"state":     "closed",
			"assignee":  "a",
			"creator":   "c",
			"mentioned": "m",
			"labels":    "a,b",
			"sort":      "updated",
			"direction": "asc",
			"since":     "2002-02-10T15:30:00Z",
		})
		fmt.Fprint(w, `[{"number":1}]`)
	})

	opt := &IssueListByRepoOptions{
		"*", "closed", "a", "c", "m", []string{"a", "b"}, "updated", "asc",
		time.Date(2002, time.February, 10, 15, 30, 0, 0, time.UTC),
		ListOptions{0, 0},
	}
	issues, _, err := client.Issues.ListByRepo("o", "r", opt)
	if err != nil {
		t.Errorf("Issues.ListByOrg returned error: %v", err)
	}

	want := []Issue{{Number: Int(1)}}
	if !reflect.DeepEqual(issues, want) {
		t.Errorf("Issues.List returned %+v, want %+v", issues, want)
	}
}

func TestIssuesService_ListByRepo_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.ListByRepo("%", "r", nil)
	testURLParseError(t, err)
}

func TestIssuesService_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/issues/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"number":1, "labels": [{"url": "u", "name": "n", "color": "c"}]}`)
	})

	issue, _, err := client.Issues.Get("o", "r", 1)
	if err != nil {
		t.Errorf("Issues.Get returned error: %v", err)
	}

	want := &Issue{
		Number: Int(1),
		Labels: []Label{{
			URL:   String("u"),
			Name:  String("n"),
			Color: String("c"),
		}},
	}
	if !reflect.DeepEqual(issue, want) {
		t.Errorf("Issues.Get returned %+v, want %+v", issue, want)
	}
}

func TestIssuesService_Get_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.Get("%", "r", 1)
	testURLParseError(t, err)
}

func TestIssuesService_Create(t *testing.T) {
	setup()
	defer teardown()

	input := &IssueRequest{
		Title:    String("t"),
		Body:     String("b"),
		Assignee: String("a"),
		Labels:   &[]string{"l1", "l2"},
	}

	mux.HandleFunc("/repos/o/r/issues", func(w http.ResponseWriter, r *http.Request) {
		v := new(IssueRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"number":1}`)
	})

	issue, _, err := client.Issues.Create("o", "r", input)
	if err != nil {
		t.Errorf("Issues.Create returned error: %v", err)
	}

	want := &Issue{Number: Int(1)}
	if !reflect.DeepEqual(issue, want) {
		t.Errorf("Issues.Create returned %+v, want %+v", issue, want)
	}
}

func TestIssuesService_Create_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.Create("%", "r", nil)
	testURLParseError(t, err)
}

func TestIssuesService_Edit(t *testing.T) {
	setup()
	defer teardown()

	input := &IssueRequest{Title: String("t")}

	mux.HandleFunc("/repos/o/r/issues/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(IssueRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"number":1}`)
	})

	issue, _, err := client.Issues.Edit("o", "r", 1, input)
	if err != nil {
		t.Errorf("Issues.Edit returned error: %v", err)
	}

	want := &Issue{Number: Int(1)}
	if !reflect.DeepEqual(issue, want) {
		t.Errorf("Issues.Edit returned %+v, want %+v", issue, want)
	}
}

func TestIssuesService_Edit_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.Edit("%", "r", 1, nil)
	testURLParseError(t, err)
}
