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

func TestPullRequestsService_ListComments_allPulls(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/pulls/comments", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"sort":      "updated",
			"direction": "desc",
			"since":     "2002-02-10T15:30:00Z",
			"page":      "2",
		})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &PullRequestListCommentsOptions{
		Sort:        "updated",
		Direction:   "desc",
		Since:       time.Date(2002, time.February, 10, 15, 30, 0, 0, time.UTC),
		ListOptions: ListOptions{Page: 2},
	}
	pulls, _, err := client.PullRequests.ListComments("o", "r", 0, opt)

	if err != nil {
		t.Errorf("PullRequests.ListComments returned error: %v", err)
	}

	want := []PullRequestComment{{ID: Int(1)}}
	if !reflect.DeepEqual(pulls, want) {
		t.Errorf("PullRequests.ListComments returned %+v, want %+v", pulls, want)
	}
}

func TestPullRequestsService_ListComments_specificPull(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/pulls/1/comments", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	pulls, _, err := client.PullRequests.ListComments("o", "r", 1, nil)

	if err != nil {
		t.Errorf("PullRequests.ListComments returned error: %v", err)
	}

	want := []PullRequestComment{{ID: Int(1)}}
	if !reflect.DeepEqual(pulls, want) {
		t.Errorf("PullRequests.ListComments returned %+v, want %+v", pulls, want)
	}
}

func TestPullRequestsService_ListComments_invalidOwner(t *testing.T) {
	_, _, err := client.PullRequests.ListComments("%", "r", 1, nil)
	testURLParseError(t, err)
}

func TestPullRequestsService_GetComment(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/pulls/comments/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	comment, _, err := client.PullRequests.GetComment("o", "r", 1)

	if err != nil {
		t.Errorf("PullRequests.GetComment returned error: %v", err)
	}

	want := &PullRequestComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("PullRequests.GetComment returned %+v, want %+v", comment, want)
	}
}

func TestPullRequestsService_GetComment_invalidOwner(t *testing.T) {
	_, _, err := client.PullRequests.GetComment("%", "r", 1)
	testURLParseError(t, err)
}

func TestPullRequestsService_CreateComment(t *testing.T) {
	setup()
	defer teardown()

	input := &PullRequestComment{Body: String("b")}

	mux.HandleFunc("/repos/o/r/pulls/1/comments", func(w http.ResponseWriter, r *http.Request) {
		v := new(PullRequestComment)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	comment, _, err := client.PullRequests.CreateComment("o", "r", 1, input)

	if err != nil {
		t.Errorf("PullRequests.CreateComment returned error: %v", err)
	}

	want := &PullRequestComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("PullRequests.CreateComment returned %+v, want %+v", comment, want)
	}
}

func TestPullRequestsService_CreateComment_invalidOwner(t *testing.T) {
	_, _, err := client.PullRequests.CreateComment("%", "r", 1, nil)
	testURLParseError(t, err)
}

func TestPullRequestsService_EditComment(t *testing.T) {
	setup()
	defer teardown()

	input := &PullRequestComment{Body: String("b")}

	mux.HandleFunc("/repos/o/r/pulls/comments/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(PullRequestComment)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	comment, _, err := client.PullRequests.EditComment("o", "r", 1, input)

	if err != nil {
		t.Errorf("PullRequests.EditComment returned error: %v", err)
	}

	want := &PullRequestComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("PullRequests.EditComment returned %+v, want %+v", comment, want)
	}
}

func TestPullRequestsService_EditComment_invalidOwner(t *testing.T) {
	_, _, err := client.PullRequests.EditComment("%", "r", 1, nil)
	testURLParseError(t, err)
}

func TestPullRequestsService_DeleteComment(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/pulls/comments/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.PullRequests.DeleteComment("o", "r", 1)
	if err != nil {
		t.Errorf("PullRequests.DeleteComment returned error: %v", err)
	}
}

func TestPullRequestsService_DeleteComment_invalidOwner(t *testing.T) {
	_, err := client.PullRequests.DeleteComment("%", "r", 1)
	testURLParseError(t, err)
}
