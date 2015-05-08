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
)

func TestIssuesService_ListMilestones(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/milestones", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"state":     "closed",
			"sort":      "due_date",
			"direction": "asc",
		})
		fmt.Fprint(w, `[{"number":1}]`)
	})

	opt := &MilestoneListOptions{"closed", "due_date", "asc"}
	milestones, _, err := client.Issues.ListMilestones("o", "r", opt)
	if err != nil {
		t.Errorf("IssuesService.ListMilestones returned error: %v", err)
	}

	want := []Milestone{{Number: Int(1)}}
	if !reflect.DeepEqual(milestones, want) {
		t.Errorf("IssuesService.ListMilestones returned %+v, want %+v", milestones, want)
	}
}

func TestIssuesService_ListMilestones_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.ListMilestones("%", "r", nil)
	testURLParseError(t, err)
}

func TestIssuesService_GetMilestone(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/milestones/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"number":1}`)
	})

	milestone, _, err := client.Issues.GetMilestone("o", "r", 1)
	if err != nil {
		t.Errorf("IssuesService.GetMilestone returned error: %v", err)
	}

	want := &Milestone{Number: Int(1)}
	if !reflect.DeepEqual(milestone, want) {
		t.Errorf("IssuesService.GetMilestone returned %+v, want %+v", milestone, want)
	}
}

func TestIssuesService_GetMilestone_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.GetMilestone("%", "r", 1)
	testURLParseError(t, err)
}

func TestIssuesService_CreateMilestone(t *testing.T) {
	setup()
	defer teardown()

	input := &Milestone{Title: String("t")}

	mux.HandleFunc("/repos/o/r/milestones", func(w http.ResponseWriter, r *http.Request) {
		v := new(Milestone)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"number":1}`)
	})

	milestone, _, err := client.Issues.CreateMilestone("o", "r", input)
	if err != nil {
		t.Errorf("IssuesService.CreateMilestone returned error: %v", err)
	}

	want := &Milestone{Number: Int(1)}
	if !reflect.DeepEqual(milestone, want) {
		t.Errorf("IssuesService.CreateMilestone returned %+v, want %+v", milestone, want)
	}
}

func TestIssuesService_CreateMilestone_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.CreateMilestone("%", "r", nil)
	testURLParseError(t, err)
}

func TestIssuesService_EditMilestone(t *testing.T) {
	setup()
	defer teardown()

	input := &Milestone{Title: String("t")}

	mux.HandleFunc("/repos/o/r/milestones/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(Milestone)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"number":1}`)
	})

	milestone, _, err := client.Issues.EditMilestone("o", "r", 1, input)
	if err != nil {
		t.Errorf("IssuesService.EditMilestone returned error: %v", err)
	}

	want := &Milestone{Number: Int(1)}
	if !reflect.DeepEqual(milestone, want) {
		t.Errorf("IssuesService.EditMilestone returned %+v, want %+v", milestone, want)
	}
}

func TestIssuesService_EditMilestone_invalidOwner(t *testing.T) {
	_, _, err := client.Issues.EditMilestone("%", "r", 1, nil)
	testURLParseError(t, err)
}

func TestIssuesService_DeleteMilestone(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/milestones/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Issues.DeleteMilestone("o", "r", 1)
	if err != nil {
		t.Errorf("IssuesService.DeleteMilestone returned error: %v", err)
	}
}

func TestIssuesService_DeleteMilestone_invalidOwner(t *testing.T) {
	_, err := client.Issues.DeleteMilestone("%", "r", 1)
	testURLParseError(t, err)
}
