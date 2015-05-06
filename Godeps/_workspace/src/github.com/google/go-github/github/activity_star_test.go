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

func TestActivityService_ListStargazers(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/stargazers", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"page": "2",
		})

		fmt.Fprint(w, `[{"id":1}]`)
	})

	stargazers, _, err := client.Activity.ListStargazers("o", "r", &ListOptions{Page: 2})
	if err != nil {
		t.Errorf("Activity.ListStargazers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(stargazers, want) {
		t.Errorf("Activity.ListStargazers returned %+v, want %+v", stargazers, want)
	}
}

func TestActivityService_ListStarred_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/starred", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	repos, _, err := client.Activity.ListStarred("", nil)
	if err != nil {
		t.Errorf("Activity.ListStarred returned error: %v", err)
	}

	want := []Repository{{ID: Int(1)}}
	if !reflect.DeepEqual(repos, want) {
		t.Errorf("Activity.ListStarred returned %+v, want %+v", repos, want)
	}
}

func TestActivityService_ListStarred_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/starred", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"sort":      "created",
			"direction": "asc",
			"page":      "2",
		})
		fmt.Fprint(w, `[{"id":2}]`)
	})

	opt := &ActivityListStarredOptions{"created", "asc", ListOptions{Page: 2}}
	repos, _, err := client.Activity.ListStarred("u", opt)
	if err != nil {
		t.Errorf("Activity.ListStarred returned error: %v", err)
	}

	want := []Repository{{ID: Int(2)}}
	if !reflect.DeepEqual(repos, want) {
		t.Errorf("Activity.ListStarred returned %+v, want %+v", repos, want)
	}
}

func TestActivityService_ListStarred_invalidUser(t *testing.T) {
	_, _, err := client.Activity.ListStarred("%", nil)
	testURLParseError(t, err)
}

func TestActivityService_IsStarred_hasStar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/starred/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	star, _, err := client.Activity.IsStarred("o", "r")
	if err != nil {
		t.Errorf("Activity.IsStarred returned error: %v", err)
	}
	if want := true; star != want {
		t.Errorf("Activity.IsStarred returned %+v, want %+v", star, want)
	}
}

func TestActivityService_IsStarred_noStar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/starred/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	star, _, err := client.Activity.IsStarred("o", "r")
	if err != nil {
		t.Errorf("Activity.IsStarred returned error: %v", err)
	}
	if want := false; star != want {
		t.Errorf("Activity.IsStarred returned %+v, want %+v", star, want)
	}
}

func TestActivityService_IsStarred_invalidID(t *testing.T) {
	_, _, err := client.Activity.IsStarred("%", "%")
	testURLParseError(t, err)
}

func TestActivityService_Star(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/starred/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
	})

	_, err := client.Activity.Star("o", "r")
	if err != nil {
		t.Errorf("Activity.Star returned error: %v", err)
	}
}

func TestActivityService_Star_invalidID(t *testing.T) {
	_, err := client.Activity.Star("%", "%")
	testURLParseError(t, err)
}

func TestActivityService_Unstar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/starred/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Activity.Unstar("o", "r")
	if err != nil {
		t.Errorf("Activity.Unstar returned error: %v", err)
	}
}

func TestActivityService_Unstar_invalidID(t *testing.T) {
	_, err := client.Activity.Unstar("%", "%")
	testURLParseError(t, err)
}
