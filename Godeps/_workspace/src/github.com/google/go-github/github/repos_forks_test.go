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

func TestRepositoriesService_ListForks(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/forks", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"sort": "newest",
			"page": "3",
		})
		fmt.Fprint(w, `[{"id":1},{"id":2}]`)
	})

	opt := &RepositoryListForksOptions{
		Sort:        "newest",
		ListOptions: ListOptions{Page: 3},
	}
	repos, _, err := client.Repositories.ListForks("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.ListForks returned error: %v", err)
	}

	want := []Repository{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(repos, want) {
		t.Errorf("Repositories.ListForks returned %+v, want %+v", repos, want)
	}
}

func TestRepositoriesService_ListForks_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.ListForks("%", "r", nil)
	testURLParseError(t, err)
}

func TestRepositoriesService_CreateFork(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/forks", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
		testFormValues(t, r, values{"organization": "o"})
		fmt.Fprint(w, `{"id":1}`)
	})

	opt := &RepositoryCreateForkOptions{Organization: "o"}
	repo, _, err := client.Repositories.CreateFork("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.CreateFork returned error: %v", err)
	}

	want := &Repository{ID: Int(1)}
	if !reflect.DeepEqual(repo, want) {
		t.Errorf("Repositories.CreateFork returned %+v, want %+v", repo, want)
	}
}

func TestRepositoriesService_CreateFork_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.CreateFork("%", "r", nil)
	testURLParseError(t, err)
}
