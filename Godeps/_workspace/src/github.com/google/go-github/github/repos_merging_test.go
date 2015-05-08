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

func TestRepositoriesService_Merge(t *testing.T) {
	setup()
	defer teardown()

	input := &RepositoryMergeRequest{
		Base:          String("b"),
		Head:          String("h"),
		CommitMessage: String("c"),
	}

	mux.HandleFunc("/repos/o/r/merges", func(w http.ResponseWriter, r *http.Request) {
		v := new(RepositoryMergeRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"sha":"s"}`)
	})

	commit, _, err := client.Repositories.Merge("o", "r", input)
	if err != nil {
		t.Errorf("Repositories.Merge returned error: %v", err)
	}

	want := &RepositoryCommit{SHA: String("s")}
	if !reflect.DeepEqual(commit, want) {
		t.Errorf("Repositories.Merge returned %+v, want %+v", commit, want)
	}
}
