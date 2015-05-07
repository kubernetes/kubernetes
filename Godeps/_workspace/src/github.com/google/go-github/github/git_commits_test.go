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

func TestGitService_GetCommit(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/commits/s", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"sha":"s","message":"m","author":{"name":"n"}}`)
	})

	commit, _, err := client.Git.GetCommit("o", "r", "s")
	if err != nil {
		t.Errorf("Git.GetCommit returned error: %v", err)
	}

	want := &Commit{SHA: String("s"), Message: String("m"), Author: &CommitAuthor{Name: String("n")}}
	if !reflect.DeepEqual(commit, want) {
		t.Errorf("Git.GetCommit returned %+v, want %+v", commit, want)
	}
}

func TestGitService_GetCommit_invalidOwner(t *testing.T) {
	_, _, err := client.Git.GetCommit("%", "%", "%")
	testURLParseError(t, err)
}

func TestGitService_CreateCommit(t *testing.T) {
	setup()
	defer teardown()

	input := &Commit{
		Message: String("m"),
		Tree:    &Tree{SHA: String("t")},
		Parents: []Commit{{SHA: String("p")}},
	}

	mux.HandleFunc("/repos/o/r/git/commits", func(w http.ResponseWriter, r *http.Request) {
		v := new(createCommit)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")

		want := &createCommit{
			Message: input.Message,
			Tree:    String("t"),
			Parents: []string{"p"},
		}
		if !reflect.DeepEqual(v, want) {
			t.Errorf("Request body = %+v, want %+v", v, want)
		}
		fmt.Fprint(w, `{"sha":"s"}`)
	})

	commit, _, err := client.Git.CreateCommit("o", "r", input)
	if err != nil {
		t.Errorf("Git.CreateCommit returned error: %v", err)
	}

	want := &Commit{SHA: String("s")}
	if !reflect.DeepEqual(commit, want) {
		t.Errorf("Git.CreateCommit returned %+v, want %+v", commit, want)
	}
}

func TestGitService_CreateCommit_invalidOwner(t *testing.T) {
	_, _, err := client.Git.CreateCommit("%", "%", nil)
	testURLParseError(t, err)
}
