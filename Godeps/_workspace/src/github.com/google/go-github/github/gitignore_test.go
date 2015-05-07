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

func TestGitignoresService_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gitignore/templates", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `["C", "Go"]`)
	})

	available, _, err := client.Gitignores.List()
	if err != nil {
		t.Errorf("Gitignores.List returned error: %v", err)
	}

	want := []string{"C", "Go"}
	if !reflect.DeepEqual(available, want) {
		t.Errorf("Gitignores.List returned %+v, want %+v", available, want)
	}
}

func TestGitignoresService_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gitignore/templates/name", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"name":"Name","source":"template source"}`)
	})

	gitignore, _, err := client.Gitignores.Get("name")
	if err != nil {
		t.Errorf("Gitignores.List returned error: %v", err)
	}

	want := &Gitignore{Name: String("Name"), Source: String("template source")}
	if !reflect.DeepEqual(gitignore, want) {
		t.Errorf("Gitignores.Get returned %+v, want %+v", gitignore, want)
	}
}

func TestGitignoresService_Get_invalidTemplate(t *testing.T) {
	_, _, err := client.Gitignores.Get("%")
	testURLParseError(t, err)
}
