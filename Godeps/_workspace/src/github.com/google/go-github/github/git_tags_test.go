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

func TestGitService_GetTag(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/tags/s", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")

		fmt.Fprint(w, `{"tag": "t"}`)
	})

	tag, _, err := client.Git.GetTag("o", "r", "s")

	if err != nil {
		t.Errorf("Git.GetTag returned error: %v", err)
	}

	want := &Tag{Tag: String("t")}
	if !reflect.DeepEqual(tag, want) {
		t.Errorf("Git.GetTag returned %+v, want %+v", tag, want)
	}
}

func TestGitService_CreateTag(t *testing.T) {
	setup()
	defer teardown()

	input := &createTagRequest{Tag: String("t"), Object: String("s")}

	mux.HandleFunc("/repos/o/r/git/tags", func(w http.ResponseWriter, r *http.Request) {
		v := new(createTagRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"tag": "t"}`)
	})

	tag, _, err := client.Git.CreateTag("o", "r", &Tag{
		Tag:    input.Tag,
		Object: &GitObject{SHA: input.Object},
	})
	if err != nil {
		t.Errorf("Git.CreateTag returned error: %v", err)
	}

	want := &Tag{Tag: String("t")}
	if !reflect.DeepEqual(tag, want) {
		t.Errorf("Git.GetTag returned %+v, want %+v", tag, want)
	}
}
