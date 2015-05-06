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

func TestGitService_GetTree(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/trees/s", func(w http.ResponseWriter, r *http.Request) {
		if m := "GET"; m != r.Method {
			t.Errorf("Request method = %v, want %v", r.Method, m)
		}
		fmt.Fprint(w, `{
			  "sha": "s",
			  "tree": [ { "type": "blob" } ]
			}`)
	})

	tree, _, err := client.Git.GetTree("o", "r", "s", true)
	if err != nil {
		t.Errorf("Git.GetTree returned error: %v", err)
	}

	want := Tree{
		SHA: String("s"),
		Entries: []TreeEntry{
			{
				Type: String("blob"),
			},
		},
	}
	if !reflect.DeepEqual(*tree, want) {
		t.Errorf("Tree.Get returned %+v, want %+v", *tree, want)
	}
}

func TestGitService_GetTree_invalidOwner(t *testing.T) {
	_, _, err := client.Git.GetTree("%", "%", "%", false)
	testURLParseError(t, err)
}

func TestGitService_CreateTree(t *testing.T) {
	setup()
	defer teardown()

	input := []TreeEntry{
		{
			Path: String("file.rb"),
			Mode: String("100644"),
			Type: String("blob"),
			SHA:  String("7c258a9869f33c1e1e1f74fbb32f07c86cb5a75b"),
		},
	}

	mux.HandleFunc("/repos/o/r/git/trees", func(w http.ResponseWriter, r *http.Request) {
		v := new(createTree)
		json.NewDecoder(r.Body).Decode(v)

		if m := "POST"; m != r.Method {
			t.Errorf("Request method = %v, want %v", r.Method, m)
		}

		want := &createTree{
			BaseTree: "b",
			Entries:  input,
		}
		if !reflect.DeepEqual(v, want) {
			t.Errorf("Git.CreateTree request body: %+v, want %+v", v, want)
		}

		fmt.Fprint(w, `{
		  "sha": "cd8274d15fa3ae2ab983129fb037999f264ba9a7",
		  "tree": [
		    {
		      "path": "file.rb",
		      "mode": "100644",
		      "type": "blob",
		      "size": 132,
		      "sha": "7c258a9869f33c1e1e1f74fbb32f07c86cb5a75b"
		    }
		  ]
		}`)
	})

	tree, _, err := client.Git.CreateTree("o", "r", "b", input)
	if err != nil {
		t.Errorf("Git.CreateTree returned error: %v", err)
	}

	want := Tree{
		String("cd8274d15fa3ae2ab983129fb037999f264ba9a7"),
		[]TreeEntry{
			{
				Path: String("file.rb"),
				Mode: String("100644"),
				Type: String("blob"),
				Size: Int(132),
				SHA:  String("7c258a9869f33c1e1e1f74fbb32f07c86cb5a75b"),
			},
		},
	}

	if !reflect.DeepEqual(*tree, want) {
		t.Errorf("Git.CreateTree returned %+v, want %+v", *tree, want)
	}
}

func TestGitService_CreateTree_Content(t *testing.T) {
	setup()
	defer teardown()

	input := []TreeEntry{
		{
			Path:    String("content.md"),
			Mode:    String("100644"),
			Content: String("file content"),
		},
	}

	mux.HandleFunc("/repos/o/r/git/trees", func(w http.ResponseWriter, r *http.Request) {
		v := new(createTree)
		json.NewDecoder(r.Body).Decode(v)

		if m := "POST"; m != r.Method {
			t.Errorf("Request method = %v, want %v", r.Method, m)
		}

		want := &createTree{
			BaseTree: "b",
			Entries:  input,
		}
		if !reflect.DeepEqual(v, want) {
			t.Errorf("Git.CreateTree request body: %+v, want %+v", v, want)
		}

		fmt.Fprint(w, `{
		  "sha": "5c6780ad2c68743383b740fd1dab6f6a33202b11",
		  "url": "https://api.github.com/repos/o/r/git/trees/5c6780ad2c68743383b740fd1dab6f6a33202b11",
		  "tree": [
		    {
			  "mode": "100644",
			  "type": "blob",
			  "sha":  "aad8feacf6f8063150476a7b2bd9770f2794c08b",
			  "path": "content.md",
			  "size": 12,
			  "url": "https://api.github.com/repos/o/r/git/blobs/aad8feacf6f8063150476a7b2bd9770f2794c08b"
		    }
		  ]
		}`)
	})

	tree, _, err := client.Git.CreateTree("o", "r", "b", input)
	if err != nil {
		t.Errorf("Git.CreateTree returned error: %v", err)
	}

	want := Tree{
		String("5c6780ad2c68743383b740fd1dab6f6a33202b11"),
		[]TreeEntry{
			{
				Path: String("content.md"),
				Mode: String("100644"),
				Type: String("blob"),
				Size: Int(12),
				SHA:  String("aad8feacf6f8063150476a7b2bd9770f2794c08b"),
			},
		},
	}

	if !reflect.DeepEqual(*tree, want) {
		t.Errorf("Git.CreateTree returned %+v, want %+v", *tree, want)
	}
}

func TestGitService_CreateTree_invalidOwner(t *testing.T) {
	_, _, err := client.Git.CreateTree("%", "%", "", nil)
	testURLParseError(t, err)
}
