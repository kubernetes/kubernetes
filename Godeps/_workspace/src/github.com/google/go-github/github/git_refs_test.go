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

func TestGitService_GetRef(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/refs/heads/b", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `
		  {
		    "ref": "refs/heads/b",
		    "url": "https://api.github.com/repos/o/r/git/refs/heads/b",
		    "object": {
		      "type": "commit",
		      "sha": "aa218f56b14c9653891f9e74264a383fa43fefbd",
		      "url": "https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"
		    }
		  }`)
	})

	ref, _, err := client.Git.GetRef("o", "r", "refs/heads/b")
	if err != nil {
		t.Errorf("Git.GetRef returned error: %v", err)
	}

	want := &Reference{
		Ref: String("refs/heads/b"),
		URL: String("https://api.github.com/repos/o/r/git/refs/heads/b"),
		Object: &GitObject{
			Type: String("commit"),
			SHA:  String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
			URL:  String("https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"),
		},
	}
	if !reflect.DeepEqual(ref, want) {
		t.Errorf("Git.GetRef returned %+v, want %+v", ref, want)
	}

	// without 'refs/' prefix
	if _, _, err := client.Git.GetRef("o", "r", "heads/b"); err != nil {
		t.Errorf("Git.GetRef returned error: %v", err)
	}
}

func TestGitService_ListRefs(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/refs", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `
		  [
		    {
		      "ref": "refs/heads/branchA",
		      "url": "https://api.github.com/repos/o/r/git/refs/heads/branchA",
		      "object": {
			"type": "commit",
			"sha": "aa218f56b14c9653891f9e74264a383fa43fefbd",
			"url": "https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"
		      }
		    },
		    {
		      "ref": "refs/heads/branchB",
		      "url": "https://api.github.com/repos/o/r/git/refs/heads/branchB",
		      "object": {
			"type": "commit",
			"sha": "aa218f56b14c9653891f9e74264a383fa43fefbd",
			"url": "https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"
		      }
		    }
		  ]`)
	})

	refs, _, err := client.Git.ListRefs("o", "r", nil)
	if err != nil {
		t.Errorf("Git.ListRefs returned error: %v", err)
	}

	want := []Reference{
		{
			Ref: String("refs/heads/branchA"),
			URL: String("https://api.github.com/repos/o/r/git/refs/heads/branchA"),
			Object: &GitObject{
				Type: String("commit"),
				SHA:  String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
				URL:  String("https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"),
			},
		},
		{
			Ref: String("refs/heads/branchB"),
			URL: String("https://api.github.com/repos/o/r/git/refs/heads/branchB"),
			Object: &GitObject{
				Type: String("commit"),
				SHA:  String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
				URL:  String("https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"),
			},
		},
	}
	if !reflect.DeepEqual(refs, want) {
		t.Errorf("Git.ListRefs returned %+v, want %+v", refs, want)
	}
}

func TestGitService_ListRefs_options(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/refs/t", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"ref": "r"}]`)
	})

	opt := &ReferenceListOptions{Type: "t", ListOptions: ListOptions{Page: 2}}
	refs, _, err := client.Git.ListRefs("o", "r", opt)
	if err != nil {
		t.Errorf("Git.ListRefs returned error: %v", err)
	}

	want := []Reference{{Ref: String("r")}}
	if !reflect.DeepEqual(refs, want) {
		t.Errorf("Git.ListRefs returned %+v, want %+v", refs, want)
	}
}

func TestGitService_CreateRef(t *testing.T) {
	setup()
	defer teardown()

	args := &createRefRequest{
		Ref: String("refs/heads/b"),
		SHA: String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
	}

	mux.HandleFunc("/repos/o/r/git/refs", func(w http.ResponseWriter, r *http.Request) {
		v := new(createRefRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, args) {
			t.Errorf("Request body = %+v, want %+v", v, args)
		}
		fmt.Fprint(w, `
		  {
		    "ref": "refs/heads/b",
		    "url": "https://api.github.com/repos/o/r/git/refs/heads/b",
		    "object": {
		      "type": "commit",
		      "sha": "aa218f56b14c9653891f9e74264a383fa43fefbd",
		      "url": "https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"
		    }
		  }`)
	})

	ref, _, err := client.Git.CreateRef("o", "r", &Reference{
		Ref: String("refs/heads/b"),
		Object: &GitObject{
			SHA: String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
		},
	})
	if err != nil {
		t.Errorf("Git.CreateRef returned error: %v", err)
	}

	want := &Reference{
		Ref: String("refs/heads/b"),
		URL: String("https://api.github.com/repos/o/r/git/refs/heads/b"),
		Object: &GitObject{
			Type: String("commit"),
			SHA:  String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
			URL:  String("https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"),
		},
	}
	if !reflect.DeepEqual(ref, want) {
		t.Errorf("Git.CreateRef returned %+v, want %+v", ref, want)
	}

	// without 'refs/' prefix
	_, _, err = client.Git.CreateRef("o", "r", &Reference{
		Ref: String("heads/b"),
		Object: &GitObject{
			SHA: String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
		},
	})
	if err != nil {
		t.Errorf("Git.CreateRef returned error: %v", err)
	}
}

func TestGitService_UpdateRef(t *testing.T) {
	setup()
	defer teardown()

	args := &updateRefRequest{
		SHA:   String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
		Force: Bool(true),
	}

	mux.HandleFunc("/repos/o/r/git/refs/heads/b", func(w http.ResponseWriter, r *http.Request) {
		v := new(updateRefRequest)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, args) {
			t.Errorf("Request body = %+v, want %+v", v, args)
		}
		fmt.Fprint(w, `
		  {
		    "ref": "refs/heads/b",
		    "url": "https://api.github.com/repos/o/r/git/refs/heads/b",
		    "object": {
		      "type": "commit",
		      "sha": "aa218f56b14c9653891f9e74264a383fa43fefbd",
		      "url": "https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"
		    }
		  }`)
	})

	ref, _, err := client.Git.UpdateRef("o", "r", &Reference{
		Ref:    String("refs/heads/b"),
		Object: &GitObject{SHA: String("aa218f56b14c9653891f9e74264a383fa43fefbd")},
	}, true)
	if err != nil {
		t.Errorf("Git.UpdateRef returned error: %v", err)
	}

	want := &Reference{
		Ref: String("refs/heads/b"),
		URL: String("https://api.github.com/repos/o/r/git/refs/heads/b"),
		Object: &GitObject{
			Type: String("commit"),
			SHA:  String("aa218f56b14c9653891f9e74264a383fa43fefbd"),
			URL:  String("https://api.github.com/repos/o/r/git/commits/aa218f56b14c9653891f9e74264a383fa43fefbd"),
		},
	}
	if !reflect.DeepEqual(ref, want) {
		t.Errorf("Git.UpdateRef returned %+v, want %+v", ref, want)
	}

	// without 'refs/' prefix
	_, _, err = client.Git.UpdateRef("o", "r", &Reference{
		Ref:    String("heads/b"),
		Object: &GitObject{SHA: String("aa218f56b14c9653891f9e74264a383fa43fefbd")},
	}, true)
	if err != nil {
		t.Errorf("Git.UpdateRef returned error: %v", err)
	}
}

func TestGitService_DeleteRef(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/refs/heads/b", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Git.DeleteRef("o", "r", "refs/heads/b")
	if err != nil {
		t.Errorf("Git.DeleteRef returned error: %v", err)
	}

	// without 'refs/' prefix
	if _, err := client.Git.DeleteRef("o", "r", "heads/b"); err != nil {
		t.Errorf("Git.DeleteRef returned error: %v", err)
	}
}
