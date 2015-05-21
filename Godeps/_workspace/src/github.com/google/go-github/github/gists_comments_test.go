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

func TestGistsService_ListComments(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/comments", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id": 1}]`)
	})

	opt := &ListOptions{Page: 2}
	comments, _, err := client.Gists.ListComments("1", opt)

	if err != nil {
		t.Errorf("Gists.Comments returned error: %v", err)
	}

	want := []GistComment{{ID: Int(1)}}
	if !reflect.DeepEqual(comments, want) {
		t.Errorf("Gists.ListComments returned %+v, want %+v", comments, want)
	}
}

func TestGistsService_ListComments_invalidID(t *testing.T) {
	_, _, err := client.Gists.ListComments("%", nil)
	testURLParseError(t, err)
}

func TestGistsService_GetComment(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/comments/2", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id": 1}`)
	})

	comment, _, err := client.Gists.GetComment("1", 2)

	if err != nil {
		t.Errorf("Gists.GetComment returned error: %v", err)
	}

	want := &GistComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("Gists.GetComment returned %+v, want %+v", comment, want)
	}
}

func TestGistsService_GetComment_invalidID(t *testing.T) {
	_, _, err := client.Gists.GetComment("%", 1)
	testURLParseError(t, err)
}

func TestGistsService_CreateComment(t *testing.T) {
	setup()
	defer teardown()

	input := &GistComment{ID: Int(1), Body: String("b")}

	mux.HandleFunc("/gists/1/comments", func(w http.ResponseWriter, r *http.Request) {
		v := new(GistComment)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	comment, _, err := client.Gists.CreateComment("1", input)
	if err != nil {
		t.Errorf("Gists.CreateComment returned error: %v", err)
	}

	want := &GistComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("Gists.CreateComment returned %+v, want %+v", comment, want)
	}
}

func TestGistsService_CreateComment_invalidID(t *testing.T) {
	_, _, err := client.Gists.CreateComment("%", nil)
	testURLParseError(t, err)
}

func TestGistsService_EditComment(t *testing.T) {
	setup()
	defer teardown()

	input := &GistComment{ID: Int(1), Body: String("b")}

	mux.HandleFunc("/gists/1/comments/2", func(w http.ResponseWriter, r *http.Request) {
		v := new(GistComment)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	comment, _, err := client.Gists.EditComment("1", 2, input)
	if err != nil {
		t.Errorf("Gists.EditComment returned error: %v", err)
	}

	want := &GistComment{ID: Int(1)}
	if !reflect.DeepEqual(comment, want) {
		t.Errorf("Gists.EditComment returned %+v, want %+v", comment, want)
	}
}

func TestGistsService_EditComment_invalidID(t *testing.T) {
	_, _, err := client.Gists.EditComment("%", 1, nil)
	testURLParseError(t, err)
}

func TestGistsService_DeleteComment(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/comments/2", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Gists.DeleteComment("1", 2)
	if err != nil {
		t.Errorf("Gists.Delete returned error: %v", err)
	}
}

func TestGistsService_DeleteComment_invalidID(t *testing.T) {
	_, err := client.Gists.DeleteComment("%", 1)
	testURLParseError(t, err)
}
