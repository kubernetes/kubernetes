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

func TestRepositoriesService_ListCollaborators(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/collaborators", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprintf(w, `[{"id":1}, {"id":2}]`)
	})

	opt := &ListOptions{Page: 2}
	users, _, err := client.Repositories.ListCollaborators("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.ListCollaborators returned error: %v", err)
	}

	want := []User{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Repositories.ListCollaborators returned %+v, want %+v", users, want)
	}
}

func TestRepositoriesService_ListCollaborators_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.ListCollaborators("%", "%", nil)
	testURLParseError(t, err)
}

func TestRepositoriesService_IsCollaborator_True(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/collaborators/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	isCollab, _, err := client.Repositories.IsCollaborator("o", "r", "u")
	if err != nil {
		t.Errorf("Repositories.IsCollaborator returned error: %v", err)
	}

	if !isCollab {
		t.Errorf("Repositories.IsCollaborator returned false, want true")
	}
}

func TestRepositoriesService_IsCollaborator_False(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/collaborators/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	isCollab, _, err := client.Repositories.IsCollaborator("o", "r", "u")
	if err != nil {
		t.Errorf("Repositories.IsCollaborator returned error: %v", err)
	}

	if isCollab {
		t.Errorf("Repositories.IsCollaborator returned true, want false")
	}
}

func TestRepositoriesService_IsCollaborator_invalidUser(t *testing.T) {
	_, _, err := client.Repositories.IsCollaborator("%", "%", "%")
	testURLParseError(t, err)
}

func TestRepositoriesService_AddCollaborator(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/collaborators/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Repositories.AddCollaborator("o", "r", "u")
	if err != nil {
		t.Errorf("Repositories.AddCollaborator returned error: %v", err)
	}
}

func TestRepositoriesService_AddCollaborator_invalidUser(t *testing.T) {
	_, err := client.Repositories.AddCollaborator("%", "%", "%")
	testURLParseError(t, err)
}

func TestRepositoriesService_RemoveCollaborator(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/collaborators/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Repositories.RemoveCollaborator("o", "r", "u")
	if err != nil {
		t.Errorf("Repositories.RemoveCollaborator returned error: %v", err)
	}
}

func TestRepositoriesService_RemoveCollaborator_invalidUser(t *testing.T) {
	_, err := client.Repositories.RemoveCollaborator("%", "%", "%")
	testURLParseError(t, err)
}
