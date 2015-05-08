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

func TestRepositoriesService_CreateHook(t *testing.T) {
	setup()
	defer teardown()

	input := &Hook{Name: String("t")}

	mux.HandleFunc("/repos/o/r/hooks", func(w http.ResponseWriter, r *http.Request) {
		v := new(Hook)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	hook, _, err := client.Repositories.CreateHook("o", "r", input)
	if err != nil {
		t.Errorf("Repositories.CreateHook returned error: %v", err)
	}

	want := &Hook{ID: Int(1)}
	if !reflect.DeepEqual(hook, want) {
		t.Errorf("Repositories.CreateHook returned %+v, want %+v", hook, want)
	}
}

func TestRepositoriesService_CreateHook_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.CreateHook("%", "%", nil)
	testURLParseError(t, err)
}

func TestRepositoriesService_ListHooks(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/hooks", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}, {"id":2}]`)
	})

	opt := &ListOptions{Page: 2}

	hooks, _, err := client.Repositories.ListHooks("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.ListHooks returned error: %v", err)
	}

	want := []Hook{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(hooks, want) {
		t.Errorf("Repositories.ListHooks returned %+v, want %+v", hooks, want)
	}
}

func TestRepositoriesService_ListHooks_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.ListHooks("%", "%", nil)
	testURLParseError(t, err)
}

func TestRepositoriesService_GetHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	hook, _, err := client.Repositories.GetHook("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.GetHook returned error: %v", err)
	}

	want := &Hook{ID: Int(1)}
	if !reflect.DeepEqual(hook, want) {
		t.Errorf("Repositories.GetHook returned %+v, want %+v", hook, want)
	}
}

func TestRepositoriesService_GetHook_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.GetHook("%", "%", 1)
	testURLParseError(t, err)
}

func TestRepositoriesService_EditHook(t *testing.T) {
	setup()
	defer teardown()

	input := &Hook{Name: String("t")}

	mux.HandleFunc("/repos/o/r/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(Hook)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	hook, _, err := client.Repositories.EditHook("o", "r", 1, input)
	if err != nil {
		t.Errorf("Repositories.EditHook returned error: %v", err)
	}

	want := &Hook{ID: Int(1)}
	if !reflect.DeepEqual(hook, want) {
		t.Errorf("Repositories.EditHook returned %+v, want %+v", hook, want)
	}
}

func TestRepositoriesService_EditHook_invalidOwner(t *testing.T) {
	_, _, err := client.Repositories.EditHook("%", "%", 1, nil)
	testURLParseError(t, err)
}

func TestRepositoriesService_DeleteHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Repositories.DeleteHook("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.DeleteHook returned error: %v", err)
	}
}

func TestRepositoriesService_DeleteHook_invalidOwner(t *testing.T) {
	_, err := client.Repositories.DeleteHook("%", "%", 1)
	testURLParseError(t, err)
}

func TestRepositoriesService_PingHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/hooks/1/pings", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
	})

	_, err := client.Repositories.PingHook("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.PingHook returned error: %v", err)
	}
}

func TestRepositoriesService_TestHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/hooks/1/tests", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
	})

	_, err := client.Repositories.TestHook("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.TestHook returned error: %v", err)
	}
}

func TestRepositoriesService_TestHook_invalidOwner(t *testing.T) {
	_, err := client.Repositories.TestHook("%", "%", 1)
	testURLParseError(t, err)
}
