// Copyright 2015 The go-github AUTHORS. All rights reserved.
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

func TestOrganizationsService_ListHooks(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/hooks", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}, {"id":2}]`)
	})

	opt := &ListOptions{Page: 2}

	hooks, _, err := client.Organizations.ListHooks("o", opt)
	if err != nil {
		t.Errorf("Organizations.ListHooks returned error: %v", err)
	}

	want := []Hook{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(hooks, want) {
		t.Errorf("Organizations.ListHooks returned %+v, want %+v", hooks, want)
	}
}

func TestOrganizationsService_ListHooks_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.ListHooks("%", nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_GetHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	hook, _, err := client.Organizations.GetHook("o", 1)
	if err != nil {
		t.Errorf("Organizations.GetHook returned error: %v", err)
	}

	want := &Hook{ID: Int(1)}
	if !reflect.DeepEqual(hook, want) {
		t.Errorf("Organizations.GetHook returned %+v, want %+v", hook, want)
	}
}

func TestOrganizationsService_GetHook_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.GetHook("%", 1)
	testURLParseError(t, err)
}

func TestOrganizationsService_EditHook(t *testing.T) {
	setup()
	defer teardown()

	input := &Hook{Name: String("t")}

	mux.HandleFunc("/orgs/o/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(Hook)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	hook, _, err := client.Organizations.EditHook("o", 1, input)
	if err != nil {
		t.Errorf("Organizations.EditHook returned error: %v", err)
	}

	want := &Hook{ID: Int(1)}
	if !reflect.DeepEqual(hook, want) {
		t.Errorf("Organizations.EditHook returned %+v, want %+v", hook, want)
	}
}

func TestOrganizationsService_EditHook_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.EditHook("%", 1, nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_PingHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/hooks/1/pings", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
	})

	_, err := client.Organizations.PingHook("o", 1)
	if err != nil {
		t.Errorf("Organizations.PingHook returned error: %v", err)
	}
}

func TestOrganizationsService_DeleteHook(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/hooks/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Organizations.DeleteHook("o", 1)
	if err != nil {
		t.Errorf("Organizations.DeleteHook returned error: %v", err)
	}
}

func TestOrganizationsService_DeleteHook_invalidOrg(t *testing.T) {
	_, err := client.Organizations.DeleteHook("%", 1)
	testURLParseError(t, err)
}
