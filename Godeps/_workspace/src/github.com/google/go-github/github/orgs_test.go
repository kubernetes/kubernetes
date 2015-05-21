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

func TestOrganizationsService_List_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/orgs", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1},{"id":2}]`)
	})

	orgs, _, err := client.Organizations.List("", nil)
	if err != nil {
		t.Errorf("Organizations.List returned error: %v", err)
	}

	want := []Organization{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(orgs, want) {
		t.Errorf("Organizations.List returned %+v, want %+v", orgs, want)
	}
}

func TestOrganizationsService_List_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/orgs", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1},{"id":2}]`)
	})

	opt := &ListOptions{Page: 2}
	orgs, _, err := client.Organizations.List("u", opt)
	if err != nil {
		t.Errorf("Organizations.List returned error: %v", err)
	}

	want := []Organization{{ID: Int(1)}, {ID: Int(2)}}
	if !reflect.DeepEqual(orgs, want) {
		t.Errorf("Organizations.List returned %+v, want %+v", orgs, want)
	}
}

func TestOrganizationsService_List_invalidUser(t *testing.T) {
	_, _, err := client.Organizations.List("%", nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1, "login":"l", "url":"u", "avatar_url": "a", "location":"l"}`)
	})

	org, _, err := client.Organizations.Get("o")
	if err != nil {
		t.Errorf("Organizations.Get returned error: %v", err)
	}

	want := &Organization{ID: Int(1), Login: String("l"), URL: String("u"), AvatarURL: String("a"), Location: String("l")}
	if !reflect.DeepEqual(org, want) {
		t.Errorf("Organizations.Get returned %+v, want %+v", org, want)
	}
}

func TestOrganizationsService_Get_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.Get("%")
	testURLParseError(t, err)
}

func TestOrganizationsService_Edit(t *testing.T) {
	setup()
	defer teardown()

	input := &Organization{Login: String("l")}

	mux.HandleFunc("/orgs/o", func(w http.ResponseWriter, r *http.Request) {
		v := new(Organization)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	org, _, err := client.Organizations.Edit("o", input)
	if err != nil {
		t.Errorf("Organizations.Edit returned error: %v", err)
	}

	want := &Organization{ID: Int(1)}
	if !reflect.DeepEqual(org, want) {
		t.Errorf("Organizations.Edit returned %+v, want %+v", org, want)
	}
}

func TestOrganizationsService_Edit_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.Edit("%", nil)
	testURLParseError(t, err)
}
