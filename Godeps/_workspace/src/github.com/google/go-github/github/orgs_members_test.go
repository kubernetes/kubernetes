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

func TestOrganizationsService_ListMembers(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/members", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"filter": "2fa_disabled",
			"page":   "2",
		})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListMembersOptions{
		PublicOnly:  false,
		Filter:      "2fa_disabled",
		ListOptions: ListOptions{Page: 2},
	}
	members, _, err := client.Organizations.ListMembers("o", opt)
	if err != nil {
		t.Errorf("Organizations.ListMembers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(members, want) {
		t.Errorf("Organizations.ListMembers returned %+v, want %+v", members, want)
	}
}

func TestOrganizationsService_ListMembers_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.ListMembers("%", nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_ListMembers_public(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListMembersOptions{PublicOnly: true}
	members, _, err := client.Organizations.ListMembers("o", opt)
	if err != nil {
		t.Errorf("Organizations.ListMembers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(members, want) {
		t.Errorf("Organizations.ListMembers returned %+v, want %+v", members, want)
	}
}

func TestOrganizationsService_IsMember(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	member, _, err := client.Organizations.IsMember("o", "u")
	if err != nil {
		t.Errorf("Organizations.IsMember returned error: %v", err)
	}
	if want := true; member != want {
		t.Errorf("Organizations.IsMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 404 response is interpreted as "false" and not an error
func TestOrganizationsService_IsMember_notMember(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	member, _, err := client.Organizations.IsMember("o", "u")
	if err != nil {
		t.Errorf("Organizations.IsMember returned error: %+v", err)
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 400 response is interpreted as an actual error, and not simply
// as "false" like the above case of a 404
func TestOrganizationsService_IsMember_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	member, _, err := client.Organizations.IsMember("o", "u")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsMember returned %+v, want %+v", member, want)
	}
}

func TestOrganizationsService_IsMember_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.IsMember("%", "u")
	testURLParseError(t, err)
}

func TestOrganizationsService_IsPublicMember(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	member, _, err := client.Organizations.IsPublicMember("o", "u")
	if err != nil {
		t.Errorf("Organizations.IsPublicMember returned error: %v", err)
	}
	if want := true; member != want {
		t.Errorf("Organizations.IsPublicMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 404 response is interpreted as "false" and not an error
func TestOrganizationsService_IsPublicMember_notMember(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	member, _, err := client.Organizations.IsPublicMember("o", "u")
	if err != nil {
		t.Errorf("Organizations.IsPublicMember returned error: %v", err)
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsPublicMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 400 response is interpreted as an actual error, and not simply
// as "false" like the above case of a 404
func TestOrganizationsService_IsPublicMember_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	member, _, err := client.Organizations.IsPublicMember("o", "u")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsPublicMember returned %+v, want %+v", member, want)
	}
}

func TestOrganizationsService_IsPublicMember_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.IsPublicMember("%", "u")
	testURLParseError(t, err)
}

func TestOrganizationsService_RemoveMember(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Organizations.RemoveMember("o", "u")
	if err != nil {
		t.Errorf("Organizations.RemoveMember returned error: %v", err)
	}
}

func TestOrganizationsService_RemoveMember_invalidOrg(t *testing.T) {
	_, err := client.Organizations.RemoveMember("%", "u")
	testURLParseError(t, err)
}

func TestOrganizationsService_ListOrgMemberships(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/memberships/orgs", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"state": "active",
			"page":  "2",
		})
		fmt.Fprint(w, `[{"url":"u"}]`)
	})

	opt := &ListOrgMembershipsOptions{
		State:       "active",
		ListOptions: ListOptions{Page: 2},
	}
	memberships, _, err := client.Organizations.ListOrgMemberships(opt)
	if err != nil {
		t.Errorf("Organizations.ListOrgMemberships returned error: %v", err)
	}

	want := []Membership{{URL: String("u")}}
	if !reflect.DeepEqual(memberships, want) {
		t.Errorf("Organizations.ListOrgMemberships returned %+v, want %+v", memberships, want)
	}
}

func TestOrganizationsService_GetOrgMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/memberships/orgs/o", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"url":"u"}`)
	})

	membership, _, err := client.Organizations.GetOrgMembership("o")
	if err != nil {
		t.Errorf("Organizations.GetOrgMembership returned error: %v", err)
	}

	want := &Membership{URL: String("u")}
	if !reflect.DeepEqual(membership, want) {
		t.Errorf("Organizations.GetOrgMembership returned %+v, want %+v", membership, want)
	}
}

func TestOrganizationsService_EditOrgMembership(t *testing.T) {
	setup()
	defer teardown()

	input := &Membership{State: String("active")}

	mux.HandleFunc("/user/memberships/orgs/o", func(w http.ResponseWriter, r *http.Request) {
		v := new(Membership)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"url":"u"}`)
	})

	membership, _, err := client.Organizations.EditOrgMembership("o", input)
	if err != nil {
		t.Errorf("Organizations.EditOrgMembership returned error: %v", err)
	}

	want := &Membership{URL: String("u")}
	if !reflect.DeepEqual(membership, want) {
		t.Errorf("Organizations.EditOrgMembership returned %+v, want %+v", membership, want)
	}
}
