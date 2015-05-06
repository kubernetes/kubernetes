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

func TestOrganizationsService_ListTeams(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/teams", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	teams, _, err := client.Organizations.ListTeams("o", opt)
	if err != nil {
		t.Errorf("Organizations.ListTeams returned error: %v", err)
	}

	want := []Team{{ID: Int(1)}}
	if !reflect.DeepEqual(teams, want) {
		t.Errorf("Organizations.ListTeams returned %+v, want %+v", teams, want)
	}
}

func TestOrganizationsService_ListTeams_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.ListTeams("%", nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_GetTeam(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1, "name":"n", "url":"u", "slug": "s", "permission":"p"}`)
	})

	team, _, err := client.Organizations.GetTeam(1)
	if err != nil {
		t.Errorf("Organizations.GetTeam returned error: %v", err)
	}

	want := &Team{ID: Int(1), Name: String("n"), URL: String("u"), Slug: String("s"), Permission: String("p")}
	if !reflect.DeepEqual(team, want) {
		t.Errorf("Organizations.GetTeam returned %+v, want %+v", team, want)
	}
}

func TestOrganizationsService_CreateTeam(t *testing.T) {
	setup()
	defer teardown()

	input := &Team{Name: String("n")}

	mux.HandleFunc("/orgs/o/teams", func(w http.ResponseWriter, r *http.Request) {
		v := new(Team)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	team, _, err := client.Organizations.CreateTeam("o", input)
	if err != nil {
		t.Errorf("Organizations.CreateTeam returned error: %v", err)
	}

	want := &Team{ID: Int(1)}
	if !reflect.DeepEqual(team, want) {
		t.Errorf("Organizations.CreateTeam returned %+v, want %+v", team, want)
	}
}

func TestOrganizationsService_CreateTeam_invalidOrg(t *testing.T) {
	_, _, err := client.Organizations.CreateTeam("%", nil)
	testURLParseError(t, err)
}

func TestOrganizationsService_EditTeam(t *testing.T) {
	setup()
	defer teardown()

	input := &Team{Name: String("n")}

	mux.HandleFunc("/teams/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(Team)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	team, _, err := client.Organizations.EditTeam(1, input)
	if err != nil {
		t.Errorf("Organizations.EditTeam returned error: %v", err)
	}

	want := &Team{ID: Int(1)}
	if !reflect.DeepEqual(team, want) {
		t.Errorf("Organizations.EditTeam returned %+v, want %+v", team, want)
	}
}

func TestOrganizationsService_DeleteTeam(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Organizations.DeleteTeam(1)
	if err != nil {
		t.Errorf("Organizations.DeleteTeam returned error: %v", err)
	}
}

func TestOrganizationsService_ListTeamMembers(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/members", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	members, _, err := client.Organizations.ListTeamMembers(1, opt)
	if err != nil {
		t.Errorf("Organizations.ListTeamMembers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(members, want) {
		t.Errorf("Organizations.ListTeamMembers returned %+v, want %+v", members, want)
	}
}

func TestOrganizationsService_IsTeamMember_true(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
	})

	member, _, err := client.Organizations.IsTeamMember(1, "u")
	if err != nil {
		t.Errorf("Organizations.IsTeamMember returned error: %v", err)
	}
	if want := true; member != want {
		t.Errorf("Organizations.IsTeamMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 404 response is interpreted as "false" and not an error
func TestOrganizationsService_IsTeamMember_false(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	member, _, err := client.Organizations.IsTeamMember(1, "u")
	if err != nil {
		t.Errorf("Organizations.IsTeamMember returned error: %+v", err)
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsTeamMember returned %+v, want %+v", member, want)
	}
}

// ensure that a 400 response is interpreted as an actual error, and not simply
// as "false" like the above case of a 404
func TestOrganizationsService_IsTeamMember_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	member, _, err := client.Organizations.IsTeamMember(1, "u")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; member != want {
		t.Errorf("Organizations.IsTeamMember returned %+v, want %+v", member, want)
	}
}

func TestOrganizationsService_IsTeamMember_invalidUser(t *testing.T) {
	_, _, err := client.Organizations.IsTeamMember(1, "%")
	testURLParseError(t, err)
}

func TestOrganizationsService_PublicizeMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Organizations.PublicizeMembership("o", "u")
	if err != nil {
		t.Errorf("Organizations.PublicizeMembership returned error: %v", err)
	}
}

func TestOrganizationsService_PublicizeMembership_invalidOrg(t *testing.T) {
	_, err := client.Organizations.PublicizeMembership("%", "u")
	testURLParseError(t, err)
}

func TestOrganizationsService_ConcealMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/orgs/o/public_members/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Organizations.ConcealMembership("o", "u")
	if err != nil {
		t.Errorf("Organizations.ConcealMembership returned error: %v", err)
	}
}

func TestOrganizationsService_ConcealMembership_invalidOrg(t *testing.T) {
	_, err := client.Organizations.ConcealMembership("%", "u")
	testURLParseError(t, err)
}

func TestOrganizationsService_ListTeamRepos(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	members, _, err := client.Organizations.ListTeamRepos(1, opt)
	if err != nil {
		t.Errorf("Organizations.ListTeamRepos returned error: %v", err)
	}

	want := []Repository{{ID: Int(1)}}
	if !reflect.DeepEqual(members, want) {
		t.Errorf("Organizations.ListTeamRepos returned %+v, want %+v", members, want)
	}
}

func TestOrganizationsService_IsTeamRepo_true(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	managed, _, err := client.Organizations.IsTeamRepo(1, "o", "r")
	if err != nil {
		t.Errorf("Organizations.IsTeamRepo returned error: %v", err)
	}
	if want := true; managed != want {
		t.Errorf("Organizations.IsTeamRepo returned %+v, want %+v", managed, want)
	}
}

func TestOrganizationsService_IsTeamRepo_false(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	managed, _, err := client.Organizations.IsTeamRepo(1, "o", "r")
	if err != nil {
		t.Errorf("Organizations.IsTeamRepo returned error: %v", err)
	}
	if want := false; managed != want {
		t.Errorf("Organizations.IsTeamRepo returned %+v, want %+v", managed, want)
	}
}

func TestOrganizationsService_IsTeamRepo_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	managed, _, err := client.Organizations.IsTeamRepo(1, "o", "r")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; managed != want {
		t.Errorf("Organizations.IsTeamRepo returned %+v, want %+v", managed, want)
	}
}

func TestOrganizationsService_IsTeamRepo_invalidOwner(t *testing.T) {
	_, _, err := client.Organizations.IsTeamRepo(1, "%", "r")
	testURLParseError(t, err)
}

func TestOrganizationsService_AddTeamRepo(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Organizations.AddTeamRepo(1, "o", "r")
	if err != nil {
		t.Errorf("Organizations.AddTeamRepo returned error: %v", err)
	}
}

func TestOrganizationsService_AddTeamRepo_noAccess(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		w.WriteHeader(422)
	})

	_, err := client.Organizations.AddTeamRepo(1, "o", "r")
	if err == nil {
		t.Errorf("Expcted error to be returned")
	}
}

func TestOrganizationsService_AddTeamRepo_invalidOwner(t *testing.T) {
	_, err := client.Organizations.AddTeamRepo(1, "%", "r")
	testURLParseError(t, err)
}

func TestOrganizationsService_RemoveTeamRepo(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/repos/o/r", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Organizations.RemoveTeamRepo(1, "o", "r")
	if err != nil {
		t.Errorf("Organizations.RemoveTeamRepo returned error: %v", err)
	}
}

func TestOrganizationsService_RemoveTeamRepo_invalidOwner(t *testing.T) {
	_, err := client.Organizations.RemoveTeamRepo(1, "%", "r")
	testURLParseError(t, err)
}

func TestOrganizationsService_GetTeamMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/memberships/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"url":"u", "state":"active"}`)
	})

	membership, _, err := client.Organizations.GetTeamMembership(1, "u")
	if err != nil {
		t.Errorf("Organizations.GetTeamMembership returned error: %v", err)
	}

	want := &Membership{URL: String("u"), State: String("active")}
	if !reflect.DeepEqual(membership, want) {
		t.Errorf("Organizations.GetTeamMembership returned %+v, want %+v", membership, want)
	}
}

func TestOrganizationsService_AddTeamMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/memberships/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
		fmt.Fprint(w, `{"url":"u", "state":"pending"}`)
	})

	membership, _, err := client.Organizations.AddTeamMembership(1, "u")
	if err != nil {
		t.Errorf("Organizations.AddTeamMembership returned error: %v", err)
	}

	want := &Membership{URL: String("u"), State: String("pending")}
	if !reflect.DeepEqual(membership, want) {
		t.Errorf("Organizations.AddTeamMembership returned %+v, want %+v", membership, want)
	}
}

func TestOrganizationsService_RemoveTeamMembership(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/teams/1/memberships/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
		w.WriteHeader(http.StatusNoContent)
	})

	_, err := client.Organizations.RemoveTeamMembership(1, "u")
	if err != nil {
		t.Errorf("Organizations.RemoveTeamMembership returned error: %v", err)
	}
}

func TestOrganizationsService_ListUserTeams(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/teams", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "1"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 1}
	teams, _, err := client.Organizations.ListUserTeams(opt)
	if err != nil {
		t.Errorf("Organizations.ListUserTeams returned error: %v", err)
	}

	want := []Team{{ID: Int(1)}}
	if !reflect.DeepEqual(teams, want) {
		t.Errorf("Organizations.ListUserTeams returned %+v, want %+v", teams, want)
	}
}
