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

func TestUsersService_ListFollowers_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/followers", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	users, _, err := client.Users.ListFollowers("", opt)
	if err != nil {
		t.Errorf("Users.ListFollowers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Users.ListFollowers returned %+v, want %+v", users, want)
	}
}

func TestUsersService_ListFollowers_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/followers", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	users, _, err := client.Users.ListFollowers("u", nil)
	if err != nil {
		t.Errorf("Users.ListFollowers returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Users.ListFollowers returned %+v, want %+v", users, want)
	}
}

func TestUsersService_ListFollowers_invalidUser(t *testing.T) {
	_, _, err := client.Users.ListFollowers("%", nil)
	testURLParseError(t, err)
}

func TestUsersService_ListFollowing_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/following", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opts := &ListOptions{Page: 2}
	users, _, err := client.Users.ListFollowing("", opts)
	if err != nil {
		t.Errorf("Users.ListFollowing returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Users.ListFollowing returned %+v, want %+v", users, want)
	}
}

func TestUsersService_ListFollowing_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/following", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	users, _, err := client.Users.ListFollowing("u", nil)
	if err != nil {
		t.Errorf("Users.ListFollowing returned error: %v", err)
	}

	want := []User{{ID: Int(1)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Users.ListFollowing returned %+v, want %+v", users, want)
	}
}

func TestUsersService_ListFollowing_invalidUser(t *testing.T) {
	_, _, err := client.Users.ListFollowing("%", nil)
	testURLParseError(t, err)
}

func TestUsersService_IsFollowing_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/following/t", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	following, _, err := client.Users.IsFollowing("", "t")
	if err != nil {
		t.Errorf("Users.IsFollowing returned error: %v", err)
	}
	if want := true; following != want {
		t.Errorf("Users.IsFollowing returned %+v, want %+v", following, want)
	}
}

func TestUsersService_IsFollowing_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/following/t", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	following, _, err := client.Users.IsFollowing("u", "t")
	if err != nil {
		t.Errorf("Users.IsFollowing returned error: %v", err)
	}
	if want := true; following != want {
		t.Errorf("Users.IsFollowing returned %+v, want %+v", following, want)
	}
}

func TestUsersService_IsFollowing_false(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/following/t", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	following, _, err := client.Users.IsFollowing("u", "t")
	if err != nil {
		t.Errorf("Users.IsFollowing returned error: %v", err)
	}
	if want := false; following != want {
		t.Errorf("Users.IsFollowing returned %+v, want %+v", following, want)
	}
}

func TestUsersService_IsFollowing_error(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/following/t", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		http.Error(w, "BadRequest", http.StatusBadRequest)
	})

	following, _, err := client.Users.IsFollowing("u", "t")
	if err == nil {
		t.Errorf("Expected HTTP 400 response")
	}
	if want := false; following != want {
		t.Errorf("Users.IsFollowing returned %+v, want %+v", following, want)
	}
}

func TestUsersService_IsFollowing_invalidUser(t *testing.T) {
	_, _, err := client.Users.IsFollowing("%", "%")
	testURLParseError(t, err)
}

func TestUsersService_Follow(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/following/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
	})

	_, err := client.Users.Follow("u")
	if err != nil {
		t.Errorf("Users.Follow returned error: %v", err)
	}
}

func TestUsersService_Follow_invalidUser(t *testing.T) {
	_, err := client.Users.Follow("%")
	testURLParseError(t, err)
}

func TestUsersService_Unfollow(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/following/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Users.Unfollow("u")
	if err != nil {
		t.Errorf("Users.Follow returned error: %v", err)
	}
}

func TestUsersService_Unfollow_invalidUser(t *testing.T) {
	_, err := client.Users.Unfollow("%")
	testURLParseError(t, err)
}
