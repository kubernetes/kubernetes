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

func TestUser_marshall(t *testing.T) {
	testJSONMarshal(t, &User{}, "{}")

	u := &User{
		Login:       String("l"),
		ID:          Int(1),
		URL:         String("u"),
		AvatarURL:   String("a"),
		GravatarID:  String("g"),
		Name:        String("n"),
		Company:     String("c"),
		Blog:        String("b"),
		Location:    String("l"),
		Email:       String("e"),
		Hireable:    Bool(true),
		PublicRepos: Int(1),
		Followers:   Int(1),
		Following:   Int(1),
		CreatedAt:   &Timestamp{referenceTime},
	}
	want := `{
		"login": "l",
		"id": 1,
		"avatar_url": "a",
		"gravatar_id": "g",
		"name": "n",
		"company": "c",
		"blog": "b",
		"location": "l",
		"email": "e",
		"hireable": true,
		"public_repos": 1,
		"followers": 1,
		"following": 1,
		"created_at": ` + referenceTimeStr + `,
		"url": "u"
	}`
	testJSONMarshal(t, u, want)
}

func TestUsersService_Get_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	user, _, err := client.Users.Get("")
	if err != nil {
		t.Errorf("Users.Get returned error: %v", err)
	}

	want := &User{ID: Int(1)}
	if !reflect.DeepEqual(user, want) {
		t.Errorf("Users.Get returned %+v, want %+v", user, want)
	}
}

func TestUsersService_Get_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	user, _, err := client.Users.Get("u")
	if err != nil {
		t.Errorf("Users.Get returned error: %v", err)
	}

	want := &User{ID: Int(1)}
	if !reflect.DeepEqual(user, want) {
		t.Errorf("Users.Get returned %+v, want %+v", user, want)
	}
}

func TestUsersService_Get_invalidUser(t *testing.T) {
	_, _, err := client.Users.Get("%")
	testURLParseError(t, err)
}

func TestUsersService_Edit(t *testing.T) {
	setup()
	defer teardown()

	input := &User{Name: String("n")}

	mux.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
		v := new(User)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	user, _, err := client.Users.Edit(input)
	if err != nil {
		t.Errorf("Users.Edit returned error: %v", err)
	}

	want := &User{ID: Int(1)}
	if !reflect.DeepEqual(user, want) {
		t.Errorf("Users.Edit returned %+v, want %+v", user, want)
	}
}

func TestUsersService_ListAll(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"since": "1"})
		fmt.Fprint(w, `[{"id":2}]`)
	})

	opt := &UserListOptions{1}
	users, _, err := client.Users.ListAll(opt)
	if err != nil {
		t.Errorf("Users.Get returned error: %v", err)
	}

	want := []User{{ID: Int(2)}}
	if !reflect.DeepEqual(users, want) {
		t.Errorf("Users.ListAll returned %+v, want %+v", users, want)
	}
}
