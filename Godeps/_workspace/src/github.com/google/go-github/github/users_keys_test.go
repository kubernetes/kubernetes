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

func TestUsersService_ListKeys_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/keys", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	keys, _, err := client.Users.ListKeys("", opt)
	if err != nil {
		t.Errorf("Users.ListKeys returned error: %v", err)
	}

	want := []Key{{ID: Int(1)}}
	if !reflect.DeepEqual(keys, want) {
		t.Errorf("Users.ListKeys returned %+v, want %+v", keys, want)
	}
}

func TestUsersService_ListKeys_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/users/u/keys", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id":1}]`)
	})

	keys, _, err := client.Users.ListKeys("u", nil)
	if err != nil {
		t.Errorf("Users.ListKeys returned error: %v", err)
	}

	want := []Key{{ID: Int(1)}}
	if !reflect.DeepEqual(keys, want) {
		t.Errorf("Users.ListKeys returned %+v, want %+v", keys, want)
	}
}

func TestUsersService_ListKeys_invalidUser(t *testing.T) {
	_, _, err := client.Users.ListKeys("%", nil)
	testURLParseError(t, err)
}

func TestUsersService_GetKey(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/keys/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	key, _, err := client.Users.GetKey(1)
	if err != nil {
		t.Errorf("Users.GetKey returned error: %v", err)
	}

	want := &Key{ID: Int(1)}
	if !reflect.DeepEqual(key, want) {
		t.Errorf("Users.GetKey returned %+v, want %+v", key, want)
	}
}

func TestUsersService_CreateKey(t *testing.T) {
	setup()
	defer teardown()

	input := &Key{Key: String("k"), Title: String("t")}

	mux.HandleFunc("/user/keys", func(w http.ResponseWriter, r *http.Request) {
		v := new(Key)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w, `{"id":1}`)
	})

	key, _, err := client.Users.CreateKey(input)
	if err != nil {
		t.Errorf("Users.GetKey returned error: %v", err)
	}

	want := &Key{ID: Int(1)}
	if !reflect.DeepEqual(key, want) {
		t.Errorf("Users.GetKey returned %+v, want %+v", key, want)
	}
}

func TestUsersService_DeleteKey(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/user/keys/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Users.DeleteKey(1)
	if err != nil {
		t.Errorf("Users.DeleteKey returned error: %v", err)
	}
}
