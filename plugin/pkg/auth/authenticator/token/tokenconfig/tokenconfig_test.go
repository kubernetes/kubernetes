/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenconfig

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/auth/user"
)

func TestTokenConfigJSONFile(t *testing.T) {
	auth, err := newWithContents(t, `
[
	{ "token": "token1", "name": "user1", "uid": "uid1" },
	{ "token": "token2", "name": "user2", "uid": "uid2" },
	{ "token": "token3", "name": "user3", "uid": "uid3", "groups": [ "group1", "group2" ] },
	{ "token": "token4", "name": "user4", "uid": "uid4", "groups": [ "group1" ] }
]`)
	if err != nil {
		t.Fatalf("unable to read tokenfile: %v", err)
	}

	testCases := []struct {
		Token string
		User  *user.DefaultInfo
		Ok    bool
		Err   bool
	}{
		{
			Token: "token1",
			User:  &user.DefaultInfo{Name: "user1", UID: "uid1"},
			Ok:    true,
		},
		{
			Token: "token2",
			User:  &user.DefaultInfo{Name: "user2", UID: "uid2"},
			Ok:    true,
		},
		{
			Token: "token3",
			User:  &user.DefaultInfo{Name: "user3", UID: "uid3", Groups: []string{"group1", "group2"}},
			Ok:    true,
		},
		{
			Token: "token4",
			User:  &user.DefaultInfo{Name: "user4", UID: "uid4", Groups: []string{"group1"}},
			Ok:    true,
		},
		{
			Token: "token5",
		},
		{
			Token: "token6",
		},
	}
	for i, testCase := range testCases {
		user, ok, err := auth.AuthenticateToken(testCase.Token)
		if testCase.User == nil {
			if user != nil {
				t.Errorf("%d: unexpected non-nil user %#v", i, user)
			}
		} else if !reflect.DeepEqual(testCase.User, user) {
			t.Errorf("%d: expected user %#v, got %#v", i, testCase.User, user)
		}

		if testCase.Ok != ok {
			t.Errorf("%d: expected auth %v, got %v", i, testCase.Ok, ok)
		}
		switch {
		case err == nil && testCase.Err:
			t.Errorf("%d: unexpected nil error", i)
		case err != nil && !testCase.Err:
			t.Errorf("%d: unexpected error: %v", i, err)
		}
	}
}

func TestTokenConfigYAMLFile(t *testing.T) {
	auth, err := newWithContents(t, `
- token: token1
  name: user1
  uid: uid1
- token: token2
  name: user2
  uid: uid2
- token: token3
  name: user3
  uid: uid3
  groups: [ "group1", "group2" ]
- token: token4
  name: user4
  uid: uid4
  groups:
    - group1
`)
	if err != nil {
		t.Fatalf("unable to read tokenfile: %v", err)
	}

	testCases := []struct {
		Token string
		User  *user.DefaultInfo
		Ok    bool
		Err   bool
	}{
		{
			Token: "token1",
			User:  &user.DefaultInfo{Name: "user1", UID: "uid1"},
			Ok:    true,
		},
		{
			Token: "token2",
			User:  &user.DefaultInfo{Name: "user2", UID: "uid2"},
			Ok:    true,
		},
		{
			Token: "token3",
			User:  &user.DefaultInfo{Name: "user3", UID: "uid3", Groups: []string{"group1", "group2"}},
			Ok:    true,
		},
		{
			Token: "token4",
			User:  &user.DefaultInfo{Name: "user4", UID: "uid4", Groups: []string{"group1"}},
			Ok:    true,
		},
		{
			Token: "token5",
		},
		{
			Token: "token6",
		},
	}
	for i, testCase := range testCases {
		user, ok, err := auth.AuthenticateToken(testCase.Token)
		if testCase.User == nil {
			if user != nil {
				t.Errorf("%d: unexpected non-nil user %#v", i, user)
			}
		} else if !reflect.DeepEqual(testCase.User, user) {
			t.Errorf("%d: expected user %#v, got %#v", i, testCase.User, user)
		}

		if testCase.Ok != ok {
			t.Errorf("%d: expected auth %v, got %v", i, testCase.Ok, ok)
		}
		switch {
		case err == nil && testCase.Err:
			t.Errorf("%d: unexpected nil error", i)
		case err != nil && !testCase.Err:
			t.Errorf("%d: unexpected error: %v", i, err)
		}
	}
}

func TestBadTokenConfigFile(t *testing.T) {
	_, err := newWithContents(t, `
- token: token1
  uid: uid1
- token: token2
  name: user2
  uid: uid2
`)
	if err == nil {
		t.Fatalf("unexpected non error")
	}
}

func TestBadTokenConfigMissingYAMLFields(t *testing.T) {
	_, err := newWithContents(t, `
[
	{ "token": "token1", "name": "user1", "uid": "uid1" },
	{ "token": "token2", "name": "user2" }
]
`)
	if err == nil {
		t.Fatalf("unexpected non error")
	}
}

func TestBadTokenConfigMissingJSONFields(t *testing.T) {
	_, err := newWithContents(t, `
[
	{ "token": "token1", "name": "user1", "uid": "uid1" },
	{ "token": "token2", "name": "user2" }
]
`)
	if err == nil {
		t.Fatalf("unexpected non error")
	}
}

func newWithContents(t *testing.T, contents string) (auth *TokenConfigAuthenticator, err error) {
	f, err := ioutil.TempFile("", "tokenfile_test")
	if err != nil {
		t.Fatalf("unexpected error creating tokenfile: %v", err)
	}
	f.Close()
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(contents), 0700); err != nil {
		t.Fatalf("unexpected error writing tokenfile: %v", err)
	}

	return NewTokenConfig(f.Name())
}
