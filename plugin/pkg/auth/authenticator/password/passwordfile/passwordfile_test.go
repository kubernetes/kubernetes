/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package passwordfile

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/auth/user"
)

func TestPasswordFile(t *testing.T) {
	auth, err := newWithContents(t, `
password1,user1,uid1
password2,user2,uid2
`)
	if err != nil {
		t.Fatalf("unable to read passwordfile: %v", err)
	}

	testCases := []struct {
		Username string
		Password string
		User     *user.DefaultInfo
		Ok       bool
		Err      bool
	}{
		{
			Username: "user1",
			Password: "password1",
			User:     &user.DefaultInfo{Name: "user1", UID: "uid1"},
			Ok:       true,
		},
		{
			Username: "user2",
			Password: "password2",
			User:     &user.DefaultInfo{Name: "user2", UID: "uid2"},
			Ok:       true,
		},
		{
			Username: "user1",
			Password: "password2",
		},
		{
			Username: "user2",
			Password: "password1",
		},
		{
			Username: "user3",
			Password: "password3",
		},
		{
			Username: "user4",
			Password: "password4",
		},
	}
	for i, testCase := range testCases {
		user, ok, err := auth.AuthenticatePassword(testCase.Username, testCase.Password)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
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
	}
}

func TestBadPasswordFile(t *testing.T) {
	if _, err := newWithContents(t, `
password1,user1,uid1
password2,user2,uid2
password3,user3
password4
`); err == nil {
		t.Fatalf("unexpected non error")
	}
}

func TestInsufficientColumnsPasswordFile(t *testing.T) {
	if _, err := newWithContents(t, "password4\n"); err == nil {
		t.Fatalf("unexpected non error")
	}
}

func newWithContents(t *testing.T, contents string) (auth *PasswordAuthenticator, err error) {
	f, err := ioutil.TempFile("", "passwordfile_test")
	if err != nil {
		t.Fatalf("unexpected error creating passwordfile: %v", err)
	}
	f.Close()
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(contents), 0700); err != nil {
		t.Fatalf("unexpected error writing passwordfile: %v", err)
	}

	return NewCSV(f.Name())
}
