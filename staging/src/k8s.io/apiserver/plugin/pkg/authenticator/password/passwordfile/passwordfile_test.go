/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestPasswordFile(t *testing.T) {
	auth, err := newWithContents(t, `
password1,user1,uid1
password2,user2,uid2
password3,user3,uid3,"group1,group2"
password4,user4,uid4,"group2"
password5,user5,uid5,group5
password6,user6,uid6,group5,otherdata
password7,user7,uid7,"group1,group2",otherdata
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
			User:     &user.DefaultInfo{Name: "user3", UID: "uid3", Groups: []string{"group1", "group2"}},
			Ok:       true,
		},
		{
			Username: "user4",
			Password: "password4",
			User:     &user.DefaultInfo{Name: "user4", UID: "uid4", Groups: []string{"group2"}},
			Ok:       true,
		},
		{
			Username: "user5",
			Password: "password5",
			User:     &user.DefaultInfo{Name: "user5", UID: "uid5", Groups: []string{"group5"}},
			Ok:       true,
		},
		{
			Username: "user6",
			Password: "password6",
			User:     &user.DefaultInfo{Name: "user6", UID: "uid6", Groups: []string{"group5"}},
			Ok:       true,
		},
		{
			Username: "user7",
			Password: "password7",
			User:     &user.DefaultInfo{Name: "user7", UID: "uid7", Groups: []string{"group1", "group2"}},
			Ok:       true,
		},
		{
			Username: "user7",
			Password: "passwordbad",
		},
		{
			Username: "userbad",
			Password: "password7",
		},
		{
			Username: "user8",
			Password: "password8",
		},
	}
	for i, testCase := range testCases {
		resp, ok, err := auth.AuthenticatePassword(context.Background(), testCase.Username, testCase.Password)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if testCase.User == nil {
			if resp != nil {
				t.Errorf("%d: unexpected non-nil user %#v", i, resp.User)
			}
		} else if !reflect.DeepEqual(testCase.User, resp.User) {
			t.Errorf("%d: expected user %#v, got %#v", i, testCase.User, resp.User)
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
