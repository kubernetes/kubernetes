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

package basicauth

import (
	"encoding/base64"
	"errors"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

type testPassword struct {
	Username string
	Password string
	Called   bool

	User user.Info
	OK   bool
	Err  error
}

func (t *testPassword) AuthenticatePassword(user, password string) (user.Info, bool, error) {
	t.Called = true
	t.Username = user
	t.Password = password
	return t.User, t.OK, t.Err
}

func TestBasicAuth(t *testing.T) {
	testCases := map[string]struct {
		Header   string
		Password testPassword

		ExpectedCalled   bool
		ExpectedUsername string
		ExpectedPassword string

		ExpectedUser string
		ExpectedOK   bool
		ExpectedErr  bool
	}{
		"no header": {
			Header: "",
		},
		"non-basic header": {
			Header: "Bearer foo",
		},
		"empty value basic header": {
			Header: "Basic",
		},
		"whitespace value basic header": {
			Header: "Basic  ",
		},
		"non base-64 basic header": {
			Header:      "Basic !@#$",
			ExpectedErr: true,
		},
		"malformed basic header": {
			Header:      "Basic " + base64.StdEncoding.EncodeToString([]byte("user_without_password")),
			ExpectedErr: true,
		},
		"empty password basic header": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("user_with_empty_password:")),
			ExpectedCalled:   true,
			ExpectedUsername: "user_with_empty_password",
			ExpectedPassword: "",
		},
		"valid basic header": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("myuser:mypassword:withcolon")),
			ExpectedCalled:   true,
			ExpectedUsername: "myuser",
			ExpectedPassword: "mypassword:withcolon",
		},
		"password auth returned user": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("myuser:mypw")),
			Password:         testPassword{User: &user.DefaultInfo{Name: "returneduser"}, OK: true},
			ExpectedCalled:   true,
			ExpectedUsername: "myuser",
			ExpectedPassword: "mypw",
			ExpectedUser:     "returneduser",
			ExpectedOK:       true,
		},
		"password auth returned error": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("myuser:mypw")),
			Password:         testPassword{Err: errors.New("auth error")},
			ExpectedCalled:   true,
			ExpectedUsername: "myuser",
			ExpectedPassword: "mypw",
			ExpectedErr:      true,
		},
	}

	for k, testCase := range testCases {
		password := testCase.Password
		auth := authenticator.Request(New(&password))

		req, _ := http.NewRequest("GET", "/", nil)
		if testCase.Header != "" {
			req.Header.Set("Authorization", testCase.Header)
		}

		user, ok, err := auth.AuthenticateRequest(req)

		if testCase.ExpectedCalled != password.Called {
			t.Fatalf("%s: Expected called=%v, got %v", k, testCase.ExpectedCalled, password.Called)
			continue
		}
		if testCase.ExpectedUsername != password.Username {
			t.Fatalf("%s: Expected called with username=%v, got %v", k, testCase.ExpectedUsername, password.Username)
			continue
		}
		if testCase.ExpectedPassword != password.Password {
			t.Fatalf("%s: Expected called with password=%v, got %v", k, testCase.ExpectedPassword, password.Password)
			continue
		}

		if testCase.ExpectedErr != (err != nil) {
			t.Fatalf("%s: Expected err=%v, got err=%v", k, testCase.ExpectedErr, err)
			continue
		}
		if testCase.ExpectedOK != ok {
			t.Fatalf("%s: Expected ok=%v, got ok=%v", k, testCase.ExpectedOK, ok)
			continue
		}
		if testCase.ExpectedUser != "" && testCase.ExpectedUser != user.GetName() {
			t.Fatalf("%s: Expected user.GetName()=%v, got %v", k, testCase.ExpectedUser, user.GetName())
			continue
		}
	}
}
