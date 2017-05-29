/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"net/http"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
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
		"no auth": {},
		"empty password basic header": {
			ExpectedCalled:   true,
			ExpectedUsername: "user_with_empty_password",
			ExpectedPassword: "",
			ExpectedErr:      true,
		},
		"valid basic header": {
			ExpectedCalled:   true,
			ExpectedUsername: "myuser",
			ExpectedPassword: "mypassword:withcolon",
			ExpectedErr:      true,
		},
		"password auth returned user": {
			Password:         testPassword{User: &user.DefaultInfo{Name: "returneduser"}, OK: true},
			ExpectedCalled:   true,
			ExpectedUsername: "myuser",
			ExpectedPassword: "mypw",
			ExpectedUser:     "returneduser",
			ExpectedOK:       true,
		},
		"password auth returned error": {
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
		if testCase.ExpectedUsername != "" || testCase.ExpectedPassword != "" {
			req.SetBasicAuth(testCase.ExpectedUsername, testCase.ExpectedPassword)
		}

		user, ok, err := auth.AuthenticateRequest(req)

		if testCase.ExpectedCalled != password.Called {
			t.Errorf("%s: Expected called=%v, got %v", k, testCase.ExpectedCalled, password.Called)
			continue
		}
		if testCase.ExpectedUsername != password.Username {
			t.Errorf("%s: Expected called with username=%v, got %v", k, testCase.ExpectedUsername, password.Username)
			continue
		}
		if testCase.ExpectedPassword != password.Password {
			t.Errorf("%s: Expected called with password=%v, got %v", k, testCase.ExpectedPassword, password.Password)
			continue
		}

		if testCase.ExpectedErr != (err != nil) {
			t.Errorf("%s: Expected err=%v, got err=%v", k, testCase.ExpectedErr, err)
			continue
		}
		if testCase.ExpectedOK != ok {
			t.Errorf("%s: Expected ok=%v, got ok=%v", k, testCase.ExpectedOK, ok)
			continue
		}
		if testCase.ExpectedUser != "" && testCase.ExpectedUser != user.GetName() {
			t.Errorf("%s: Expected user.GetName()=%v, got %v", k, testCase.ExpectedUser, user.GetName())
			continue
		}
	}
}
