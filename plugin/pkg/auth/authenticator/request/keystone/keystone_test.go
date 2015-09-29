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

package keystone

import (
	"encoding/base64"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/basicauth"
)

type testKeystoneAuthenticator struct {
	User user.Info
	OK   bool
	Err  error
}

func (osClient *testKeystoneAuthenticator) AuthenticatePassword(username string, password string) (user.Info, bool, error) {

	userPasswordMap := map[string]string{
		"user1": "password1",
		"user2": "password2",
		"user3": "password3",
		"user4": "password4",
		"user5": "password5",
		"user6": "password6",
		"user7": "password7",
		"user8": "password8",
		"user9": "password9",
	}

	if userPasswordMap[username] == password {
		return &user.DefaultInfo{Name: username}, true, nil
	}
	return nil, false, nil
}

func TestKeystoneAuth(t *testing.T) {

	testCases := map[string]struct {
		Header                string
		keystoneAuthenticator testKeystoneAuthenticator

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
			Header:     "Basic " + base64.StdEncoding.EncodeToString([]byte("user1:")),
			ExpectedOK: false,
		},
		"valid basic header": {
			Header:      "Basic " + base64.StdEncoding.EncodeToString([]byte("user1:password1:withcolon")),
			ExpectedOK:  false,
			ExpectedErr: false,
		},
		"password auth returned user": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("user1:password1")),
			ExpectedCalled:   true,
			ExpectedUsername: "user1",
			ExpectedPassword: "password1",
			ExpectedOK:       true,
		},
		"password auth returned error": {
			Header:           "Basic " + base64.StdEncoding.EncodeToString([]byte("user1:password2")),
			ExpectedCalled:   true,
			ExpectedUsername: "user1",
			ExpectedPassword: "password1",
			ExpectedErr:      false,
			ExpectedOK:       false,
		},
	}

	for k, testCase := range testCases {

		ksAuth := testCase.keystoneAuthenticator

		auth := basicauth.New(&ksAuth)

		req, _ := http.NewRequest("GET", "/", nil)
		if testCase.Header != "" {
			req.Header.Set("Authorization", testCase.Header)
		}

		user, ok, err := auth.AuthenticateRequest(req)

		if testCase.ExpectedErr && err == nil {
			t.Errorf("%s: Expected error, got none", k)
			continue
		}
		if !testCase.ExpectedErr && err != nil {
			t.Errorf("%s: Did not expect error, got err:%v", k, err)
			continue
		}
		if testCase.ExpectedOK != ok {
			t.Errorf("%s: Expected ok=%v, got %v", k, testCase.ExpectedOK, ok)
			continue
		}

		if testCase.ExpectedOK {
			if testCase.ExpectedUsername != user.GetName() {
				t.Errorf("%s: Expected user.name=%v, got %v", k, testCase.ExpectedUsername, user.GetName())
				continue
			}
		}
	}
}
