/*
Copyright 2016 The Kubernetes Authors All rights reserved.
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
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

var tests = []struct {
	token   string
	success bool
	message string
}{
	// passed token
	{
		token:   "abc",
		success: true,
	},
	// expired token
	{
		token:   "def",
		success: false,
		message: "token expired",
	},
	// not found token
	{
		token:   "ghi",
		success: false,
		message: "not found",
	},
}

type fakeValidator struct{}

func (fv *fakeValidator) support(token string) bool {
	return true
}

var sample = `{
  "access": {
    "token": {
      "tenant": {
        "name": "haibzhou",
        "id": "7903124f822749d195e02e3932d696e2",
      }
    },
    "user": {
      "username": "haibzhou",
    }
  }
}`

func (fv *fakeValidator) validate(token string) (*response, error) {
	r := &response{}
	json.Unmarshal([]byte(sample), r)
	if token == "abc" {
		// update the expired time
		r.Access.Token.ExpiredAt = time.Now().Add(6 * time.Hour).Format(time.RFC3339)
		return r, nil
	}
	if token == "def" {
		// update the expired time
		r.Access.Token.ExpiredAt = time.Now().Add(-6 * time.Hour).Format(time.RFC3339)
		return r, nil
	}
	return nil, errors.New("not found")
}

func TestAuthenticateToken(t *testing.T) {
	ka := &keystoneTokenAuthenticator{
		apiCallValidator: &fakeValidator{},
	}
	for _, tc := range tests {
		_, success, err := ka.AuthenticateToken(tc.token)
		if success != tc.success {
			t.Errorf("expected to be %v but got %v", tc.success, success)
		}
		if success {
			continue
		}
		if !strings.Contains(err.Error(), tc.message) {
			t.Errorf("expected to contain %s in error message, but did not", tc.message)
		}
	}
}
