/*
Copyright 2016 The Kubernetes Authors.

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

package headerrequest

import (
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/auth/user"
)

func TestRequestHeader(t *testing.T) {
	testcases := map[string]struct {
		nameHeaders    []string
		requestHeaders http.Header

		expectedUser user.Info
		expectedOk   bool
	}{
		"empty": {},
		"no match": {
			nameHeaders: []string{"X-Remote-User"},
		},
		"match": {
			nameHeaders:    []string{"X-Remote-User"},
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}},
			expectedUser:   &user.DefaultInfo{Name: "Bob"},
			expectedOk:     true,
		},
		"exact match": {
			nameHeaders: []string{"X-Remote-User"},
			requestHeaders: http.Header{
				"Prefixed-X-Remote-User-With-Suffix": {"Bob"},
				"X-Remote-User-With-Suffix":          {"Bob"},
			},
		},
		"first match": {
			nameHeaders: []string{
				"X-Remote-User",
				"A-Second-X-Remote-User",
				"Another-X-Remote-User",
			},
			requestHeaders: http.Header{
				"X-Remote-User":          {"", "First header, second value"},
				"A-Second-X-Remote-User": {"Second header, first value", "Second header, second value"},
				"Another-X-Remote-User":  {"Third header, first value"}},
			expectedUser: &user.DefaultInfo{Name: "Second header, first value"},
			expectedOk:   true,
		},
		"case-insensitive": {
			nameHeaders:    []string{"x-REMOTE-user"},             // configured headers can be case-insensitive
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}}, // the parsed headers are normalized by the http package
			expectedUser:   &user.DefaultInfo{Name: "Bob"},
			expectedOk:     true,
		},
	}

	for k, testcase := range testcases {
		auth, err := New(testcase.nameHeaders)
		if err != nil {
			t.Fatal(err)
		}
		req := &http.Request{Header: testcase.requestHeaders}

		user, ok, _ := auth.AuthenticateRequest(req)
		if testcase.expectedOk != ok {
			t.Errorf("%v: expected %v, got %v", k, testcase.expectedOk, ok)
		}
		if e, a := testcase.expectedUser, user; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: expected %#v, got %#v", k, e, a)

		}
	}
}
