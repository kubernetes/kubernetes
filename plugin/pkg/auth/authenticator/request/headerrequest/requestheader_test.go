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
		nameHeaders        []string
		groupHeaders       []string
		extraPrefixHeaders []string
		requestHeaders     http.Header

		expectedUser user.Info
		expectedOk   bool
	}{
		"empty": {},
		"user no match": {
			nameHeaders: []string{"X-Remote-User"},
		},
		"user match": {
			nameHeaders:    []string{"X-Remote-User"},
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"user exact match": {
			nameHeaders: []string{"X-Remote-User"},
			requestHeaders: http.Header{
				"Prefixed-X-Remote-User-With-Suffix": {"Bob"},
				"X-Remote-User-With-Suffix":          {"Bob"},
			},
		},
		"user first match": {
			nameHeaders: []string{
				"X-Remote-User",
				"A-Second-X-Remote-User",
				"Another-X-Remote-User",
			},
			requestHeaders: http.Header{
				"X-Remote-User":          {"", "First header, second value"},
				"A-Second-X-Remote-User": {"Second header, first value", "Second header, second value"},
				"Another-X-Remote-User":  {"Third header, first value"}},
			expectedUser: &user.DefaultInfo{
				Name:   "Second header, first value",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"user case-insensitive": {
			nameHeaders:    []string{"x-REMOTE-user"},             // configured headers can be case-insensitive
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}}, // the parsed headers are normalized by the http package
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},

		"groups none": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},
		"groups all matches": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			requestHeaders: http.Header{
				"X-Remote-User":    {"Bob"},
				"X-Remote-Group-1": {"one-a", "one-b"},
				"X-Remote-Group-2": {"two-a", "two-b"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"one-a", "one-b", "two-a", "two-b"},
				Extra:  map[string][]string{},
			},
			expectedOk: true,
		},

		"extra prefix matches case-insensitive": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"one-a", "one-b", "two-a", "two-b"},
				Extra: map[string][]string{
					"key1": {"alfa", "bravo", "echo", "foxtrot"},
					"key2": {"charlie", "delta", "golf", "hotel"},
					"":     {"india", "juliet", "kilo", "lima"},
				},
			},
			expectedOk: true,
		},
	}

	for k, testcase := range testcases {
		auth, err := New(testcase.nameHeaders, testcase.groupHeaders, testcase.extraPrefixHeaders)
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
