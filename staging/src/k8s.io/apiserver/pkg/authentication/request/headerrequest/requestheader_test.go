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

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestRequestHeader(t *testing.T) {
	testcases := map[string]struct {
		nameHeaders        []string
		groupHeaders       []string
		extraPrefixHeaders []string
		requestHeaders     http.Header
		finalHeaders       http.Header

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
		"groups case-insensitive": {
			nameHeaders:  []string{"X-REMOTE-User"},
			groupHeaders: []string{"X-REMOTE-Group"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Bob"},
				"X-Remote-Group": {"Users"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"Users"},
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

		"extra prefix matches case-insensitive with unrelated headers": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Group-Remote":        {"snorlax"}, // unrelated header
				"X-Group-Bear":          {"panda"},   // another unrelated header
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
			finalHeaders: http.Header{
				"X-Group-Remote": {"snorlax"},
				"X-Group-Bear":   {"panda"},
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

		"escaped extra keys": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group"},
			extraPrefixHeaders: []string{"X-Remote-Extra-"},
			requestHeaders: http.Header{
				"X-Remote-User":                                            {"Bob"},
				"X-Remote-Group":                                           {"one-a", "one-b"},
				"X-Remote-Extra-Alpha":                                     {"alphabetical"},
				"X-Remote-Extra-Alph4num3r1c":                              {"alphanumeric"},
				"X-Remote-Extra-Percent%20encoded":                         {"percent encoded"},
				"X-Remote-Extra-Almost%zzpercent%xxencoded":                {"not quite percent encoded"},
				"X-Remote-Extra-Example.com%2fpercent%2520encoded":         {"url with double percent encoding"},
				"X-Remote-Extra-Example.com%2F%E4%BB%8A%E6%97%A5%E3%81%AF": {"url with unicode"},
				"X-Remote-Extra-Abc123!#$+.-_*\\^`~|'":                     {"header key legal characters"},
			},
			expectedUser: &user.DefaultInfo{
				Name:   "Bob",
				Groups: []string{"one-a", "one-b"},
				Extra: map[string][]string{
					"alpha":                         {"alphabetical"},
					"alph4num3r1c":                  {"alphanumeric"},
					"percent encoded":               {"percent encoded"},
					"almost%zzpercent%xxencoded":    {"not quite percent encoded"},
					"example.com/percent%20encoded": {"url with double percent encoding"},
					"example.com/今日は":               {"url with unicode"},
					"abc123!#$+.-_*\\^`~|'":         {"header key legal characters"},
				},
			},
			expectedOk: true,
		},
	}

	for k, testcase := range testcases {
		t.Run(k, func(t *testing.T) {
			auth, err := New(testcase.nameHeaders, testcase.groupHeaders, testcase.extraPrefixHeaders)
			if err != nil {
				t.Fatal(err)
			}
			req := &http.Request{Header: testcase.requestHeaders}

			resp, ok, _ := auth.AuthenticateRequest(req)
			if testcase.expectedOk != ok {
				t.Errorf("%v: expected %v, got %v", k, testcase.expectedOk, ok)
			}
			if !ok {
				return
			}
			if e, a := testcase.expectedUser, resp.User; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: expected %#v, got %#v", k, e, a)
			}

			want := testcase.finalHeaders
			if want == nil && testcase.requestHeaders != nil {
				want = http.Header{}
			}
			if diff := cmp.Diff(want, testcase.requestHeaders); len(diff) > 0 {
				t.Errorf("unexpected final headers (-want +got):\n%s", diff)
			}
		})
	}
}
