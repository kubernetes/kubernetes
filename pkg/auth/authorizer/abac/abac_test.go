/*
Copyright 2014 Google Inc. All rights reserved.

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

package abac

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

func TestEmptyFile(t *testing.T) {
	_, err := newWithContents(t, "")
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

func TestOneLineFileNoNewLine(t *testing.T) {
	_, err := newWithContents(t, `{"user":"scheduler",  "readonly": true, "kind": "pods", "namespace":"ns1"}`)
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

func TestTwoLineFile(t *testing.T) {
	_, err := newWithContents(t, `{"user":"scheduler",  "readonly": true, "kind": "pods"}
{"user":"scheduler",  "readonly": true, "kind": "services"}
`)
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

// Test the file that we will point users at as an example.
func TestExampleFile(t *testing.T) {
	_, err := NewFromFile("./example_policy_file.jsonl")
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

func NotTestAuthorize(t *testing.T) {
	a, err := newWithContents(t, `{                     "readonly": true, "kind": "events"}
{"user":"scheduler",  "readonly": true, "kind": "pods"}
{"user":"scheduler",              "kind": "bindings"}
{"user":"kubelet",    "readonly": true, "kind": "bindings"}
{"user":"kubelet",                "kind": "events"}
{"user":"alice",                                     "ns": "projectCaribou"}
{"user":"bob",        "readonly": true,                    "ns": "projectCaribou"}
`)
	if err != nil {
		t.Fatalf("unable to read policy file: %v", err)
	}

	uScheduler := &user.DefaultInfo{Name: "scheduler", UID: "uid1"}
	uAlice := &user.DefaultInfo{Name: "alice", UID: "uid3"}
	uChuck := &user.DefaultInfo{Name: "chuck", UID: "uid5"}

	testCases := []struct {
		Attributes  authorizer.APIAttributesRecord
		ExpectAllow bool
	}{
		// Scheduler can read pods
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "get", Resource: "pods", Namespace: "ns1"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "get", Resource: "pods", Namespace: ""}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "list", Resource: "pods", Namespace: "ns1"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "list", Resource: "pods", Namespace: ""}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "watch", Resource: "pods", Namespace: "ns1"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "watch", Resource: "pods", Namespace: ""}, ExpectAllow: true},
		// Scheduler cannot write pods
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "create", Resource: "pods", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "create", Resource: "pods", Namespace: ""}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "update", Resource: "pods", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "update", Resource: "pods", Namespace: ""}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "delete", Resource: "pods", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "delete", Resource: "pods", Namespace: ""}, ExpectAllow: false},
		// Scheduler can write bindings
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "get", Resource: "bindings", Namespace: "ns1"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uScheduler, Verb: "get", Resource: "bindings", Namespace: ""}, ExpectAllow: true},

		// Alice can read and write anything in the right namespace.
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "pods", Namespace: "projectCaribou"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "widgets", Namespace: "projectCaribou"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "", Namespace: "projectCaribou"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "create", Resource: "pods", Namespace: "projectCaribou"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "create", Resource: "widgets", Namespace: "projectCaribou"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "create", Resource: "", Namespace: "projectCaribou"}, ExpectAllow: true},
		// .. but not the wrong namespace.
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "pods", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "widgets", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uAlice, Verb: "get", Resource: "", Namespace: "ns1"}, ExpectAllow: false},

		// Chuck can read events, since anyone can.
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "get", Resource: "events", Namespace: "ns1"}, ExpectAllow: true},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "get", Resource: "events", Namespace: ""}, ExpectAllow: true},
		// Chuck can't do other things.
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "create", Resource: "events", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "get", Resource: "pods", Namespace: "ns1"}, ExpectAllow: false},
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "get", Resource: "floop", Namespace: "ns1"}, ExpectAllow: false},
		// Chunk can't access things with no kind or namespace
		// TODO: find a way to give someone access to miscelaneous endpoints, such as
		// /healthz, /version, etc.
		{Attributes: authorizer.APIAttributesRecord{UserInfo: uChuck, Verb: "get", Resource: "", Namespace: ""}, ExpectAllow: false},
	}
	for _, tc := range testCases {
		t.Logf("tc: %v -> attr %v", tc, tc.Attributes)
		err := a.AuthorizeAPIRequest(tc.Attributes)
		actualAllow := bool(err == nil)
		if tc.ExpectAllow != actualAllow {
			t.Errorf("Expected allowed=%v but actually allowed=%v, for case %v",
				tc.ExpectAllow, actualAllow, tc)
		}
	}
}

func TestSubjectMatches(t *testing.T) {
	testCases := map[string]struct {
		User        user.DefaultInfo
		PolicyUser  string
		PolicyGroup string
		ExpectMatch bool
	}{
		"empty policy matches unauthed user": {
			User:        user.DefaultInfo{},
			PolicyUser:  "",
			PolicyGroup: "",
			ExpectMatch: true,
		},
		"empty policy matches authed user": {
			User:        user.DefaultInfo{Name: "Foo"},
			PolicyUser:  "",
			PolicyGroup: "",
			ExpectMatch: true,
		},
		"empty policy matches authed user with groups": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"a", "b"}},
			PolicyUser:  "",
			PolicyGroup: "",
			ExpectMatch: true,
		},

		"user policy does not match unauthed user": {
			User:        user.DefaultInfo{},
			PolicyUser:  "Foo",
			PolicyGroup: "",
			ExpectMatch: false,
		},
		"user policy does not match different user": {
			User:        user.DefaultInfo{Name: "Bar"},
			PolicyUser:  "Foo",
			PolicyGroup: "",
			ExpectMatch: false,
		},
		"user policy is case-sensitive": {
			User:        user.DefaultInfo{Name: "foo"},
			PolicyUser:  "Foo",
			PolicyGroup: "",
			ExpectMatch: false,
		},
		"user policy does not match substring": {
			User:        user.DefaultInfo{Name: "FooBar"},
			PolicyUser:  "Foo",
			PolicyGroup: "",
			ExpectMatch: false,
		},
		"user policy matches username": {
			User:        user.DefaultInfo{Name: "Foo"},
			PolicyUser:  "Foo",
			PolicyGroup: "",
			ExpectMatch: true,
		},

		"group policy does not match unauthed user": {
			User:        user.DefaultInfo{},
			PolicyUser:  "",
			PolicyGroup: "Foo",
			ExpectMatch: false,
		},
		"group policy does not match user in different group": {
			User:        user.DefaultInfo{Name: "FooBar", Groups: []string{"B"}},
			PolicyUser:  "",
			PolicyGroup: "A",
			ExpectMatch: false,
		},
		"group policy is case-sensitive": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"A", "B", "C"}},
			PolicyUser:  "",
			PolicyGroup: "b",
			ExpectMatch: false,
		},
		"group policy does not match substring": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"A", "BBB", "C"}},
			PolicyUser:  "",
			PolicyGroup: "B",
			ExpectMatch: false,
		},
		"group policy matches user in group": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"A", "B", "C"}},
			PolicyUser:  "",
			PolicyGroup: "B",
			ExpectMatch: true,
		},

		"user and group policy requires user match": {
			User:        user.DefaultInfo{Name: "Bar", Groups: []string{"A", "B", "C"}},
			PolicyUser:  "Foo",
			PolicyGroup: "B",
			ExpectMatch: false,
		},
		"user and group policy requires group match": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"A", "B", "C"}},
			PolicyUser:  "Foo",
			PolicyGroup: "D",
			ExpectMatch: false,
		},
		"user and group policy matches": {
			User:        user.DefaultInfo{Name: "Foo", Groups: []string{"A", "B", "C"}},
			PolicyUser:  "Foo",
			PolicyGroup: "B",
			ExpectMatch: true,
		},
	}

	for k, tc := range testCases {
		actualMatch := policy{User: tc.PolicyUser, Group: tc.PolicyGroup}.subjectMatches(&tc.User)
		if tc.ExpectMatch != actualMatch {
			t.Errorf("%v: Expected actorMatches=%v but actually got=%v",
				k, tc.ExpectMatch, actualMatch)
		}
	}
}

func newWithContents(t *testing.T, contents string) (authorizer.Authorizer, error) {
	f, err := ioutil.TempFile("", "abac_test")
	if err != nil {
		t.Fatalf("unexpected error creating policyfile: %v", err)
	}
	f.Close()
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(contents), 0700); err != nil {
		t.Fatalf("unexpected error writing policyfile: %v", err)
	}

	pl, err := NewFromFile(f.Name())
	return pl, err
}

func TestAPIPolicy(t *testing.T) {
	tests := []struct {
		policy  policy
		attr    authorizer.APIAttributesRecord
		matches bool
		name    string
	}{
		{
			policy:  policy{},
			attr:    authorizer.APIAttributesRecord{},
			matches: true,
			name:    "null",
		},
		{
			policy: policy{
				Readonly: true,
			},
			attr:    authorizer.APIAttributesRecord{},
			matches: false,
			name:    "read-only mismatch",
		},
		{
			policy: policy{
				User: "foo",
			},
			attr: authorizer.APIAttributesRecord{
				UserInfo: &user.DefaultInfo{
					Name: "bar",
				},
			},
			matches: false,
			name:    "user name mis-match",
		},
		{
			policy: policy{
				Resource: "foo",
			},
			attr: authorizer.APIAttributesRecord{
				Resource: "bar",
			},
			matches: false,
			name:    "resource mis-match",
		},
		{
			policy: policy{
				User:      "foo",
				Resource:  "foo",
				Namespace: "foo",
			},
			attr: authorizer.APIAttributesRecord{
				UserInfo: &user.DefaultInfo{
					Name: "foo",
				},
				Resource:  "foo",
				Namespace: "foo",
			},
			matches: true,
			name:    "namespace mis-match",
		},
		{
			policy: policy{
				Namespace: "foo",
			},
			attr: authorizer.APIAttributesRecord{
				Namespace: "bar",
			},
			matches: false,
			name:    "resource mis-match",
		},
	}
	for _, test := range tests {
		matches := test.policy.matchesAPIRequest(test.attr)
		if test.matches != matches {
			t.Errorf("unexpected value for %s, expected: %t, saw: %t", test.name, test.matches, matches)
		}
	}
}

func TestGenericPolicy(t *testing.T) {
	tests := []struct {
		policy  policy
		attr    authorizer.GenericAttributesRecord
		matches bool
		name    string
	}{
		{
			policy:  policy{},
			attr:    authorizer.GenericAttributesRecord{},
			matches: true,
			name:    "null",
		},
		{
			policy: policy{
				Readonly: true,
			},
			attr:    authorizer.GenericAttributesRecord{},
			matches: false,
			name:    "read-only mismatch",
		},
		{
			policy: policy{
				User: "foo",
			},
			attr: authorizer.GenericAttributesRecord{
				UserInfo: &user.DefaultInfo{
					Name: "bar",
				},
			},
			matches: false,
			name:    "user name mis-match",
		},
	}
	for _, test := range tests {
		matches := test.policy.matchesGenericRequest(test.attr)
		if test.matches != matches {
			t.Errorf("unexpected value for %s, expected: %t, saw: %t", test.name, test.matches, matches)
		}
	}
}
