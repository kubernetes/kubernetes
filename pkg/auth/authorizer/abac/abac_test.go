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

package abac

import (
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
)

func TestEmptyFile(t *testing.T) {
	_, err := newWithContents(t, "")
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

func TestOneLineFileNoNewLine(t *testing.T) {
	_, err := newWithContents(t, `{"user":"scheduler",  "readonly": true, "resource": "pods", "namespace":"ns1"}`)
	if err != nil {
		t.Errorf("unable to read policy file: %v", err)
	}
}

func TestTwoLineFile(t *testing.T) {
	_, err := newWithContents(t, `{"user":"scheduler",  "readonly": true, "resource": "pods"}
{"user":"scheduler",  "readonly": true, "resource": "services"}
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

func TestNotAuthorized(t *testing.T) {
	a, err := newWithContents(t, `{                    "readonly": true, "resource": "events"   }
{"user":"scheduler", "readonly": true, "resource": "pods"     }
{"user":"scheduler",                   "resource": "bindings" }
{"user":"kubelet",   "readonly": true, "resource": "bindings" }
{"user":"kubelet",                     "resource": "events"   }
{"user":"alice",                                              "namespace": "projectCaribou"}
{"user":"bob",       "readonly": true,                        "namespace": "projectCaribou"}
`)
	if err != nil {
		t.Fatalf("unable to read policy file: %v", err)
	}

	uScheduler := user.DefaultInfo{Name: "scheduler", UID: "uid1"}
	uAlice := user.DefaultInfo{Name: "alice", UID: "uid3"}
	uChuck := user.DefaultInfo{Name: "chuck", UID: "uid5"}

	testCases := []struct {
		User        user.DefaultInfo
		RO          bool
		Resource    string
		NS          string
		ExpectAllow bool
	}{
		// Scheduler can read pods
		{User: uScheduler, RO: true, Resource: "pods", NS: "ns1", ExpectAllow: true},
		{User: uScheduler, RO: true, Resource: "pods", NS: "", ExpectAllow: true},
		// Scheduler cannot write pods
		{User: uScheduler, RO: false, Resource: "pods", NS: "ns1", ExpectAllow: false},
		{User: uScheduler, RO: false, Resource: "pods", NS: "", ExpectAllow: false},
		// Scheduler can write bindings
		{User: uScheduler, RO: true, Resource: "bindings", NS: "ns1", ExpectAllow: true},
		{User: uScheduler, RO: true, Resource: "bindings", NS: "", ExpectAllow: true},

		// Alice can read and write anything in the right namespace.
		{User: uAlice, RO: true, Resource: "pods", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: true, Resource: "widgets", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: true, Resource: "", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Resource: "pods", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Resource: "widgets", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Resource: "", NS: "projectCaribou", ExpectAllow: true},
		// .. but not the wrong namespace.
		{User: uAlice, RO: true, Resource: "pods", NS: "ns1", ExpectAllow: false},
		{User: uAlice, RO: true, Resource: "widgets", NS: "ns1", ExpectAllow: false},
		{User: uAlice, RO: true, Resource: "", NS: "ns1", ExpectAllow: false},

		// Chuck can read events, since anyone can.
		{User: uChuck, RO: true, Resource: "events", NS: "ns1", ExpectAllow: true},
		{User: uChuck, RO: true, Resource: "events", NS: "", ExpectAllow: true},
		// Chuck can't do other things.
		{User: uChuck, RO: false, Resource: "events", NS: "ns1", ExpectAllow: false},
		{User: uChuck, RO: true, Resource: "pods", NS: "ns1", ExpectAllow: false},
		{User: uChuck, RO: true, Resource: "floop", NS: "ns1", ExpectAllow: false},
		// Chunk can't access things with no kind or namespace
		// TODO: find a way to give someone access to miscellaneous endpoints, such as
		// /healthz, /version, etc.
		{User: uChuck, RO: true, Resource: "", NS: "", ExpectAllow: false},
	}
	for i, tc := range testCases {
		attr := authorizer.AttributesRecord{
			User:      &tc.User,
			ReadOnly:  tc.RO,
			Resource:  tc.Resource,
			Namespace: tc.NS,
		}
		t.Logf("tc: %v -> attr %v", tc, attr)
		err := a.Authorize(attr)
		actualAllow := bool(err == nil)
		if tc.ExpectAllow != actualAllow {
			t.Errorf("%d: Expected allowed=%v but actually allowed=%v\n\t%v",
				i, tc.ExpectAllow, actualAllow, tc)
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
		attr := authorizer.AttributesRecord{
			User: &tc.User,
		}
		actualMatch := policy{User: tc.PolicyUser, Group: tc.PolicyGroup}.subjectMatches(attr)
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

func TestPolicy(t *testing.T) {
	tests := []struct {
		policy  policy
		attr    authorizer.Attributes
		matches bool
		name    string
	}{
		{
			policy:  policy{},
			attr:    authorizer.AttributesRecord{},
			matches: true,
			name:    "null",
		},
		{
			policy: policy{
				Readonly: true,
			},
			attr:    authorizer.AttributesRecord{},
			matches: false,
			name:    "read-only mismatch",
		},
		{
			policy: policy{
				User: "foo",
			},
			attr: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
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
			attr: authorizer.AttributesRecord{
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
			attr: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
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
			attr: authorizer.AttributesRecord{
				Namespace: "bar",
			},
			matches: false,
			name:    "resource mis-match",
		},
	}
	for _, test := range tests {
		matches := test.policy.matches(test.attr)
		if test.matches != matches {
			t.Errorf("unexpected value for %s, expected: %t, saw: %t", test.name, test.matches, matches)
		}
	}
}
