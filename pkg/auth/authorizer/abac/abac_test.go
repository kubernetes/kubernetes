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

	uScheduler := user.DefaultInfo{Name: "scheduler", UID: "uid1"}
	uAlice := user.DefaultInfo{Name: "alice", UID: "uid3"}
	uChuck := user.DefaultInfo{Name: "chuck", UID: "uid5"}

	testCases := []struct {
		User        user.DefaultInfo
		RO          bool
		Kind        string
		NS          string
		ExpectAllow bool
	}{
		// Scheduler can read pods
		{User: uScheduler, RO: true, Kind: "pods", NS: "ns1", ExpectAllow: true},
		{User: uScheduler, RO: true, Kind: "pods", NS: "", ExpectAllow: true},
		// Scheduler cannot write pods
		{User: uScheduler, RO: false, Kind: "pods", NS: "ns1", ExpectAllow: false},
		{User: uScheduler, RO: false, Kind: "pods", NS: "", ExpectAllow: false},
		// Scheduler can write bindings
		{User: uScheduler, RO: true, Kind: "bindings", NS: "ns1", ExpectAllow: true},
		{User: uScheduler, RO: true, Kind: "bindings", NS: "", ExpectAllow: true},

		// Alice can read and write anything in the right namespace.
		{User: uAlice, RO: true, Kind: "pods", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: true, Kind: "widgets", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: true, Kind: "", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Kind: "pods", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Kind: "widgets", NS: "projectCaribou", ExpectAllow: true},
		{User: uAlice, RO: false, Kind: "", NS: "projectCaribou", ExpectAllow: true},
		// .. but not the wrong namespace.
		{User: uAlice, RO: true, Kind: "pods", NS: "ns1", ExpectAllow: false},
		{User: uAlice, RO: true, Kind: "widgets", NS: "ns1", ExpectAllow: false},
		{User: uAlice, RO: true, Kind: "", NS: "ns1", ExpectAllow: false},

		// Chuck can read events, since anyone can.
		{User: uChuck, RO: true, Kind: "events", NS: "ns1", ExpectAllow: true},
		{User: uChuck, RO: true, Kind: "events", NS: "", ExpectAllow: true},
		// Chuck can't do other things.
		{User: uChuck, RO: false, Kind: "events", NS: "ns1", ExpectAllow: false},
		{User: uChuck, RO: true, Kind: "pods", NS: "ns1", ExpectAllow: false},
		{User: uChuck, RO: true, Kind: "floop", NS: "ns1", ExpectAllow: false},
		// Chunk can't access things with no kind or namespace
		// TODO: find a way to give someone access to miscelaneous endpoints, such as
		// /healthz, /version, etc.
		{User: uChuck, RO: true, Kind: "", NS: "", ExpectAllow: false},
	}
	for _, tc := range testCases {
		attr := authorizer.AttributesRecord{
			User:      &tc.User,
			ReadOnly:  tc.RO,
			Kind:      tc.Kind,
			Namespace: tc.NS,
		}
		t.Logf("tc: %v -> attr %v", tc, attr)
		err := a.Authorize(attr)
		actualAllow := bool(err == nil)
		if tc.ExpectAllow != actualAllow {
			t.Errorf("Expected allowed=%v but actually allowed=%v, for case %v",
				tc.ExpectAllow, actualAllow, tc)
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
