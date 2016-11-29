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

package treecache

import (
	"testing"

	"github.com/skynetservices/skydns/msg"
)

func TestTreeCache(t *testing.T) {
	tc := NewTreeCache()

	{
		_, ok := tc.GetEntry("key1", "p1", "p2")
		if ok {
			t.Errorf("key should not exist")
		}
	}

	checkExists := func(key string, expectedSvc *msg.Service, path ...string) {
		svc, ok := tc.GetEntry(key, path...)
		if !ok {
			t.Fatalf("key %v should exist", key)
		}
		if svc := svc.(*msg.Service); svc != nil {
			if svc != expectedSvc {
				t.Errorf("value is not correct (%v != %v)", svc, expectedSvc)
			}
		} else {
			t.Errorf("entry is not of the right type: %T", svc)
		}
	}
	setEntryTC := []struct {
		key  string
		svc  *msg.Service
		fqdn string
		path []string
	}{
		{"key1", &msg.Service{}, "key1.p2.p1.", []string{"p1", "p2"}},
		{"key2", &msg.Service{}, "key2.p2.p1.", []string{"p1", "p2"}},
		{"key3", &msg.Service{}, "key3.p2.p1.", []string{"p1", "p3"}},
	}

	for _, testCase := range setEntryTC {
		tc.SetEntry(testCase.key, testCase.svc, testCase.fqdn, testCase.path...)
		checkExists(testCase.key, testCase.svc, testCase.path...)
	}

	wildcardTC := []struct {
		path  []string
		count int
	}{
		{[]string{"p1"}, 0},
		{[]string{"p1", "p2"}, 2},
		{[]string{"p1", "p3"}, 1},
		{[]string{"p1", "p2", "key1"}, 1},
		{[]string{"p1", "p2", "key2"}, 1},
		{[]string{"p1", "p2", "key3"}, 0},
		{[]string{"p1", "p3", "key3"}, 1},
		{[]string{"p1", "p2", "*"}, 2},
		{[]string{"p1", "*", "*"}, 3},
	}

	for _, testCase := range wildcardTC {
		services := tc.GetValuesForPathWithWildcards(testCase.path...)
		if len(services) != testCase.count {
			t.Fatalf("Expected %v services for path %v, got %v",
				testCase.count, testCase.path, len(services))
		}
	}

	// Delete some paths
	if !tc.DeletePath("p1", "p2") {
		t.Fatal("should delete path p2.p1.")
	}
	if _, ok := tc.GetEntry("key3", "p1", "p3"); !ok {
		t.Error("should not affect p3.p1.")
	}
	if tc.DeletePath("p1", "p2") {
		t.Fatalf("should not be able to delete p2.p1")
	}
	if !tc.DeletePath("p1", "p3") {
		t.Fatalf("should be able to delete p3.p1")
	}
	if tc.DeletePath("p1", "p3") {
		t.Fatalf("should not be able to delete p3.t1")
	}

	for _, testCase := range []struct {
		k string
		p []string
	}{
		{"key1", []string{"p1", "p2"}},
		{"key2", []string{"p1", "p2"}},
		{"key3", []string{"p1", "p3"}},
	} {
		if _, ok := tc.GetEntry(testCase.k, testCase.p...); ok {
			t.Error("path should not exist")
		}
	}
}

func TestTreeCacheSetSubCache(t *testing.T) {
	tc := NewTreeCache()

	m := &msg.Service{}

	branch := NewTreeCache()
	branch.SetEntry("key1", m, "key", "p2")

	tc.SetSubCache("p1", branch, "p0")

	if _, ok := tc.GetEntry("key1", "p0", "p1", "p2"); !ok {
		t.Errorf("should be able to get entry p0.p1.p2.key1")
	}
}

func TestTreeCacheSerialize(t *testing.T) {
	tc := NewTreeCache()
	tc.SetEntry("key1", &msg.Service{}, "key1.p2.p1.", "p1", "p2")

	const expected = `{
	"ChildNodes": {
		"p1": {
			"ChildNodes": {
				"p2": {
					"ChildNodes": {},
					"Entries": {
						"key1": {}
					}
				}
			},
			"Entries": {}
		}
	},
	"Entries": {}
}`

	actual, err := tc.Serialize()
	if err != nil {
	}

	if actual != expected {
		t.Errorf("expected %q, got %q", expected, actual)
	}
}
