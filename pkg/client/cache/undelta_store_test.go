/*
Copyright 2015 Google Inc. All rights reserved.

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

package cache

import (
	"reflect"
	"testing"
)

// store_test.go checks that UndeltaStore conforms to the Store interface
// behavior.  This test just tests that it calls the push func in addition.

type t struct{ int }

var (
	o1  interface{}   = t{1}
	o2  interface{}   = t{2}
	l1  []interface{} = []interface{}{t{1}}
	l12 []interface{} = []interface{}{t{1}, t{2}}
)

func TestUpdateCallsPush(t *testing.T) {
	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push)

	u.Add("a", o2)
	u.Update("a", o1)
	if callcount != 2 {
		t.Errorf("Expected 2 calls, got %d", callcount)
	}
	if !reflect.DeepEqual(l1, got) {
		t.Errorf("Expected %#v, Got %#v", l1, got)
	}
}

func TestDeleteCallsPush(t *testing.T) {
	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push)

	u.Add("a", o2)
	u.Delete("a")
	if callcount != 2 {
		t.Errorf("Expected 2 calls, got %d", callcount)
	}
	expected := []interface{}{}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}

func TestReadsDoNotCallPush(t *testing.T) {
	push := func(m []interface{}) {
		t.Errorf("Unexpected call to push!")
	}

	u := NewUndeltaStore(push)

	// These should not call push.
	_ = u.List()
	_ = u.ContainedIDs()
	_, _ = u.Get("1")
}

func TestReplaceCallsPush(t *testing.T) {
	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push)

	m := make(map[string]interface{})
	m["1"] = o1
	m["2"] = o2

	u.Replace(m)
	if callcount != 1 {
		t.Errorf("Expected 2 calls, got %d", callcount)
	}
	expected := l12
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}
