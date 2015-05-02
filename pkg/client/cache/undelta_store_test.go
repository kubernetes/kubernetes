/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

type testUndeltaObject struct {
	name string
	val  interface{}
}

func testUndeltaKeyFunc(obj interface{}) (string, error) {
	return obj.(testUndeltaObject).name, nil
}

/*
var (
	o1 interface{}   = t{1}
	o2 interface{}   = t{2}
	l1 []interface{} = []interface{}{t{1}}
)
*/

func TestUpdateCallsPush(t *testing.T) {
	mkObj := func(name string, val interface{}) testUndeltaObject {
		return testUndeltaObject{name: name, val: val}
	}

	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push, testUndeltaKeyFunc)

	u.Add(mkObj("a", 2))
	u.Update(mkObj("a", 1))
	if callcount != 2 {
		t.Errorf("Expected 2 calls, got %d", callcount)
	}

	l := []interface{}{mkObj("a", 1)}
	if !reflect.DeepEqual(l, got) {
		t.Errorf("Expected %#v, Got %#v", l, got)
	}
}

func TestDeleteCallsPush(t *testing.T) {
	mkObj := func(name string, val interface{}) testUndeltaObject {
		return testUndeltaObject{name: name, val: val}
	}

	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push, testUndeltaKeyFunc)

	u.Add(mkObj("a", 2))
	u.Delete(mkObj("a", ""))
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

	u := NewUndeltaStore(push, testUndeltaKeyFunc)

	// These should not call push.
	_ = u.List()
	_, _, _ = u.Get(testUndeltaObject{"a", ""})
}

func TestReplaceCallsPush(t *testing.T) {
	mkObj := func(name string, val interface{}) testUndeltaObject {
		return testUndeltaObject{name: name, val: val}
	}

	var got []interface{}
	var callcount int = 0
	push := func(m []interface{}) {
		callcount++
		got = m
	}

	u := NewUndeltaStore(push, testUndeltaKeyFunc)

	m := []interface{}{mkObj("a", 1)}

	u.Replace(m)
	if callcount != 1 {
		t.Errorf("Expected 1 calls, got %d", callcount)
	}
	expected := []interface{}{mkObj("a", 1)}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("Expected %#v, Got %#v", expected, got)
	}
}
