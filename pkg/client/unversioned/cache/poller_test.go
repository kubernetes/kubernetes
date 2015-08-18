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

package cache

import (
	"errors"
	"reflect"
	"testing"
	"time"
)

func testPairKeyFunc(obj interface{}) (string, error) {
	return obj.(testPair).id, nil
}

type testPair struct {
	id  string
	obj interface{}
}
type testEnumerator []testPair

func (t testEnumerator) Len() int { return len(t) }
func (t testEnumerator) Get(i int) interface{} {
	return t[i]
}

func TestPoller_sync(t *testing.T) {
	table := []struct {
		// each step simulates the list that a getFunc would receive.
		steps [][]testPair
	}{
		{
			steps: [][]testPair{
				{
					{"foo", "foo1"},
					{"bar", "bar1"},
					{"baz", "baz1"},
					{"qux", "qux1"},
				}, {
					{"foo", "foo2"},
					{"bar", "bar2"},
					{"qux", "qux2"},
				}, {
					{"bar", "bar3"},
					{"baz", "baz2"},
					{"qux", "qux3"},
				}, {
					{"qux", "qux4"},
				}, {
					{"foo", "foo3"},
				},
			},
		},
	}

	for testCase, item := range table {
		s := NewStore(testPairKeyFunc)
		// This is a unit test for the sync function, hence the nil getFunc.
		p := NewPoller(nil, 0, s)
		for line, pairs := range item.steps {
			p.sync(testEnumerator(pairs))

			list := s.List()
			for _, pair := range pairs {
				foundInList := false
				for _, listItem := range list {
					id, _ := testPairKeyFunc(listItem)
					if pair.id == id {
						foundInList = true
					}
				}
				if !foundInList {
					t.Errorf("%v, %v: expected to find list entry for %v, but did not.", testCase, line, pair.id)
					continue
				}
				found, ok, _ := s.Get(pair)
				if !ok {
					t.Errorf("%v, %v: unexpected absent entry for %v", testCase, line, pair.id)
					continue
				}
				if e, a := pair.obj, found.(testPair).obj; !reflect.DeepEqual(e, a) {
					t.Errorf("%v, %v: expected %v, got %v for %v", testCase, line, e, a, pair.id)
				}
			}
			if e, a := len(pairs), len(list); e != a {
				t.Errorf("%v, %v: expected len %v, got %v", testCase, line, e, a)
			}
		}
	}
}

func TestPoller_Run(t *testing.T) {
	stopCh := make(chan struct{})
	defer func() { stopCh <- struct{}{} }()
	s := NewStore(testPairKeyFunc)
	const count = 10
	var called = 0
	done := make(chan struct{})
	NewPoller(func() (Enumerator, error) {
		called++
		if called == count {
			close(done)
		}
		// test both error and regular returns.
		if called&1 == 0 {
			return testEnumerator{}, nil
		}
		return nil, errors.New("transient error")
	}, time.Millisecond, s).RunUntil(stopCh)

	// The test here is that we get called at least count times.
	<-done

	// We never added anything, verify that.
	if e, a := 0, len(s.List()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}
