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

package cache

import (
	"errors"
	"reflect"
	"testing"
	"time"
)

type testPair struct {
	id  string
	obj interface{}
}
type testEnumerator []testPair

func (t testEnumerator) Len() int { return len(t) }
func (t testEnumerator) Get(i int) (string, interface{}) {
	return t[i].id, t[i].obj
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
		s := NewStore()
		// This is a unit test for the sync function, hence the nil getFunc.
		p := NewPoller(nil, 0, s)
		for line, pairs := range item.steps {
			p.sync(testEnumerator(pairs))

			ids := s.Contains()
			for _, pair := range pairs {
				if !ids.Has(pair.id) {
					t.Errorf("%v, %v: expected to find entry for %v, but did not.", testCase, line, pair.id)
					continue
				}
				found, ok := s.Get(pair.id)
				if !ok {
					t.Errorf("%v, %v: unexpected absent entry for %v", testCase, line, pair.id)
					continue
				}
				if e, a := pair.obj, found; !reflect.DeepEqual(e, a) {
					t.Errorf("%v, %v: expected %v, got %v for %v", testCase, line, e, a, pair.id)
				}
			}
			if e, a := len(pairs), len(ids); e != a {
				t.Errorf("%v, %v: expected len %v, got %v", testCase, line, e, a)
			}
		}
	}
}

func TestPoller_Run(t *testing.T) {
	s := NewStore()
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
	}, time.Millisecond, s).Run()

	// The test here is that we get called at least count times.
	<-done

	// We never added anything, verify that.
	if e, a := 0, len(s.Contains()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}
