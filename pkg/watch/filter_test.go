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

package watch

import (
	"reflect"
	"testing"
)

func TestFilter(t *testing.T) {
	table := []Event{
		{Added, testType("foo")},
		{Added, testType("bar")},
		{Added, testType("baz")},
		{Added, testType("qux")},
		{Added, testType("zoo")},
	}

	source := NewFake()
	filtered := Filter(source, func(e Event) (Event, bool) {
		return e, e.Object.(testType)[0] != 'b'
	})

	go func() {
		for _, item := range table {
			source.Action(item.Type, item.Object)
		}
		source.Stop()
	}()

	var got []string
	for {
		event, ok := <-filtered.ResultChan()
		if !ok {
			break
		}
		got = append(got, string(event.Object.(testType)))
	}

	if e, a := []string{"foo", "qux", "zoo"}, got; !reflect.DeepEqual(e, a) {
		t.Errorf("got %v, wanted %v", e, a)
	}
}

func TestFilterStop(t *testing.T) {
	source := NewFake()
	filtered := Filter(source, func(e Event) (Event, bool) {
		return e, e.Object.(testType)[0] != 'b'
	})

	go func() {
		source.Add(testType("foo"))
		filtered.Stop()
	}()

	var got []string
	for {
		event, ok := <-filtered.ResultChan()
		if !ok {
			break
		}
		got = append(got, string(event.Object.(testType)))
	}

	if e, a := []string{"foo"}, got; !reflect.DeepEqual(e, a) {
		t.Errorf("got %v, wanted %v", e, a)
	}
}
