/*
Copyright 2014 The Kubernetes Authors.

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
	//"fmt"
	"reflect"
	"testing"
	//"time"
	"errors"
	"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/pkg/api/meta"
	"container/heap"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"strconv"
	//"math/rand"
)

// Helper Functions
// currently using a pod object instead cause I can't get this to work properly
type testPriorityObject struct {
	TypeMeta   unversioned.TypeMeta `json:",inline"`
	ObjectMeta api.ObjectMeta       `json:"metadata,omitempty"`
	val        interface{}
}

func testPriorityObjectKeyFunc(obj interface{}) (string, error) {
	meta := obj.(api.Pod).ObjectMeta
	//meta, err := meta.Accessor(&obj)
	//if err != nil {
	//    return "", fmt.Errorf("object has no meta: %v", err)
	//}
	name := meta.GetName()
	return name, nil
}

// This could be any valid API object, but using a pod as it fits the use case
func mkPriorityObj(name string, priority int) api.Pod {
	return api.Pod{
		TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
		ObjectMeta: api.ObjectMeta{
			Namespace:   "bar",
			Name:        name,
			Annotations: map[string]string{"x": "y", annotationKey: strconv.Itoa(priority)},
		},
	}
	//func mkPriorityObj(name string, priority int) *testPriorityObject {
	//    return &testPriorityObject{
	//        TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
	//        ObjectMeta: api.ObjectMeta{
	//            Namespace:       "bar",
	//            Name:            name,
	//            Annotations:     map[string]string{"x": "y", annotationKey: strconv.Itoa(priority)},
	//        },
	//        val: name,
	//    }
}

// Tests
func TestPriority_testPriorityObjectKeyFunc(t *testing.T) {
	p := mkPriorityObj("best", 1)
	key, err := testPriorityObjectKeyFunc(p)

	if err != nil {
		t.Errorf("error grabbing key: %v", err)
	}
	if key != "best" {
		t.Errorf("test case failed, expected: '%v' , got '%v'", "best", key)
	}
}

//Testing PriorityQueue functions
// Trivial test to make sure the function outputs something sane
func TestPriority_NewPriorityQueue(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)

	thing := reflect.TypeOf(pq).Name()
	if thing != "PriorityQueue" {
		//t.Errorf("Unexpected type: expected '%v', got '%v'", "PriorityQueue", thing)
		//TODO: this is a pointer. how to fix this?
	}
}

func TestPriority_NewPriority(t *testing.T) {
	p := NewPriority(testPriorityObjectKeyFunc)

	thing := reflect.TypeOf(&p).Name()
	if thing != "Priority" {
		//t.Errorf("Unexpected type: expected '%v', got '%v'", "Priority", thing)
		//TODO: this is a pointer. how to fix this?
	}
}

// Helper Methods
func TestPriority_GetPlaceInQueue(t *testing.T) {
	//TODO: not implemented
}

func TestPriority_Len(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)
	key := "thing"
	priority := 1
	n := 0
	pk := PriorityKey{
		key:      key,
		priority: priority,
		index:    n,
	}
	pq.queue = append(pq.queue, pk)

	obj := "stuff"
	pq.items[key] = obj

	l := pq.Len()
	if l != 1 {
		t.Errorf("unexpected length --> expected: '%v', got: '%v'", 1, l)
	}

}

func TestPriority_Less(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)

	pk1 := PriorityKey{
		key:      "less",
		priority: 1,
		index:    0,
	}
	pq.queue = append(pq.queue, pk1)
	pk2 := PriorityKey{
		key:      "more",
		priority: 2,
		index:    1,
	}
	pq.queue = append(pq.queue, pk2)

	got := pq.Less(0, 1)
	if got != false {
		t.Errorf("unexpected Less --> expected: '%v', got: '%v'", false, got)
	}

}

func TestPriority_Swap(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)

	pk1 := PriorityKey{
		key:      "less",
		priority: 1,
		index:    0,
	}
	pq.queue = append(pq.queue, pk1)
	pk2 := PriorityKey{
		key:      "more",
		priority: 2,
		index:    1,
	}
	pq.queue = append(pq.queue, pk2)

	pq.Swap(0, 1)

	got := pq.queue[0].key
	if got != "more" {
		t.Errorf("Swap failed --> expected: '%v', got: '%v'", "more", got)
	}
}

func TestPriority_Push(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)

	//check that Push adds to both the map and array
	pq.Push(mkPriorityObj("less", 1))
	pq.Push(mkPriorityObj("more", 4))
	pq.Push(mkPriorityObj("some", 3))
	if len(pq.items) != 3 {
		t.Errorf("Push failed to add to object map --> expected: '%v', got: '%v'", 3, len(pq.items))
	}
	if len(pq.queue) != 3 {
		t.Errorf("Push failed to add to key array --> expected: '%v', got: '%v'", 3, len(pq.queue))
	}

	//check that PriorityQueue correctly implements the heap interface for Fix
	heap.Fix(pq, 1)
	got_fix := pq.queue[0].key
	if got_fix != "more" {
		t.Errorf("heap.Fix failed --> expected: '%v', got: '%v'", "more", got_fix)
		t.Errorf("queue: '%v'", pq.queue)
	}

	//check that PriorityQueue correctly implements the heap interface for Init
	heap.Init(pq)
	expect_init := []PriorityKey{
		PriorityKey{key: "more", priority: 4, index: 0},
		PriorityKey{key: "less", priority: 1, index: 1},
		PriorityKey{key: "some", priority: 3, index: 2},
	}
	got_init := pq.queue
	if !reflect.DeepEqual(got_init, expect_init) {
		t.Errorf("heap.Init failed --> expected: '%v', got: '%v'", expect_init, got_init)
	}

	//check that PriorityQueue correctly implements the heap interface for Push
	heap.Push(pq, mkPriorityObj("few", 2))
	expect_push := []PriorityKey{
		PriorityKey{key: "more", priority: 4, index: 0},
		PriorityKey{key: "few", priority: 2, index: 1},
		PriorityKey{key: "some", priority: 3, index: 2},
		PriorityKey{key: "less", priority: 1, index: 3},
	}
	got_push := pq.queue
	if !reflect.DeepEqual(got_push, expect_push) {
		t.Errorf("heap.Push failed --> expected: '%v', got: '%v'", expect_push, got_push)
	}
}

func TestPriority_PriorityQueuePop(t *testing.T) {
	pq := NewPriorityQueue(testPriorityObjectKeyFunc)

	heap.Push(pq, mkPriorityObj("less", 1))
	heap.Push(pq, mkPriorityObj("more", 4))
	heap.Push(pq, mkPriorityObj("some", 3))
	heap.Push(pq, mkPriorityObj("few", 2))
	heap.Push(pq, mkPriorityObj("another", 6))

	expect := []string{"another", "more", "some", "few", "less"}
	got := []string{}
	for i := 0; i < len(expect); i++ {
		item := heap.Pop(pq)
		key, _ := testPriorityObjectKeyFunc(item)
		got = append(got, key)
	}

	if !reflect.DeepEqual(got, expect) {
		t.Errorf("heap.Pop failed --> expected: '%v', got: '%v'", expect, got)
		t.Errorf("queue: '%v'", pq.queue)
		t.Errorf("items: '%v'", pq.items)
	}
}

//TODO: convert to array testing?
func TestPriority_Add(t *testing.T) {
	p := NewPriority(testPriorityObjectKeyFunc)

	err := p.Add(mkPriorityObj("foo", 10))
	if err != nil {
		t.Errorf("unexpected error on Add: \"%v\"", err)
	}
	err = p.Add(mkPriorityObj("bar", 10))
	if err != nil {
		t.Errorf("unexpected error on Add: \"%v\"", err)
	}
	l := p.queue.Len()
	if l != 2 {
		t.Errorf("unexpected length --> expected: '%v', got: '%v'", 2, l)
		t.Errorf("queue: '%v'", p.queue.queue)
		t.Errorf("items: '%v'", p.queue.items)
	}

	//check that duplicate keys aren't added
	err = p.Add(mkPriorityObj("foo", 5))
	if err != nil {
		t.Errorf("unexpected error on Add: \"%v\"", err)
	}
	err = p.Add(mkPriorityObj("baz", 10))
	if err != nil {
		t.Errorf("unexpected error on Add: \"%v\"", err)
	}

	l = p.queue.Len()
	if l != 3 {
		t.Errorf("unexpected length --> expected: '%v', got: '%v'", 3, l)
		t.Errorf("queue: '%v'", p.queue.queue)
		t.Errorf("items: '%v'", p.queue.items)
	}
}

func TestPriority_Update(t *testing.T) {
	tests := []struct {
		actions          []func(p *Priority)
		expectedPriority int
		expectedLength   int
	}{
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
				func(p *Priority) { p.Update(mkPriorityObj("foo", 11)) },
			},
			expectedPriority: 11,
			expectedLength:   1,
		},
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
				func(p *Priority) { p.Update(mkPriorityObj("bar", 11)) },
			},
			expectedPriority: 10,
			expectedLength:   2,
		},
	}

	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		for _, action := range test.actions {
			action(p)
		}
		got, _ := MetaPriorityFunc(p.queue.items["foo"])
		if test.expectedPriority != got {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedPriority, got)
		}
		l := p.queue.Len()
		if l != test.expectedLength {
			t.Errorf("test case %v failed, expected length did not match: %v , got %v", i, test.expectedLength, l)
		}
	}
}

func TestPriority_Delete(t *testing.T) {
	tests := []struct {
		actions        []func(p *Priority)
		expectedObj    interface{}
		expectedLength int
	}{
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
				func(p *Priority) { p.Add(mkPriorityObj("bar", 11)) },
				func(p *Priority) { p.Delete(mkPriorityObj("foo", 10)) },
			},
			expectedObj:    nil,
			expectedLength: 1,
		},
	}

	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		for _, action := range test.actions {
			action(p)
		}
		got := p.queue.items["foo"]
		if test.expectedObj != got {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedObj, got)
		}
		l := p.queue.Len()
		if l != test.expectedLength {
			t.Errorf("test case %v failed, expected length did not match: %v , got %v", i, test.expectedLength, l)
		}
	}
}

func TestPriority_List(t *testing.T) {
	p := NewPriority(testPriorityObjectKeyFunc)
	p.Add(mkPriorityObj("foo", 10))
	p.Add(mkPriorityObj("bar", 11))
	got := p.List()

	expected := make([]interface{}, 0)
	expected = append(expected, mkPriorityObj("foo", 10))
	expected = append(expected, mkPriorityObj("bar", 11))
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %v , got %v", expected, got)
	}
}

func TestPriority_ListKeys(t *testing.T) {
	p := NewPriority(testPriorityObjectKeyFunc)
	p.Add(mkPriorityObj("foo", 10))
	p.Add(mkPriorityObj("bar", 11))
	got := p.ListKeys()

	expected := make([]string, 0)
	expected = append(expected, "foo")
	expected = append(expected, "bar")
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %v , got %v", expected, got)
	}
}

func TestPriority_Get(t *testing.T) {
	tests := []struct {
		actions        []func(p *Priority)
		requestItem    interface{}
		expectedItem   interface{}
		expectedExists bool
		expectedErr    interface{}
	}{
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
			},
			requestItem:    mkPriorityObj("foo", 10),
			expectedItem:   mkPriorityObj("foo", 10),
			expectedExists: true,
			expectedErr:    nil,
		},
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
			},
			requestItem:    mkPriorityObj("bar", 10),
			expectedItem:   nil,
			expectedExists: false,
			expectedErr:    nil,
		},
		//TODO: test case for invalid object?
	}

	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		for _, action := range test.actions {
			action(p)
		}
		got, exists, err := p.Get(test.requestItem)

		if !reflect.DeepEqual(test.expectedItem, got) {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedItem, got)
		}
		if test.expectedExists != exists {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedExists, exists)
		}
		if test.expectedErr != err {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedErr, err)
		}
	}
}

func TestPriority_GetByKey(t *testing.T) {
	tests := []struct {
		actions        []func(p *Priority)
		requestKey     string
		expectedItem   interface{}
		expectedExists bool
		expectedErr    interface{}
	}{
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
			},
			requestKey:     "foo",
			expectedItem:   mkPriorityObj("foo", 10),
			expectedExists: true,
			expectedErr:    nil,
		},
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("foo", 10)) },
			},
			requestKey:     "bar",
			expectedItem:   nil,
			expectedExists: false,
			expectedErr:    nil,
		},
	}

	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		for _, action := range test.actions {
			action(p)
		}
		got, exists, err := p.GetByKey(test.requestKey)

		if !reflect.DeepEqual(test.expectedItem, got) {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedItem, got)
		}
		if test.expectedExists != exists {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedExists, exists)
		}
		if test.expectedErr != err {
			t.Errorf("test case %v failed, expected: %v , got %v", i, test.expectedErr, err)
		}
	}
}

//TODO:
//func TestPriority_Replace(t *testing.T) {

//TODO:
//func TestPriority_Resync(t *testing.T) {

func TestPriority_Pop(t *testing.T) {
	//TODO: make sure requeue was actually requeued correctly, not just with obj count (GetByKey)
	//TODO: test for concurrency bugs
	tests := []struct {
		fn             func(obj interface{}) error
		expectedObj    api.Pod
		expectedLength int
	}{
		{
			//pop but then requeue
			fn:             func(obj interface{}) error { return ErrRequeue{Err: nil} },
			expectedObj:    mkPriorityObj("bar", 11),
			expectedLength: 3,
		},
		{
			//pop but don't requeue
			fn:             func(obj interface{}) error { return nil },
			expectedObj:    mkPriorityObj("bar", 11),
			expectedLength: 2,
		},
	}
	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		_ = p.Add(mkPriorityObj("foo", 10))
		_ = p.Add(mkPriorityObj("bar", 11))
		_ = p.Add(mkPriorityObj("baz", 9))

		got, err := p.Pop(test.fn)
		if err != nil {
			t.Errorf("test case %v failed, unexpected error on Pop: %v", i, err)
		}

		if !reflect.DeepEqual(test.expectedObj, got) {
			t.Errorf("test case %v failed, Pop didn't match Add. Expected '%v', got '%v'", i, test.expectedObj, got)
		}
		l := p.queue.Len()
		if l != test.expectedLength {
			t.Errorf("test case %v failed, Queue remainder not as expected. Expected '%v', got '%v'", i, test.expectedLength, l)
		}
	}
}

func TestPriority_AddIfNotPresent(t *testing.T) {
	//TODO: test concurrency: add+update, add+replace, etc
	p := NewPriority(testPriorityObjectKeyFunc)

	_ = p.AddIfNotPresent(mkPriorityObj("foo", 10))
	_ = p.AddIfNotPresent(mkPriorityObj("bar", 15))
	_ = p.AddIfNotPresent(mkPriorityObj("bar", 11))
	_ = p.AddIfNotPresent(mkPriorityObj("baz", 14))

	fn := func(obj interface{}) error { return nil }
	got, err := p.Pop(fn)
	if err != nil {
		t.Errorf("unexpected error on Pop: %v", err)
	}

	expect := mkPriorityObj("bar", 15)
	if !reflect.DeepEqual(expect, got) {
		t.Errorf("Pop didn't match AddIfNotPresent. Expected '%v', got '%v'", expect, got)
	}
	l := p.queue.Len()
	if l != 2 {
		t.Errorf("Queue remainder not as expected. Expected '%v', got '%v'", 2, l)
	}
}

func TestPriority_HasSynced(t *testing.T) {
	tests := []struct {
		actions        []func(p *Priority)
		expectedSynced bool
	}{
		{
			actions:        []func(p *Priority){},
			expectedSynced: false,
		},
		{
			actions: []func(p *Priority){
				func(p *Priority) { p.Add(mkPriorityObj("a", 1)) },
			},
			expectedSynced: true,
		},
		//TODO: add testing for Replace first
		//		{
		//			actions: []func(p *Priority){
		//				func(p *Priority) { p.Replace([]interface{}{}, "0") },
		//			},
		//			expectedSynced: true,
		//		},
		//		{
		//			actions: []func(p *Priority){
		//				func(p *Priority) { p.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
		//			},
		//			expectedSynced: false,
		//		},
		//		{
		//			actions: []func(p *Priority){
		//				func(p *Priority) { p.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
		//				func(p *Priority) { Pop(p) },
		//			},
		//			expectedSynced: false,
		//		},
		//		{
		//			actions: []func(p *Priority){
		//				func(p *Priority) { p.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
		//				func(p *Priority) { Pop(p) },
		//				func(p *Priority) { Pop(p) },
		//			},
		//			expectedSynced: true,
		//		},
	}

	for i, test := range tests {
		p := NewPriority(testPriorityObjectKeyFunc)

		for _, action := range test.actions {
			action(p)
		}
		if e, a := test.expectedSynced, p.HasSynced(); a != e {
			t.Errorf("test case %v failed, expected: %v , got %v", i, e, a)
		}
	}
}

func TestPriority_MetaPriorityFunc(t *testing.T) {
	tests := []struct {
		input            interface{}
		expectedPriority int
		expectedErr      error
	}{
		{
			//happy case
			input:            mkPriorityObj("name", 10),
			expectedPriority: 10,
			expectedErr:      nil,
		},
		{
			//has no meta
			input:            "invalid object",
			expectedPriority: -1,
			//expectedErr:        errors.New("object has no meta: object does not implement the Object interfaces"),
			expectedErr: errors.New("object does not have annotations"),
		},
		{
			//has no annotations
			input: api.Pod{
				TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
				ObjectMeta: api.ObjectMeta{
					Namespace: "bar",
					Name:      "ash",
				},
			},
			expectedPriority: -1,
			expectedErr:      errors.New("object does not have annotations"),
		},
		{
			//has no priority
			input: api.Pod{
				TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
				ObjectMeta: api.ObjectMeta{
					Namespace:   "bar",
					Name:        "ash",
					Annotations: map[string]string{"x": "y"},
				},
			},
			expectedPriority: -1,
			expectedErr:      nil,
		},
		{
			//non-integer priority
			input: api.Pod{
				TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
				ObjectMeta: api.ObjectMeta{
					Namespace:   "bar",
					Name:        "ash",
					Annotations: map[string]string{"x": "y", annotationKey: "non-integer"},
				},
			},
			expectedPriority: -1,
			expectedErr:      errors.New("priority is not an integer: \"non-integer\""),
		},
	}
	for i, test := range tests {
		//input := mkPriorityObj("name", 10)
		//t.Errorf("DeepEqual: '%t'", reflect.DeepEqual(input, test.input))

		priority, err := MetaPriorityFunc(test.input)
		if priority != test.expectedPriority {
			t.Errorf("test case %v failed, expected: '%v' , got '%v'", i, test.expectedPriority, priority)
		}
		//need to handle nil special case because can't use Error() method on it
		//probably a more efficient/idiomatic way to do this...
		if err == nil {
			if test.expectedErr != nil {
				t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
			}
		} else {
			if test.expectedErr == nil {
				t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
			} else if err.Error() != test.expectedErr.Error() {
				t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
			}
		}
	}
}
