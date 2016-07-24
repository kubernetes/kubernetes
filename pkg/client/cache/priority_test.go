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
	//"reflect"
	"testing"
	//"time"
    "errors"
    "k8s.io/kubernetes/pkg/api"
    //"k8s.io/kubernetes/pkg/api/meta"
    "k8s.io/kubernetes/pkg/api/unversioned"
    "strconv"
)

// Helper Functions
// currently using a pod object instead cause I can't get this to work properly
type testPriorityObject struct {
    TypeMeta    unversioned.TypeMeta `json:",inline"`
    ObjectMeta  api.ObjectMeta `json:"metadata,omitempty"`
    val         interface{}
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
            Namespace:       "bar",
            Name:            name,
            Annotations:     map[string]string{"x": "y", annotationKey: strconv.Itoa(priority)},
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
////TODO need to assign priority to an object before adding it.
////this is testing FIFO currently...
//func TestPriority_basic(t *testing.T) {
//	p := NewPriority(testPriorityObjectKeyFunc)
//	const amount = 10
//	go func() {
//		for i := 0; i < amount; i++ {
//			p.Add(mkPriorityObj(string([]rune{'a', rune(i)}), strconv.Itoa(i+1), i*2))
//		}
//	}()
//	go func() {
//		for u := uint64(0); u < amount; u++ {
//			f.Add(mkPriorityObj(string([]rune{'b', rune(u)}), strconv.Itoa(u+1), u*2))
//		}
//	}()
//
//	lastInt := int(0)
//	lastUint := uint64(0)
//	for i := 0; i < amount*2; i++ {
//		switch obj := Pop(f).(testPriorityObject).val.(type) {
//		case int:
//			if obj <= lastInt {
//				t.Errorf("got %v (int) out of order, last was %v", obj, lastInt)
//			}
//			lastInt = obj
//		case uint64:
//			if obj <= lastUint {
//				t.Errorf("got %v (uint) out of order, last was %v", obj, lastUint)
//			} else {
//				lastUint = obj
//			}
//		default:
//			t.Fatalf("unexpected type %#v", obj)
//		}
//	}
//}

//func TestPriority_AddPop(t *testing.T) {
//	p := NewPriority(testPriorityObjectKeyFunc)
//    obj := mkPriorityObj("foo", 10)
//	err := p.Add(obj)
//	if err != nil {
//		t.Fatalf("unexpected error on Add: \"%v\"", err)
//	}
//
//    got, err := p.Pop(func(obj interface{}) error { 
//            fmt.Println("Obj: %v", obj)
//            return ErrRequeue{Err: nil} 
//        })
//	if err != nil {
//		t.Fatalf("unexpected error on Pop: %v", err)
//	}
//
//    if !reflect.DeepEqual(obj, got) {
//		t.Fatalf("Pop didn't match Add. Expected '%v', got '%v'", obj, got)
//    }
//}

//func TestPriority_requeueOnPop(t *testing.T) {
//	//p := NewPriority(testPriorityObjectKeyFunc)
//
//	//p.Add(mkPriorityObj("foo", 10))
//	//_, err := p.Pop(func(obj interface{}) error {
//	//	if obj.(api.Pod).name != "foo" {
//	//		t.Fatalf("unexpected object: %#v", obj)
//	//	}
//	//	return ErrRequeue{Err: nil}
//	//})
//	//if err != nil {
//	//	t.Fatalf("unexpected error: %v", err)
//	//}
//	//if _, ok, err := p.GetByKey("foo"); !ok || err != nil {
//	//	t.Fatalf("object should have been requeued: %t %v", ok, err)
//	//}
//
//	//_, err = f.Pop(func(obj interface{}) error {
//	//	if obj.(testPriorityObject).name != "foo" {
//	//		t.Fatalf("unexpected object: %#v", obj)
//	//	}
//	//	return ErrRequeue{Err: fmt.Errorf("test error")}
//	//})
//	//if err == nil || err.Error() != "test error" {
//	//	t.Fatalf("unexpected error: %v", err)
//	//}
//	//if _, ok, err := f.GetByKey("foo"); !ok || err != nil {
//	//	t.Fatalf("object should have been requeued: %t %v", ok, err)
//	//}
//
//	//_, err = f.Pop(func(obj interface{}) error {
//	//	if obj.(testPriorityObject).name != "foo" {
//	//		t.Fatalf("unexpected object: %#v", obj)
//	//	}
//	//	return nil
//	//})
//	//if err != nil {
//	//	t.Fatalf("unexpected error: %v", err)
//	//}
//	//if _, ok, err := f.GetByKey("foo"); ok || err != nil {
//	//	t.Fatalf("object should have been removed: %t %v", ok, err)
//	//}
//}

//func TestPriority_addUpdate(t *testing.T) {
//	f := NewPriority(testPriorityObjectKeyFunc)
//	f.Add(mkPriorityObj("foo", 10))
//	f.Update(mkPriorityObj("foo", 15))
//
//	if e, a := []interface{}{mkPriorityObj("foo", 15)}, f.List(); !reflect.DeepEqual(e, a) {
//		t.Errorf("Expected %+v, got %+v", e, a)
//	}
//	if e, a := []string{"foo"}, f.ListKeys(); !reflect.DeepEqual(e, a) {
//		t.Errorf("Expected %+v, got %+v", e, a)
//	}
//
//	got := make(chan testPriorityObject, 2)
//	go func() {
//		for {
//			got <- Pop(f).(testPriorityObject)
//		}
//	}()
//
//	first := <-got
//	if e, a := 15, first.val; e != a {
//		t.Errorf("Didn't get updated value (%v), got %v", e, a)
//	}
//	select {
//	case unexpected := <-got:
//		t.Errorf("Got second value %v", unexpected.val)
//	case <-time.After(50 * time.Millisecond):
//	}
//	_, exists, _ := f.Get(mkPriorityObj("foo", ""))
//	if exists {
//		t.Errorf("item did not get removed")
//	}
//}
//
//func TestPriority_addReplace(t *testing.T) {
//	f := NewPriority(testPriorityObjectKeyFunc)
//	f.Add(mkPriorityObj("foo", 10))
//	f.Replace([]interface{}{mkPriorityObj("foo", 15)}, "15")
//	got := make(chan testPriorityObject, 2)
//	go func() {
//		for {
//			got <- Pop(f).(testPriorityObject)
//		}
//	}()
//
//	first := <-got
//	if e, a := 15, first.val; e != a {
//		t.Errorf("Didn't get updated value (%v), got %v", e, a)
//	}
//	select {
//	case unexpected := <-got:
//		t.Errorf("Got second value %v", unexpected.val)
//	case <-time.After(50 * time.Millisecond):
//	}
//	_, exists, _ := f.Get(mkPriorityObj("foo", ""))
//	if exists {
//		t.Errorf("item did not get removed")
//	}
//}
//
//func TestPriority_detectLineJumpers(t *testing.T) {
//	f := NewPriority(testPriorityObjectKeyFunc)
//
//	f.Add(mkPriorityObj("foo", 10))
//	f.Add(mkPriorityObj("bar", 1))
//	f.Add(mkPriorityObj("foo", 11))
//	f.Add(mkPriorityObj("foo", 13))
//	f.Add(mkPriorityObj("zab", 30))
//
//	if e, a := 13, Pop(f).(testPriorityObject).val; a != e {
//		t.Fatalf("expected %d, got %d", e, a)
//	}
//
//	f.Add(mkPriorityObj("foo", 14)) // ensure foo doesn't jump back in line
//
//	if e, a := 1, Pop(f).(testPriorityObject).val; a != e {
//		t.Fatalf("expected %d, got %d", e, a)
//	}
//
//	if e, a := 30, Pop(f).(testPriorityObject).val; a != e {
//		t.Fatalf("expected %d, got %d", e, a)
//	}
//
//	if e, a := 14, Pop(f).(testPriorityObject).val; a != e {
//		t.Fatalf("expected %d, got %d", e, a)
//	}
//}
//
//func TestPriority_addIfNotPresent(t *testing.T) {
//	f := NewPriority(testPriorityObjectKeyFunc)
//
//	f.Add(mkPriorityObj("a", 1))
//	f.Add(mkPriorityObj("b", 2))
//	f.AddIfNotPresent(mkPriorityObj("b", 3))
//	f.AddIfNotPresent(mkPriorityObj("c", 4))
//
//	if e, a := 3, len(f.items); a != e {
//		t.Fatalf("expected queue length %d, got %d", e, a)
//	}
//
//	expectedValues := []int{1, 2, 4}
//	for _, expected := range expectedValues {
//		if actual := Pop(f).(testPriorityObject).val; actual != expected {
//			t.Fatalf("expected value %d, got %d", expected, actual)
//		}
//	}
//}
//
////+1
//func TestPriority_HasSynced(t *testing.T) {
//	tests := []struct {
//		actions        []func(f *Priority)
//		expectedSynced bool
//	}{
//		{
//			actions:        []func(f *Priority){},
//			expectedSynced: false,
//		},
//		{
//			actions: []func(f *Priority){
//				func(f *Priority) { f.Add(mkPriorityObj("a", 1)) },
//			},
//			expectedSynced: true,
//		},
//		{
//			actions: []func(f *Priority){
//				func(f *Priority) { f.Replace([]interface{}{}, "0") },
//			},
//			expectedSynced: true,
//		},
//		{
//			actions: []func(f *Priority){
//				func(f *Priority) { f.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
//			},
//			expectedSynced: false,
//		},
//		{
//			actions: []func(f *Priority){
//				func(f *Priority) { f.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
//				func(f *Priority) { Pop(f) },
//			},
//			expectedSynced: false,
//		},
//		{
//			actions: []func(f *Priority){
//				func(f *Priority) { f.Replace([]interface{}{mkPriorityObj("a", 1), mkPriorityObj("b", 2)}, "0") },
//				func(f *Priority) { Pop(f) },
//				func(f *Priority) { Pop(f) },
//			},
//			expectedSynced: true,
//		},
//	}
//
//	for i, test := range tests {
//		f := NewPriority(testPriorityObjectKeyFunc)
//
//		for _, action := range test.actions {
//			action(f)
//		}
//		if e, a := test.expectedSynced, f.HasSynced(); a != e {
//			t.Errorf("test case %v failed, expected: %v , got %v", i, e, a)
//		}
//	}
//}
//
////https://github.com/kubernetes/kubernetes/blob/f2ddd60eb9e7e9e29f7a105a9a8fa020042e8e52/pkg/controller/lookup_cache.go#L28
//func mkTestAPIObjectWithPriority(priority string) interface{} {
//    p := api.Pod{
//        TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
//        ObjectMeta: api.ObjectMeta{
//            Namespace:       "bar",
//            Name:            "foo",
//            GenerateName:    "prefix",
//            UID:             "uid",
//            ResourceVersion: "1",
//            SelfLink:        "some/place/only/we/know",
//            Labels:          map[string]string{"foo": "bar"},
//            Annotations:     map[string]string{"x": "y", annotationKey: priority},
//            Finalizers: []string{
//                "finalizer.1",
//                "finalizer.2",
//            },
//        },
//    }
//    return p
//}
//
//func mkTestAPIObjectWithNoPriority() interface{} {
//    p := &api.Pod{
//        TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
//        ObjectMeta: api.ObjectMeta{
//            Namespace:       "bar",
//            Name:            "foo",
//            GenerateName:    "prefix",
//            UID:             "uid",
//            ResourceVersion: "1",
//            SelfLink:        "some/place/only/we/know",
//            Labels:          map[string]string{"foo": "bar"},
//            Annotations:     map[string]string{"x": "y"},
//            Finalizers: []string{
//                "finalizer.1",
//                "finalizer.2",
//            },
//        },
//    }
//    return p
//}
//
//func mkTestAPIObjectWithNoAnnotations() interface{} {
//    p := &api.Pod{
//        TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
//        ObjectMeta: api.ObjectMeta{
//            Namespace:       "bar",
//            Name:            "foo",
//            GenerateName:    "prefix",
//            UID:             "uid",
//            ResourceVersion: "1",
//            SelfLink:        "some/place/only/we/know",
//            Labels:          map[string]string{"foo": "bar"},
//            Finalizers: []string{
//                "finalizer.1",
//                "finalizer.2",
//            },
//        },
//    }
//    return p
//}

//+1
func TestPriority_MetaPriorityFunc(t *testing.T) {
    tests := []struct{
        input               interface{}
        expectedPriority    int
        expectedErr         error
    }{
        {
            //happy case
            input:              mkPriorityObj("name", 10),
            expectedPriority:   10,
            expectedErr:        nil,
        },
        {
            //has no meta
            input:              "invalid object",
            expectedPriority:   -1,
            //expectedErr:        errors.New("object has no meta: object does not implement the Object interfaces"),
            expectedErr:        errors.New("object does not have annotations"),
        },
        {
            //has no annotations
            input:              api.Pod{
                                    TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
                                    ObjectMeta: api.ObjectMeta{
                                        Namespace:       "bar",
                                        Name:            "ash",
                                    },
                                },
            expectedPriority:   -1,
            expectedErr:        errors.New("object does not have annotations"),
        },
        {
            //has no priority
            input:              api.Pod{
                                    TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
                                    ObjectMeta: api.ObjectMeta{
                                        Namespace:       "bar",
                                        Name:            "ash",
                                        Annotations:     map[string]string{"x": "y"},
                                    },
                                },
            expectedPriority:   -1,
            expectedErr:        nil,
        },
        {
            //non-integer priority
            input:              api.Pod{
                                    TypeMeta: unversioned.TypeMeta{APIVersion: "/a", Kind: "b"},
                                    ObjectMeta: api.ObjectMeta{
                                        Namespace:       "bar",
                                        Name:            "ash",
                                        Annotations:     map[string]string{"x": "y", annotationKey: "non-integer"},
                                    },
                                },
            expectedPriority:   -1,
            expectedErr:        errors.New("priority is not an integer: \"non-integer\""),
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
        if (err == nil) {
            if (test.expectedErr != nil) {
                t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
            }
        } else {
            if (test.expectedErr == nil) {
                t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
            } else if (err.Error() != test.expectedErr.Error()) {
                t.Errorf("test case %v failed, expected: '%#v', got '%#v'", i, test.expectedErr, err)
            }
        }
    }
}
