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

package util

import (
	"fmt"
	"testing"
	"time"
)

func TestUntil(t *testing.T) {
	ch := make(chan struct{})
	close(ch)
	Until(func() {
		t.Fatal("should not have been invoked")
	}, 0, ch)

	ch = make(chan struct{})
	called := make(chan struct{})
	go func() {
		Until(func() {
			called <- struct{}{}
		}, 0, ch)
		close(called)
	}()
	<-called
	close(ch)
	<-called
}

func TestUntilReturnsImmediately(t *testing.T) {
	now := time.Now()
	ch := make(chan struct{})
	Until(func() {
		close(ch)
	}, 30*time.Second, ch)
	if now.Add(25 * time.Second).Before(time.Now()) {
		t.Errorf("Until did not return immediately when the stop chan was closed inside the func")
	}
}

func TestHandleCrash(t *testing.T) {
	count := 0
	expect := 10
	for i := 0; i < expect; i = i + 1 {
		defer HandleCrash()
		if i%2 == 0 {
			panic("Test Panic")
		}
		count = count + 1
	}
	if count != expect {
		t.Errorf("Expected %d iterations, found %d", expect, count)
	}
}

func TestCustomHandleCrash(t *testing.T) {
	old := PanicHandlers
	defer func() { PanicHandlers = old }()
	var result interface{}
	PanicHandlers = []func(interface{}){
		func(r interface{}) {
			result = r
		},
	}
	func() {
		defer HandleCrash()
		panic("test")
	}()
	if result != "test" {
		t.Errorf("did not receive custom handler")
	}
}

func TestCustomHandleError(t *testing.T) {
	old := ErrorHandlers
	defer func() { ErrorHandlers = old }()
	var result error
	ErrorHandlers = []func(error){
		func(err error) {
			result = err
		},
	}
	err := fmt.Errorf("test")
	HandleError(err)
	if result != err {
		t.Errorf("did not receive custom handler")
	}
}

func TestStringDiff(t *testing.T) {
	diff := StringDiff("aaabb", "aaacc")
	expect := "aaa\n\nA: bb\n\nB: cc\n\n"
	if diff != expect {
		t.Errorf("diff returned %v", diff)
	}
}

func TestCompileRegex(t *testing.T) {
	uncompiledRegexes := []string{"endsWithMe$", "^startingWithMe"}
	regexes, err := CompileRegexps(uncompiledRegexes)

	if err != nil {
		t.Errorf("Failed to compile legal regexes: '%v': %v", uncompiledRegexes, err)
	}
	if len(regexes) != len(uncompiledRegexes) {
		t.Errorf("Wrong number of regexes returned: '%v': %v", uncompiledRegexes, regexes)
	}

	if !regexes[0].MatchString("Something that endsWithMe") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if regexes[0].MatchString("Something that doesn't endsWithMe.") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if !regexes[1].MatchString("startingWithMe is very important") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
	if regexes[1].MatchString("not startingWithMe should fail") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
}

func TestAllPtrFieldsNil(t *testing.T) {
	testCases := []struct {
		obj      interface{}
		expected bool
	}{
		{struct{}{}, true},
		{struct{ Foo int }{12345}, true},
		{&struct{ Foo int }{12345}, true},
		{struct{ Foo *int }{nil}, true},
		{&struct{ Foo *int }{nil}, true},
		{struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{&struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{&struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{struct{ Foo *int }{new(int)}, false},
		{&struct{ Foo *int }{new(int)}, false},
		{struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{&struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{(*struct{})(nil), true},
	}
	for i, tc := range testCases {
		if AllPtrFieldsNil(tc.obj) != tc.expected {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expected, !tc.expected)
		}
	}
}
