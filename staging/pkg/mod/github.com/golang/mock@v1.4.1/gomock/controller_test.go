// Copyright 2011 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gomock_test

import (
	"fmt"
	"reflect"
	"testing"

	"strings"

	"github.com/golang/mock/gomock"
)

type ErrorReporter struct {
	t          *testing.T
	log        []string
	failed     bool
	fatalToken struct{}
}

func NewErrorReporter(t *testing.T) *ErrorReporter {
	return &ErrorReporter{t: t}
}

func (e *ErrorReporter) reportLog() {
	for _, entry := range e.log {
		e.t.Log(entry)
	}
}

func (e *ErrorReporter) assertPass(msg string) {
	if e.failed {
		e.t.Errorf("Expected pass, but got failure(s): %s", msg)
		e.reportLog()
	}
}

func (e *ErrorReporter) assertFail(msg string) {
	if !e.failed {
		e.t.Errorf("Expected failure, but got pass: %s", msg)
	}
}

// Use to check that code triggers a fatal test failure.
func (e *ErrorReporter) assertFatal(fn func(), expectedErrMsgs ...string) {
	defer func() {
		err := recover()
		if err == nil {
			var actual string
			if e.failed {
				actual = "non-fatal failure"
			} else {
				actual = "pass"
			}
			e.t.Error("Expected fatal failure, but got a", actual)
		} else if token, ok := err.(*struct{}); ok && token == &e.fatalToken {
			// This is okay - the panic is from Fatalf().
			if expectedErrMsgs != nil {
				// assert that the actual error message
				// contains expectedErrMsgs

				// check the last actualErrMsg, because the previous messages come from previous errors
				actualErrMsg := e.log[len(e.log)-1]
				for _, expectedErrMsg := range expectedErrMsgs {
					if !strings.Contains(actualErrMsg, expectedErrMsg) {
						e.t.Errorf("Error message:\ngot: %q\nwant to contain: %q\n", actualErrMsg, expectedErrMsg)
					}
				}
			}
			return
		} else {
			// Some other panic.
			panic(err)
		}
	}()

	fn()
}

// recoverUnexpectedFatal can be used as a deferred call in test cases to
// recover from and display a call to ErrorReporter.Fatalf().
func (e *ErrorReporter) recoverUnexpectedFatal() {
	err := recover()
	if err == nil {
		// No panic.
	} else if token, ok := err.(*struct{}); ok && token == &e.fatalToken {
		// Unexpected fatal error happened.
		e.t.Error("Got unexpected fatal error(s). All errors up to this point:")
		e.reportLog()
		return
	} else {
		// Some other panic.
		panic(err)
	}
}

func (e *ErrorReporter) Logf(format string, args ...interface{}) {
	e.log = append(e.log, fmt.Sprintf(format, args...))
}

func (e *ErrorReporter) Errorf(format string, args ...interface{}) {
	e.Logf(format, args...)
	e.failed = true
}

func (e *ErrorReporter) Fatalf(format string, args ...interface{}) {
	e.Logf(format, args...)
	e.failed = true
	panic(&e.fatalToken)
}

type HelperReporter struct {
	gomock.TestReporter
	helper int
}

func (h *HelperReporter) Helper() {
	h.helper++
}

// A type purely for use as a receiver in testing the Controller.
type Subject struct{}

func (s *Subject) FooMethod(arg string) int {
	return 0
}

func (s *Subject) BarMethod(arg string) int {
	return 0
}

func (s *Subject) VariadicMethod(arg int, vararg ...string) {}

// A type purely for ActOnTestStructMethod
type TestStruct struct {
	Number  int
	Message string
}

func (s *Subject) ActOnTestStructMethod(arg TestStruct, arg1 int) int {
	return 0
}

func (s *Subject) SetArgMethod(sliceArg []byte, ptrArg *int) {}

func assertEqual(t *testing.T, expected interface{}, actual interface{}) {
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected %+v, but got %+v", expected, actual)
	}
}

func createFixtures(t *testing.T) (reporter *ErrorReporter, ctrl *gomock.Controller) {
	// reporter acts as a testing.T-like object that we pass to the
	// Controller. We use it to test that the mock considered tests
	// successful or failed.
	reporter = NewErrorReporter(t)
	ctrl = gomock.NewController(reporter)
	return
}

func TestNoCalls(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	ctrl.Finish()
	reporter.assertPass("No calls expected or made.")
}

func TestNoRecordedCallsForAReceiver(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	reporter.assertFatal(func() {
		ctrl.Call(subject, "NotRecordedMethod", "argument")
	}, "Unexpected call to", "there are no expected calls of the method \"NotRecordedMethod\" for that receiver")
	ctrl.Finish()
}

func TestNoRecordedMatchingMethodNameForAReceiver(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	ctrl.RecordCall(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		ctrl.Call(subject, "NotRecordedMethod", "argument")
	}, "Unexpected call to", "there are no expected calls of the method \"NotRecordedMethod\" for that receiver")
	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

// This tests that a call with an arguments of some primitive type matches a recorded call.
func TestExpectedMethodCall(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	ctrl.RecordCall(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Finish()

	reporter.assertPass("Expected method call made.")
}

func TestUnexpectedMethodCall(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	reporter.assertFatal(func() {
		ctrl.Call(subject, "FooMethod", "argument")
	})

	ctrl.Finish()
}

func TestRepeatedCall(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	ctrl.RecordCall(subject, "FooMethod", "argument").Times(3)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	reporter.assertPass("After expected repeated method calls.")
	reporter.assertFatal(func() {
		ctrl.Call(subject, "FooMethod", "argument")
	})
	ctrl.Finish()
	reporter.assertFail("After calling one too many times.")
}

func TestUnexpectedArgCount(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subject := new(Subject)

	ctrl.RecordCall(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		// This call is made with the wrong number of arguments...
		ctrl.Call(subject, "FooMethod", "argument", "extra_argument")
	}, "Unexpected call to", "wrong number of arguments", "Got: 2, want: 1")
	reporter.assertFatal(func() {
		// ... so is this.
		ctrl.Call(subject, "FooMethod")
	}, "Unexpected call to", "wrong number of arguments", "Got: 0, want: 1")
	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

// This tests that a call with complex arguments (a struct and some primitive type) matches a recorded call.
func TestExpectedMethodCall_CustomStruct(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	expectedArg0 := TestStruct{Number: 123, Message: "hello"}
	ctrl.RecordCall(subject, "ActOnTestStructMethod", expectedArg0, 15)
	ctrl.Call(subject, "ActOnTestStructMethod", expectedArg0, 15)

	reporter.assertPass("Expected method call made.")
}

func TestUnexpectedArgValue_FirstArg(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subject := new(Subject)

	expectedArg0 := TestStruct{Number: 123, Message: "hello"}
	ctrl.RecordCall(subject, "ActOnTestStructMethod", expectedArg0, 15)

	reporter.assertFatal(func() {
		// the method argument (of TestStruct type) has 1 unexpected value (for the Message field)
		ctrl.Call(subject, "ActOnTestStructMethod", TestStruct{Number: 123, Message: "no message"}, 15)
	}, "Unexpected call to", "doesn't match the argument at index 0",
		"Got: {123 no message}\nWant: is equal to {123 hello}")

	reporter.assertFatal(func() {
		// the method argument (of TestStruct type) has 2 unexpected values (for both fields)
		ctrl.Call(subject, "ActOnTestStructMethod", TestStruct{Number: 11, Message: "no message"}, 15)
	}, "Unexpected call to", "doesn't match the argument at index 0",
		"Got: {11 no message}\nWant: is equal to {123 hello}")

	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

func TestUnexpectedArgValue_SecondArg(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subject := new(Subject)

	expectedArg0 := TestStruct{Number: 123, Message: "hello"}
	ctrl.RecordCall(subject, "ActOnTestStructMethod", expectedArg0, 15)

	reporter.assertFatal(func() {
		ctrl.Call(subject, "ActOnTestStructMethod", TestStruct{Number: 123, Message: "hello"}, 3)
	}, "Unexpected call to", "doesn't match the argument at index 1",
		"Got: 3\nWant: is equal to 15")

	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

func TestUnexpectedArgValue_WantFormatter(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subject := new(Subject)

	expectedArg0 := TestStruct{Number: 123, Message: "hello"}
	ctrl.RecordCall(
		subject,
		"ActOnTestStructMethod",
		expectedArg0,
		gomock.WantFormatter(
			gomock.StringerFunc(func() string { return "is equal to fifteen" }),
			gomock.Eq(15),
		),
	)

	reporter.assertFatal(func() {
		ctrl.Call(subject, "ActOnTestStructMethod", TestStruct{Number: 123, Message: "hello"}, 3)
	}, "Unexpected call to", "doesn't match the argument at index 1",
		"Got: 3\nWant: is equal to fifteen")

	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

func TestUnexpectedArgValue_GotFormatter(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subject := new(Subject)

	expectedArg0 := TestStruct{Number: 123, Message: "hello"}
	ctrl.RecordCall(
		subject,
		"ActOnTestStructMethod",
		expectedArg0,
		gomock.GotFormatterAdapter(
			gomock.GotFormatterFunc(func(i interface{}) string {
				// Leading 0s
				return fmt.Sprintf("%02d", i)
			}),
			gomock.Eq(15),
		),
	)

	reporter.assertFatal(func() {
		ctrl.Call(subject, "ActOnTestStructMethod", TestStruct{Number: 123, Message: "hello"}, 3)
	}, "Unexpected call to", "doesn't match the argument at index 1",
		"Got: 03\nWant: is equal to 15")

	reporter.assertFatal(func() {
		// The expected call wasn't made.
		ctrl.Finish()
	})
}

func TestAnyTimes(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)

	ctrl.RecordCall(subject, "FooMethod", "argument").AnyTimes()
	for i := 0; i < 100; i++ {
		ctrl.Call(subject, "FooMethod", "argument")
	}
	reporter.assertPass("After 100 method calls.")
	ctrl.Finish()
}

func TestMinTimes1(t *testing.T) {
	// It fails if there are no calls
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(1)
	reporter.assertFatal(func() {
		ctrl.Finish()
	})

	// It succeeds if there is one call
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(1)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Finish()

	// It succeeds if there are many calls
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(1)
	for i := 0; i < 100; i++ {
		ctrl.Call(subject, "FooMethod", "argument")
	}
	ctrl.Finish()
}

func TestMaxTimes1(t *testing.T) {
	// It succeeds if there are no calls
	_, ctrl := createFixtures(t)
	subject := new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MaxTimes(1)
	ctrl.Finish()

	// It succeeds if there is one call
	_, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MaxTimes(1)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Finish()

	// It fails if there are more
	reporter, ctrl := createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MaxTimes(1)
	ctrl.Call(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		ctrl.Call(subject, "FooMethod", "argument")
	})
	ctrl.Finish()
}

func TestMinMaxTimes(t *testing.T) {
	// It fails if there are less calls than specified
	reporter, ctrl := createFixtures(t)
	subject := new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(2).MaxTimes(2)
	ctrl.Call(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		ctrl.Finish()
	})

	// It fails if there are more calls than specified
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(2).MaxTimes(2)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		ctrl.Call(subject, "FooMethod", "argument")
	})

	// It succeeds if there is just the right number of calls
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MaxTimes(2).MinTimes(2)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Finish()

	// If MaxTimes is called after MinTimes is called with 1, MaxTimes takes precedence.
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MinTimes(1).MaxTimes(2)
	ctrl.Call(subject, "FooMethod", "argument")
	ctrl.Call(subject, "FooMethod", "argument")
	reporter.assertFatal(func() {
		ctrl.Call(subject, "FooMethod", "argument")
	})

	// If MinTimes is called after MaxTimes is called with 1, MinTimes takes precedence.
	reporter, ctrl = createFixtures(t)
	subject = new(Subject)
	ctrl.RecordCall(subject, "FooMethod", "argument").MaxTimes(1).MinTimes(2)
	for i := 0; i < 100; i++ {
		ctrl.Call(subject, "FooMethod", "argument")
	}
	ctrl.Finish()
}

func TestDo(t *testing.T) {
	_, ctrl := createFixtures(t)
	subject := new(Subject)

	doCalled := false
	var argument string
	ctrl.RecordCall(subject, "FooMethod", "argument").Do(
		func(arg string) {
			doCalled = true
			argument = arg
		})
	if doCalled {
		t.Error("Do() callback called too early.")
	}

	ctrl.Call(subject, "FooMethod", "argument")

	if !doCalled {
		t.Error("Do() callback not called.")
	}
	if "argument" != argument {
		t.Error("Do callback received wrong argument.")
	}

	ctrl.Finish()
}

func TestDoAndReturn(t *testing.T) {
	_, ctrl := createFixtures(t)
	subject := new(Subject)

	doCalled := false
	var argument string
	ctrl.RecordCall(subject, "FooMethod", "argument").DoAndReturn(
		func(arg string) int {
			doCalled = true
			argument = arg
			return 5
		})
	if doCalled {
		t.Error("Do() callback called too early.")
	}

	rets := ctrl.Call(subject, "FooMethod", "argument")

	if !doCalled {
		t.Error("Do() callback not called.")
	}
	if "argument" != argument {
		t.Error("Do callback received wrong argument.")
	}
	if len(rets) != 1 {
		t.Fatalf("Return values from Call: got %d, want 1", len(rets))
	}
	if ret, ok := rets[0].(int); !ok {
		t.Fatalf("Return value is not an int")
	} else if ret != 5 {
		t.Errorf("DoAndReturn return value: got %d, want 5", ret)
	}

	ctrl.Finish()
}

func TestSetArgSlice(t *testing.T) {
	_, ctrl := createFixtures(t)
	subject := new(Subject)

	var in = []byte{4, 5, 6}
	var set = []byte{1, 2, 3}
	ctrl.RecordCall(subject, "SetArgMethod", in, nil).SetArg(0, set)
	ctrl.Call(subject, "SetArgMethod", in, nil)

	if !reflect.DeepEqual(in, set) {
		t.Error("Expected SetArg() to modify input slice argument")
	}

	ctrl.Finish()
}

func TestSetArgPtr(t *testing.T) {
	_, ctrl := createFixtures(t)
	subject := new(Subject)

	var in int = 43
	const set = 42
	ctrl.RecordCall(subject, "SetArgMethod", nil, &in).SetArg(1, set)
	ctrl.Call(subject, "SetArgMethod", nil, &in)

	if in != set {
		t.Error("Expected SetArg() to modify value pointed to by argument")
	}

	ctrl.Finish()
}

func TestReturn(t *testing.T) {
	_, ctrl := createFixtures(t)
	subject := new(Subject)

	// Unspecified return should produce "zero" result.
	ctrl.RecordCall(subject, "FooMethod", "zero")
	ctrl.RecordCall(subject, "FooMethod", "five").Return(5)

	assertEqual(
		t,
		[]interface{}{0},
		ctrl.Call(subject, "FooMethod", "zero"))

	assertEqual(
		t,
		[]interface{}{5},
		ctrl.Call(subject, "FooMethod", "five"))
	ctrl.Finish()
}

func TestUnorderedCalls(t *testing.T) {
	reporter, ctrl := createFixtures(t)
	defer reporter.recoverUnexpectedFatal()
	subjectTwo := new(Subject)
	subjectOne := new(Subject)

	ctrl.RecordCall(subjectOne, "FooMethod", "1")
	ctrl.RecordCall(subjectOne, "BarMethod", "2")
	ctrl.RecordCall(subjectTwo, "FooMethod", "3")
	ctrl.RecordCall(subjectTwo, "BarMethod", "4")

	// Make the calls in a different order, which should be fine.
	ctrl.Call(subjectOne, "BarMethod", "2")
	ctrl.Call(subjectTwo, "FooMethod", "3")
	ctrl.Call(subjectTwo, "BarMethod", "4")
	ctrl.Call(subjectOne, "FooMethod", "1")

	reporter.assertPass("After making all calls in different order")

	ctrl.Finish()

	reporter.assertPass("After finish")
}

func commonTestOrderedCalls(t *testing.T) (reporter *ErrorReporter, ctrl *gomock.Controller, subjectOne, subjectTwo *Subject) {
	reporter, ctrl = createFixtures(t)

	subjectOne = new(Subject)
	subjectTwo = new(Subject)

	gomock.InOrder(
		ctrl.RecordCall(subjectOne, "FooMethod", "1").AnyTimes(),
		ctrl.RecordCall(subjectTwo, "FooMethod", "2"),
		ctrl.RecordCall(subjectTwo, "BarMethod", "3"),
	)

	return
}

func TestOrderedCallsCorrect(t *testing.T) {
	reporter, ctrl, subjectOne, subjectTwo := commonTestOrderedCalls(t)

	ctrl.Call(subjectOne, "FooMethod", "1")
	ctrl.Call(subjectTwo, "FooMethod", "2")
	ctrl.Call(subjectTwo, "BarMethod", "3")

	ctrl.Finish()

	reporter.assertPass("After finish")
}

func TestOrderedCallsInCorrect(t *testing.T) {
	reporter, ctrl, subjectOne, subjectTwo := commonTestOrderedCalls(t)

	ctrl.Call(subjectOne, "FooMethod", "1")
	reporter.assertFatal(func() {
		// FooMethod(2) should be called before BarMethod(3)
		ctrl.Call(subjectTwo, "BarMethod", "3")
	}, "Unexpected call to", "Subject.BarMethod([3])", "doesn't have a prerequisite call satisfied")
}

// Test that calls that are prerequisites to other calls but have maxCalls >
// minCalls are removed from the expected call set.
func TestOrderedCallsWithPreReqMaxUnbounded(t *testing.T) {
	reporter, ctrl, subjectOne, subjectTwo := commonTestOrderedCalls(t)

	// Initially we should be able to call FooMethod("1") as many times as we
	// want.
	ctrl.Call(subjectOne, "FooMethod", "1")
	ctrl.Call(subjectOne, "FooMethod", "1")

	// But calling something that has it as a prerequite should remove it from
	// the expected call set. This allows tests to ensure that FooMethod("1") is
	// *not* called after FooMethod("2").
	ctrl.Call(subjectTwo, "FooMethod", "2")

	// Therefore this call should fail:
	reporter.assertFatal(func() {
		ctrl.Call(subjectOne, "FooMethod", "1")
	})
}

func TestCallAfterLoopPanic(t *testing.T) {
	_, ctrl := createFixtures(t)

	subject := new(Subject)

	firstCall := ctrl.RecordCall(subject, "FooMethod", "1")
	secondCall := ctrl.RecordCall(subject, "FooMethod", "2")
	thirdCall := ctrl.RecordCall(subject, "FooMethod", "3")

	gomock.InOrder(firstCall, secondCall, thirdCall)

	defer func() {
		err := recover()
		if err == nil {
			t.Error("Call.After creation of dependency loop did not panic.")
		}
	}()

	// This should panic due to dependency loop.
	firstCall.After(thirdCall)
}

func TestPanicOverridesExpectationChecks(t *testing.T) {
	ctrl := gomock.NewController(t)
	reporter := NewErrorReporter(t)

	reporter.assertFatal(func() {
		ctrl.RecordCall(new(Subject), "FooMethod", "1")
		defer ctrl.Finish()
		reporter.Fatalf("Intentional panic")
	})
}

func TestSetArgWithBadType(t *testing.T) {
	rep, ctrl := createFixtures(t)
	defer ctrl.Finish()

	s := new(Subject)
	// This should catch a type error:
	rep.assertFatal(func() {
		ctrl.RecordCall(s, "FooMethod", "1").SetArg(0, "blah")
	})
	ctrl.Call(s, "FooMethod", "1")
}

func TestTimes0(t *testing.T) {
	rep, ctrl := createFixtures(t)
	defer ctrl.Finish()

	s := new(Subject)
	ctrl.RecordCall(s, "FooMethod", "arg").Times(0)
	rep.assertFatal(func() {
		ctrl.Call(s, "FooMethod", "arg")
	})
}

func TestVariadicMatching(t *testing.T) {
	rep, ctrl := createFixtures(t)
	defer rep.recoverUnexpectedFatal()

	s := new(Subject)
	ctrl.RecordCall(s, "VariadicMethod", 0, "1", "2")
	ctrl.Call(s, "VariadicMethod", 0, "1", "2")
	ctrl.Finish()
	rep.assertPass("variadic matching works")
}

func TestVariadicNoMatch(t *testing.T) {
	rep, ctrl := createFixtures(t)
	defer rep.recoverUnexpectedFatal()

	s := new(Subject)
	ctrl.RecordCall(s, "VariadicMethod", 0)
	rep.assertFatal(func() {
		ctrl.Call(s, "VariadicMethod", 1)
	}, "expected call at", "doesn't match the argument at index 0",
		"Got: 1\nWant: is equal to 0")
	ctrl.Call(s, "VariadicMethod", 0)
	ctrl.Finish()
}

func TestVariadicMatchingWithSlice(t *testing.T) {
	testCases := [][]string{
		{"1"},
		{"1", "2"},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d arguments", len(tc)), func(t *testing.T) {
			rep, ctrl := createFixtures(t)
			defer rep.recoverUnexpectedFatal()

			s := new(Subject)
			ctrl.RecordCall(s, "VariadicMethod", 1, tc)
			args := make([]interface{}, len(tc)+1)
			args[0] = 1
			for i, arg := range tc {
				args[i+1] = arg
			}
			ctrl.Call(s, "VariadicMethod", args...)
			ctrl.Finish()
			rep.assertPass("slices can be used as matchers for variadic arguments")
		})
	}
}

func TestDuplicateFinishCallFails(t *testing.T) {
	rep, ctrl := createFixtures(t)

	ctrl.Finish()
	rep.assertPass("the first Finish call should succeed")

	rep.assertFatal(ctrl.Finish, "Controller.Finish was called more than once. It has to be called exactly once.")
}

func TestNoHelper(t *testing.T) {
	ctrlNoHelper := gomock.NewController(NewErrorReporter(t))

	// doesn't panic
	ctrlNoHelper.T.Helper()
}

func TestWithHelper(t *testing.T) {
	withHelper := &HelperReporter{TestReporter: NewErrorReporter(t)}
	ctrlWithHelper := gomock.NewController(withHelper)

	ctrlWithHelper.T.Helper()

	if withHelper.helper == 0 {
		t.Fatal("expected Helper to be invoked")
	}
}
