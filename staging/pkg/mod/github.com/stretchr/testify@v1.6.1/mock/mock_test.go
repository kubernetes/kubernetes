package mock

import (
	"errors"
	"fmt"
	"regexp"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

/*
	Test objects
*/

// ExampleInterface represents an example interface.
type ExampleInterface interface {
	TheExampleMethod(a, b, c int) (int, error)
}

// TestExampleImplementation is a test implementation of ExampleInterface
type TestExampleImplementation struct {
	Mock
}

func (i *TestExampleImplementation) TheExampleMethod(a, b, c int) (int, error) {
	args := i.Called(a, b, c)
	return args.Int(0), errors.New("Whoops")
}

//go:noinline
func (i *TestExampleImplementation) TheExampleMethod2(yesorno bool) {
	i.Called(yesorno)
}

type ExampleType struct {
	ran bool
}

func (i *TestExampleImplementation) TheExampleMethod3(et *ExampleType) error {
	args := i.Called(et)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethod4(v ExampleInterface) error {
	args := i.Called(v)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethod5(ch chan struct{}) error {
	args := i.Called(ch)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethod6(m map[string]bool) error {
	args := i.Called(m)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethod7(slice []bool) error {
	args := i.Called(slice)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethodFunc(fn func(string) error) error {
	args := i.Called(fn)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethodVariadic(a ...int) error {
	args := i.Called(a)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethodVariadicInterface(a ...interface{}) error {
	args := i.Called(a)
	return args.Error(0)
}

func (i *TestExampleImplementation) TheExampleMethodMixedVariadic(a int, b ...int) error {
	args := i.Called(a, b)
	return args.Error(0)
}

type ExampleFuncType func(string) error

func (i *TestExampleImplementation) TheExampleMethodFuncType(fn ExampleFuncType) error {
	args := i.Called(fn)
	return args.Error(0)
}

// MockTestingT mocks a test struct
type MockTestingT struct {
	logfCount, errorfCount, failNowCount int
}

const mockTestingTFailNowCalled = "FailNow was called"

func (m *MockTestingT) Logf(string, ...interface{}) {
	m.logfCount++
}

func (m *MockTestingT) Errorf(string, ...interface{}) {
	m.errorfCount++
}

// FailNow mocks the FailNow call.
// It panics in order to mimic the FailNow behavior in the sense that
// the execution stops.
// When expecting this method, the call that invokes it should use the following code:
//
//     assert.PanicsWithValue(t, mockTestingTFailNowCalled, func() {...})
func (m *MockTestingT) FailNow() {
	m.failNowCount++

	// this function should panic now to stop the execution as expected
	panic(mockTestingTFailNowCalled)
}

/*
	Mock
*/

func Test_Mock_TestData(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	if assert.NotNil(t, mockedService.TestData()) {

		mockedService.TestData().Set("something", 123)
		assert.Equal(t, 123, mockedService.TestData().Get("something").Data())
	}
}

func Test_Mock_On(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.On("TheExampleMethod")
	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, "TheExampleMethod", c.Method)
}

func Test_Mock_Chained_On(t *testing.T) {
	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	// determine our current line number so we can assert the expected calls callerInfo properly
	_, _, line, _ := runtime.Caller(0)
	mockedService.
		On("TheExampleMethod", 1, 2, 3).
		Return(0).
		On("TheExampleMethod3", AnythingOfType("*mock.ExampleType")).
		Return(nil)

	expectedCalls := []*Call{
		{
			Parent:          &mockedService.Mock,
			Method:          "TheExampleMethod",
			Arguments:       []interface{}{1, 2, 3},
			ReturnArguments: []interface{}{0},
			callerInfo:      []string{fmt.Sprintf("mock_test.go:%d", line+2)},
		},
		{
			Parent:          &mockedService.Mock,
			Method:          "TheExampleMethod3",
			Arguments:       []interface{}{AnythingOfType("*mock.ExampleType")},
			ReturnArguments: []interface{}{nil},
			callerInfo:      []string{fmt.Sprintf("mock_test.go:%d", line+4)},
		},
	}
	assert.Equal(t, expectedCalls, mockedService.ExpectedCalls)
}

func Test_Mock_On_WithArgs(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.On("TheExampleMethod", 1, 2, 3, 4)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, "TheExampleMethod", c.Method)
	assert.Equal(t, Arguments{1, 2, 3, 4}, c.Arguments)
}

func Test_Mock_On_WithFuncArg(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethodFunc", AnythingOfType("func(string) error")).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, "TheExampleMethodFunc", c.Method)
	assert.Equal(t, 1, len(c.Arguments))
	assert.Equal(t, AnythingOfType("func(string) error"), c.Arguments[0])

	fn := func(string) error { return nil }

	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodFunc(fn)
	})
}

func Test_Mock_On_WithIntArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod",
		MatchedBy(func(a int) bool {
			return a == 1
		}), MatchedBy(func(b int) bool {
			return b == 2
		}), MatchedBy(func(c int) bool {
			return c == 3
		})).Return(0, nil)

	assert.Panics(t, func() {
		mockedService.TheExampleMethod(1, 2, 4)
	})
	assert.Panics(t, func() {
		mockedService.TheExampleMethod(2, 2, 3)
	})
	assert.NotPanics(t, func() {
		mockedService.TheExampleMethod(1, 2, 3)
	})
}

func TestMock_WithTest(t *testing.T) {
	var (
		mockedService TestExampleImplementation
		mockedTest    MockTestingT
	)

	mockedService.Test(&mockedTest)
	mockedService.On("TheExampleMethod", 1, 2, 3).Return(0, nil)

	// Test that on an expected call, the test was not failed

	mockedService.TheExampleMethod(1, 2, 3)

	// Assert that Errorf and FailNow were not called
	assert.Equal(t, 0, mockedTest.errorfCount)
	assert.Equal(t, 0, mockedTest.failNowCount)

	// Test that on unexpected call, the mocked test was called to fail the test

	assert.PanicsWithValue(t, mockTestingTFailNowCalled, func() {
		mockedService.TheExampleMethod(1, 1, 1)
	})

	// Assert that Errorf and FailNow were called once
	assert.Equal(t, 1, mockedTest.errorfCount)
	assert.Equal(t, 1, mockedTest.failNowCount)
}

func Test_Mock_On_WithPtrArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod3",
		MatchedBy(func(a *ExampleType) bool { return a != nil && a.ran == true }),
	).Return(nil)

	mockedService.On("TheExampleMethod3",
		MatchedBy(func(a *ExampleType) bool { return a != nil && a.ran == false }),
	).Return(errors.New("error"))

	mockedService.On("TheExampleMethod3",
		MatchedBy(func(a *ExampleType) bool { return a == nil }),
	).Return(errors.New("error2"))

	assert.Equal(t, mockedService.TheExampleMethod3(&ExampleType{true}), nil)
	assert.EqualError(t, mockedService.TheExampleMethod3(&ExampleType{false}), "error")
	assert.EqualError(t, mockedService.TheExampleMethod3(nil), "error2")
}

func Test_Mock_On_WithFuncArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	fixture1, fixture2 := errors.New("fixture1"), errors.New("fixture2")

	mockedService.On("TheExampleMethodFunc",
		MatchedBy(func(a func(string) error) bool { return a != nil && a("string") == fixture1 }),
	).Return(errors.New("fixture1"))

	mockedService.On("TheExampleMethodFunc",
		MatchedBy(func(a func(string) error) bool { return a != nil && a("string") == fixture2 }),
	).Return(errors.New("fixture2"))

	mockedService.On("TheExampleMethodFunc",
		MatchedBy(func(a func(string) error) bool { return a == nil }),
	).Return(errors.New("fixture3"))

	assert.EqualError(t, mockedService.TheExampleMethodFunc(
		func(string) error { return fixture1 }), "fixture1")
	assert.EqualError(t, mockedService.TheExampleMethodFunc(
		func(string) error { return fixture2 }), "fixture2")
	assert.EqualError(t, mockedService.TheExampleMethodFunc(nil), "fixture3")
}

func Test_Mock_On_WithInterfaceArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod4",
		MatchedBy(func(a ExampleInterface) bool { return a == nil }),
	).Return(errors.New("fixture1"))

	assert.EqualError(t, mockedService.TheExampleMethod4(nil), "fixture1")
}

func Test_Mock_On_WithChannelArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod5",
		MatchedBy(func(ch chan struct{}) bool { return ch == nil }),
	).Return(errors.New("fixture1"))

	assert.EqualError(t, mockedService.TheExampleMethod5(nil), "fixture1")
}

func Test_Mock_On_WithMapArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod6",
		MatchedBy(func(m map[string]bool) bool { return m == nil }),
	).Return(errors.New("fixture1"))

	assert.EqualError(t, mockedService.TheExampleMethod6(nil), "fixture1")
}

func Test_Mock_On_WithSliceArgMatcher(t *testing.T) {
	var mockedService TestExampleImplementation

	mockedService.On("TheExampleMethod7",
		MatchedBy(func(slice []bool) bool { return slice == nil }),
	).Return(errors.New("fixture1"))

	assert.EqualError(t, mockedService.TheExampleMethod7(nil), "fixture1")
}

func Test_Mock_On_WithVariadicFunc(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethodVariadic", []int{1, 2, 3}).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, 1, len(c.Arguments))
	assert.Equal(t, []int{1, 2, 3}, c.Arguments[0])

	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodVariadic(1, 2, 3)
	})
	assert.Panics(t, func() {
		mockedService.TheExampleMethodVariadic(1, 2)
	})

}

func Test_Mock_On_WithMixedVariadicFunc(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethodMixedVariadic", 1, []int{2, 3, 4}).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, 2, len(c.Arguments))
	assert.Equal(t, 1, c.Arguments[0])
	assert.Equal(t, []int{2, 3, 4}, c.Arguments[1])

	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodMixedVariadic(1, 2, 3, 4)
	})
	assert.Panics(t, func() {
		mockedService.TheExampleMethodMixedVariadic(1, 2, 3, 5)
	})

}

func Test_Mock_On_WithVariadicFuncWithInterface(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.On("TheExampleMethodVariadicInterface", []interface{}{1, 2, 3}).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, 1, len(c.Arguments))
	assert.Equal(t, []interface{}{1, 2, 3}, c.Arguments[0])

	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodVariadicInterface(1, 2, 3)
	})
	assert.Panics(t, func() {
		mockedService.TheExampleMethodVariadicInterface(1, 2)
	})

}

func Test_Mock_On_WithVariadicFuncWithEmptyInterfaceArray(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	var expected []interface{}
	c := mockedService.
		On("TheExampleMethodVariadicInterface", expected).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, 1, len(c.Arguments))
	assert.Equal(t, expected, c.Arguments[0])

	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodVariadicInterface()
	})
	assert.Panics(t, func() {
		mockedService.TheExampleMethodVariadicInterface(1, 2)
	})

}

func Test_Mock_On_WithFuncPanics(t *testing.T) {
	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	assert.Panics(t, func() {
		mockedService.On("TheExampleMethodFunc", func(string) error { return nil })
	})
}

func Test_Mock_On_WithFuncTypeArg(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethodFuncType", AnythingOfType("mock.ExampleFuncType")).
		Return(nil)

	assert.Equal(t, []*Call{c}, mockedService.ExpectedCalls)
	assert.Equal(t, 1, len(c.Arguments))
	assert.Equal(t, AnythingOfType("mock.ExampleFuncType"), c.Arguments[0])

	fn := func(string) error { return nil }
	assert.NotPanics(t, func() {
		mockedService.TheExampleMethodFuncType(fn)
	})
}

func Test_Mock_Return(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethod", "A", "B", true).
		Return(1, "two", true)

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 0, call.Repeatability)
	assert.Nil(t, call.WaitFor)
}

func Test_Mock_Panic(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethod", "A", "B", true).
		Panic("panic message for example method")

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 0, call.Repeatability)
	assert.Equal(t, 0, call.Repeatability)
	assert.Equal(t, "panic message for example method", *call.PanicMsg)
	assert.Nil(t, call.WaitFor)
}

func Test_Mock_Return_WaitUntil(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)
	ch := time.After(time.Second)

	c := mockedService.Mock.
		On("TheExampleMethod", "A", "B", true).
		WaitUntil(ch).
		Return(1, "two", true)

	// assert that the call was created
	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 0, call.Repeatability)
	assert.Equal(t, ch, call.WaitFor)
}

func Test_Mock_Return_After(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.Mock.
		On("TheExampleMethod", "A", "B", true).
		Return(1, "two", true).
		After(time.Second)

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.Mock.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 0, call.Repeatability)
	assert.NotEqual(t, nil, call.WaitFor)

}

func Test_Mock_Return_Run(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	fn := func(args Arguments) {
		arg := args.Get(0).(*ExampleType)
		arg.ran = true
	}

	c := mockedService.Mock.
		On("TheExampleMethod3", AnythingOfType("*mock.ExampleType")).
		Return(nil).
		Run(fn)

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.Mock.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod3", call.Method)
	assert.Equal(t, AnythingOfType("*mock.ExampleType"), call.Arguments[0])
	assert.Equal(t, nil, call.ReturnArguments[0])
	assert.Equal(t, 0, call.Repeatability)
	assert.NotEqual(t, nil, call.WaitFor)
	assert.NotNil(t, call.Run)

	et := ExampleType{}
	assert.Equal(t, false, et.ran)
	mockedService.TheExampleMethod3(&et)
	assert.Equal(t, true, et.ran)
}

func Test_Mock_Return_Run_Out_Of_Order(t *testing.T) {
	// make a test impl object
	var mockedService = new(TestExampleImplementation)
	f := func(args Arguments) {
		arg := args.Get(0).(*ExampleType)
		arg.ran = true
	}

	c := mockedService.Mock.
		On("TheExampleMethod3", AnythingOfType("*mock.ExampleType")).
		Run(f).
		Return(nil)

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.Mock.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod3", call.Method)
	assert.Equal(t, AnythingOfType("*mock.ExampleType"), call.Arguments[0])
	assert.Equal(t, nil, call.ReturnArguments[0])
	assert.Equal(t, 0, call.Repeatability)
	assert.NotEqual(t, nil, call.WaitFor)
	assert.NotNil(t, call.Run)
}

func Test_Mock_Return_Once(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.On("TheExampleMethod", "A", "B", true).
		Return(1, "two", true).
		Once()

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 1, call.Repeatability)
	assert.Nil(t, call.WaitFor)
}

func Test_Mock_Return_Twice(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethod", "A", "B", true).
		Return(1, "two", true).
		Twice()

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 2, call.Repeatability)
	assert.Nil(t, call.WaitFor)
}

func Test_Mock_Return_Times(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethod", "A", "B", true).
		Return(1, "two", true).
		Times(5)

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 1, call.ReturnArguments[0])
	assert.Equal(t, "two", call.ReturnArguments[1])
	assert.Equal(t, true, call.ReturnArguments[2])
	assert.Equal(t, 5, call.Repeatability)
	assert.Nil(t, call.WaitFor)
}

func Test_Mock_Return_Nothing(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	c := mockedService.
		On("TheExampleMethod", "A", "B", true).
		Return()

	require.Equal(t, []*Call{c}, mockedService.ExpectedCalls)

	call := mockedService.ExpectedCalls[0]

	assert.Equal(t, "TheExampleMethod", call.Method)
	assert.Equal(t, "A", call.Arguments[0])
	assert.Equal(t, "B", call.Arguments[1])
	assert.Equal(t, true, call.Arguments[2])
	assert.Equal(t, 0, len(call.ReturnArguments))
}

func Test_Mock_findExpectedCall(t *testing.T) {

	m := new(Mock)
	m.On("One", 1).Return("one")
	m.On("Two", 2).Return("two")
	m.On("Two", 3).Return("three")

	f, c := m.findExpectedCall("Two", 3)

	if assert.Equal(t, 2, f) {
		if assert.NotNil(t, c) {
			assert.Equal(t, "Two", c.Method)
			assert.Equal(t, 3, c.Arguments[0])
			assert.Equal(t, "three", c.ReturnArguments[0])
		}
	}

}

func Test_Mock_findExpectedCall_For_Unknown_Method(t *testing.T) {

	m := new(Mock)
	m.On("One", 1).Return("one")
	m.On("Two", 2).Return("two")
	m.On("Two", 3).Return("three")

	f, _ := m.findExpectedCall("Two")

	assert.Equal(t, -1, f)

}

func Test_Mock_findExpectedCall_Respects_Repeatability(t *testing.T) {

	m := new(Mock)
	m.On("One", 1).Return("one")
	m.On("Two", 2).Return("two").Once()
	m.On("Two", 3).Return("three").Twice()
	m.On("Two", 3).Return("three").Times(8)

	f, c := m.findExpectedCall("Two", 3)

	if assert.Equal(t, 2, f) {
		if assert.NotNil(t, c) {
			assert.Equal(t, "Two", c.Method)
			assert.Equal(t, 3, c.Arguments[0])
			assert.Equal(t, "three", c.ReturnArguments[0])
		}
	}

	c = m.On("Once", 1).Return("one").Once()
	c.Repeatability = -1
	f, c = m.findExpectedCall("Once", 1)
	if assert.Equal(t, -1, f) {
		if assert.NotNil(t, c) {
			assert.Equal(t, "Once", c.Method)
			assert.Equal(t, 1, c.Arguments[0])
			assert.Equal(t, "one", c.ReturnArguments[0])
		}
	}
}

func Test_callString(t *testing.T) {

	assert.Equal(t, `Method(int,bool,string)`, callString("Method", []interface{}{1, true, "something"}, false))
	assert.Equal(t, `Method(<nil>)`, callString("Method", []interface{}{nil}, false))

}

func Test_Mock_Called(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_Called", 1, 2, 3).Return(5, "6", true)

	returnArguments := mockedService.Called(1, 2, 3)

	if assert.Equal(t, 1, len(mockedService.Calls)) {
		assert.Equal(t, "Test_Mock_Called", mockedService.Calls[0].Method)
		assert.Equal(t, 1, mockedService.Calls[0].Arguments[0])
		assert.Equal(t, 2, mockedService.Calls[0].Arguments[1])
		assert.Equal(t, 3, mockedService.Calls[0].Arguments[2])
	}

	if assert.Equal(t, 3, len(returnArguments)) {
		assert.Equal(t, 5, returnArguments[0])
		assert.Equal(t, "6", returnArguments[1])
		assert.Equal(t, true, returnArguments[2])
	}

}

func asyncCall(m *Mock, ch chan Arguments) {
	ch <- m.Called(1, 2, 3)
}

func Test_Mock_Called_blocks(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.Mock.On("asyncCall", 1, 2, 3).Return(5, "6", true).After(2 * time.Millisecond)

	ch := make(chan Arguments)

	go asyncCall(&mockedService.Mock, ch)

	select {
	case <-ch:
		t.Fatal("should have waited")
	case <-time.After(1 * time.Millisecond):
	}

	returnArguments := <-ch

	if assert.Equal(t, 1, len(mockedService.Mock.Calls)) {
		assert.Equal(t, "asyncCall", mockedService.Mock.Calls[0].Method)
		assert.Equal(t, 1, mockedService.Mock.Calls[0].Arguments[0])
		assert.Equal(t, 2, mockedService.Mock.Calls[0].Arguments[1])
		assert.Equal(t, 3, mockedService.Mock.Calls[0].Arguments[2])
	}

	if assert.Equal(t, 3, len(returnArguments)) {
		assert.Equal(t, 5, returnArguments[0])
		assert.Equal(t, "6", returnArguments[1])
		assert.Equal(t, true, returnArguments[2])
	}

}

func Test_Mock_Called_For_Bounded_Repeatability(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.
		On("Test_Mock_Called_For_Bounded_Repeatability", 1, 2, 3).
		Return(5, "6", true).
		Once()
	mockedService.
		On("Test_Mock_Called_For_Bounded_Repeatability", 1, 2, 3).
		Return(-1, "hi", false)

	returnArguments1 := mockedService.Called(1, 2, 3)
	returnArguments2 := mockedService.Called(1, 2, 3)

	if assert.Equal(t, 2, len(mockedService.Calls)) {
		assert.Equal(t, "Test_Mock_Called_For_Bounded_Repeatability", mockedService.Calls[0].Method)
		assert.Equal(t, 1, mockedService.Calls[0].Arguments[0])
		assert.Equal(t, 2, mockedService.Calls[0].Arguments[1])
		assert.Equal(t, 3, mockedService.Calls[0].Arguments[2])

		assert.Equal(t, "Test_Mock_Called_For_Bounded_Repeatability", mockedService.Calls[1].Method)
		assert.Equal(t, 1, mockedService.Calls[1].Arguments[0])
		assert.Equal(t, 2, mockedService.Calls[1].Arguments[1])
		assert.Equal(t, 3, mockedService.Calls[1].Arguments[2])
	}

	if assert.Equal(t, 3, len(returnArguments1)) {
		assert.Equal(t, 5, returnArguments1[0])
		assert.Equal(t, "6", returnArguments1[1])
		assert.Equal(t, true, returnArguments1[2])
	}

	if assert.Equal(t, 3, len(returnArguments2)) {
		assert.Equal(t, -1, returnArguments2[0])
		assert.Equal(t, "hi", returnArguments2[1])
		assert.Equal(t, false, returnArguments2[2])
	}

}

func Test_Mock_Called_For_SetTime_Expectation(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("TheExampleMethod", 1, 2, 3).Return(5, "6", true).Times(4)

	mockedService.TheExampleMethod(1, 2, 3)
	mockedService.TheExampleMethod(1, 2, 3)
	mockedService.TheExampleMethod(1, 2, 3)
	mockedService.TheExampleMethod(1, 2, 3)
	assert.Panics(t, func() {
		mockedService.TheExampleMethod(1, 2, 3)
	})

}

func Test_Mock_Called_Unexpected(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	// make sure it panics if no expectation was made
	assert.Panics(t, func() {
		mockedService.Called(1, 2, 3)
	}, "Calling unexpected method should panic")

}

func Test_AssertExpectationsForObjects_Helper(t *testing.T) {

	var mockedService1 = new(TestExampleImplementation)
	var mockedService2 = new(TestExampleImplementation)
	var mockedService3 = new(TestExampleImplementation)

	mockedService1.On("Test_AssertExpectationsForObjects_Helper", 1).Return()
	mockedService2.On("Test_AssertExpectationsForObjects_Helper", 2).Return()
	mockedService3.On("Test_AssertExpectationsForObjects_Helper", 3).Return()

	mockedService1.Called(1)
	mockedService2.Called(2)
	mockedService3.Called(3)

	assert.True(t, AssertExpectationsForObjects(t, &mockedService1.Mock, &mockedService2.Mock, &mockedService3.Mock))
	assert.True(t, AssertExpectationsForObjects(t, mockedService1, mockedService2, mockedService3))

}

func Test_AssertExpectationsForObjects_Helper_Failed(t *testing.T) {

	var mockedService1 = new(TestExampleImplementation)
	var mockedService2 = new(TestExampleImplementation)
	var mockedService3 = new(TestExampleImplementation)

	mockedService1.On("Test_AssertExpectationsForObjects_Helper_Failed", 1).Return()
	mockedService2.On("Test_AssertExpectationsForObjects_Helper_Failed", 2).Return()
	mockedService3.On("Test_AssertExpectationsForObjects_Helper_Failed", 3).Return()

	mockedService1.Called(1)
	mockedService3.Called(3)

	tt := new(testing.T)
	assert.False(t, AssertExpectationsForObjects(tt, &mockedService1.Mock, &mockedService2.Mock, &mockedService3.Mock))
	assert.False(t, AssertExpectationsForObjects(tt, mockedService1, mockedService2, mockedService3))

}

func Test_Mock_AssertExpectations(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertExpectations", 1, 2, 3).Return(5, 6, 7)

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	// make the call now
	mockedService.Called(1, 2, 3)

	// now assert expectations
	assert.True(t, mockedService.AssertExpectations(tt))

}

func Test_Mock_AssertExpectations_Placeholder_NoArgs(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertExpectations_Placeholder_NoArgs").Return(5, 6, 7).Once()
	mockedService.On("Test_Mock_AssertExpectations_Placeholder_NoArgs").Return(7, 6, 5)

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	// make the call now
	mockedService.Called()

	// now assert expectations
	assert.True(t, mockedService.AssertExpectations(tt))

}

func Test_Mock_AssertExpectations_Placeholder(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertExpectations_Placeholder", 1, 2, 3).Return(5, 6, 7).Once()
	mockedService.On("Test_Mock_AssertExpectations_Placeholder", 3, 2, 1).Return(7, 6, 5)

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	// make the call now
	mockedService.Called(1, 2, 3)

	// now assert expectations
	assert.False(t, mockedService.AssertExpectations(tt))

	// make call to the second expectation
	mockedService.Called(3, 2, 1)

	// now assert expectations again
	assert.True(t, mockedService.AssertExpectations(tt))
}

func Test_Mock_AssertExpectations_With_Pointers(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertExpectations_With_Pointers", &struct{ Foo int }{1}).Return(1)
	mockedService.On("Test_Mock_AssertExpectations_With_Pointers", &struct{ Foo int }{2}).Return(2)

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	s := struct{ Foo int }{1}
	// make the calls now
	mockedService.Called(&s)
	s.Foo = 2
	mockedService.Called(&s)

	// now assert expectations
	assert.True(t, mockedService.AssertExpectations(tt))

}

func Test_Mock_AssertExpectationsCustomType(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("TheExampleMethod3", AnythingOfType("*mock.ExampleType")).Return(nil).Once()

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	// make the call now
	mockedService.TheExampleMethod3(&ExampleType{})

	// now assert expectations
	assert.True(t, mockedService.AssertExpectations(tt))

}

func Test_Mock_AssertExpectations_With_Repeatability(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertExpectations_With_Repeatability", 1, 2, 3).Return(5, 6, 7).Twice()

	tt := new(testing.T)
	assert.False(t, mockedService.AssertExpectations(tt))

	// make the call now
	mockedService.Called(1, 2, 3)

	assert.False(t, mockedService.AssertExpectations(tt))

	mockedService.Called(1, 2, 3)

	// now assert expectations
	assert.True(t, mockedService.AssertExpectations(tt))

}

func Test_Mock_TwoCallsWithDifferentArguments(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_TwoCallsWithDifferentArguments", 1, 2, 3).Return(5, 6, 7)
	mockedService.On("Test_Mock_TwoCallsWithDifferentArguments", 4, 5, 6).Return(5, 6, 7)

	args1 := mockedService.Called(1, 2, 3)
	assert.Equal(t, 5, args1.Int(0))
	assert.Equal(t, 6, args1.Int(1))
	assert.Equal(t, 7, args1.Int(2))

	args2 := mockedService.Called(4, 5, 6)
	assert.Equal(t, 5, args2.Int(0))
	assert.Equal(t, 6, args2.Int(1))
	assert.Equal(t, 7, args2.Int(2))

}

func Test_Mock_AssertNumberOfCalls(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertNumberOfCalls", 1, 2, 3).Return(5, 6, 7)

	mockedService.Called(1, 2, 3)
	assert.True(t, mockedService.AssertNumberOfCalls(t, "Test_Mock_AssertNumberOfCalls", 1))

	mockedService.Called(1, 2, 3)
	assert.True(t, mockedService.AssertNumberOfCalls(t, "Test_Mock_AssertNumberOfCalls", 2))

}

func Test_Mock_AssertCalled(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertCalled", 1, 2, 3).Return(5, 6, 7)

	mockedService.Called(1, 2, 3)

	assert.True(t, mockedService.AssertCalled(t, "Test_Mock_AssertCalled", 1, 2, 3))

}

func Test_Mock_AssertCalled_WithAnythingOfTypeArgument(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.
		On("Test_Mock_AssertCalled_WithAnythingOfTypeArgument", Anything, Anything, Anything).
		Return()

	mockedService.Called(1, "two", []uint8("three"))

	assert.True(t, mockedService.AssertCalled(t, "Test_Mock_AssertCalled_WithAnythingOfTypeArgument", AnythingOfType("int"), AnythingOfType("string"), AnythingOfType("[]uint8")))

}

func Test_Mock_AssertCalled_WithArguments(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertCalled_WithArguments", 1, 2, 3).Return(5, 6, 7)

	mockedService.Called(1, 2, 3)

	tt := new(testing.T)
	assert.True(t, mockedService.AssertCalled(tt, "Test_Mock_AssertCalled_WithArguments", 1, 2, 3))
	assert.False(t, mockedService.AssertCalled(tt, "Test_Mock_AssertCalled_WithArguments", 2, 3, 4))

}

func Test_Mock_AssertCalled_WithArguments_With_Repeatability(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertCalled_WithArguments_With_Repeatability", 1, 2, 3).Return(5, 6, 7).Once()
	mockedService.On("Test_Mock_AssertCalled_WithArguments_With_Repeatability", 2, 3, 4).Return(5, 6, 7).Once()

	mockedService.Called(1, 2, 3)
	mockedService.Called(2, 3, 4)

	tt := new(testing.T)
	assert.True(t, mockedService.AssertCalled(tt, "Test_Mock_AssertCalled_WithArguments_With_Repeatability", 1, 2, 3))
	assert.True(t, mockedService.AssertCalled(tt, "Test_Mock_AssertCalled_WithArguments_With_Repeatability", 2, 3, 4))
	assert.False(t, mockedService.AssertCalled(tt, "Test_Mock_AssertCalled_WithArguments_With_Repeatability", 3, 4, 5))

}

func Test_Mock_AssertNotCalled(t *testing.T) {

	var mockedService = new(TestExampleImplementation)

	mockedService.On("Test_Mock_AssertNotCalled", 1, 2, 3).Return(5, 6, 7)

	mockedService.Called(1, 2, 3)

	assert.True(t, mockedService.AssertNotCalled(t, "Test_Mock_NotCalled"))

}

func Test_Mock_IsMethodCallable(t *testing.T) {
	var mockedService = new(TestExampleImplementation)

	arg := []Call{{Repeatability: 1}, {Repeatability: 2}}
	arg2 := []Call{{Repeatability: 1}, {Repeatability: 1}}
	arg3 := []Call{{Repeatability: 1}, {Repeatability: 1}}

	mockedService.On("Test_Mock_IsMethodCallable", arg2).Return(true).Twice()

	assert.False(t, mockedService.IsMethodCallable(t, "Test_Mock_IsMethodCallable", arg))
	assert.True(t, mockedService.IsMethodCallable(t, "Test_Mock_IsMethodCallable", arg2))
	assert.True(t, mockedService.IsMethodCallable(t, "Test_Mock_IsMethodCallable", arg3))

	mockedService.MethodCalled("Test_Mock_IsMethodCallable", arg2)
	mockedService.MethodCalled("Test_Mock_IsMethodCallable", arg2)

	assert.False(t, mockedService.IsMethodCallable(t, "Test_Mock_IsMethodCallable", arg2))
}

func TestIsArgsEqual(t *testing.T) {
	var expected = Arguments{5, 3, 4, 6, 7, 2}
	var args = make([]interface{}, 5)
	for i := 1; i < len(expected); i++ {
		args[i-1] = expected[i]
	}
	args[2] = expected[1]
	assert.False(t, isArgsEqual(expected, args))

	var arr = make([]interface{}, 6)
	for i := 0; i < len(expected); i++ {
		arr[i] = expected[i]
	}
	assert.True(t, isArgsEqual(expected, arr))
}

func Test_Mock_AssertOptional(t *testing.T) {
	// Optional called
	var ms1 = new(TestExampleImplementation)
	ms1.On("TheExampleMethod", 1, 2, 3).Maybe().Return(4, nil)
	ms1.TheExampleMethod(1, 2, 3)

	tt1 := new(testing.T)
	assert.Equal(t, true, ms1.AssertExpectations(tt1))

	// Optional not called
	var ms2 = new(TestExampleImplementation)
	ms2.On("TheExampleMethod", 1, 2, 3).Maybe().Return(4, nil)

	tt2 := new(testing.T)
	assert.Equal(t, true, ms2.AssertExpectations(tt2))

	// Non-optional called
	var ms3 = new(TestExampleImplementation)
	ms3.On("TheExampleMethod", 1, 2, 3).Return(4, nil)
	ms3.TheExampleMethod(1, 2, 3)

	tt3 := new(testing.T)
	assert.Equal(t, true, ms3.AssertExpectations(tt3))
}

/*
	Arguments helper methods
*/
func Test_Arguments_Get(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})

	assert.Equal(t, "string", args.Get(0).(string))
	assert.Equal(t, 123, args.Get(1).(int))
	assert.Equal(t, true, args.Get(2).(bool))

}

func Test_Arguments_Is(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})

	assert.True(t, args.Is("string", 123, true))
	assert.False(t, args.Is("wrong", 456, false))

}

func Test_Arguments_Diff(t *testing.T) {

	var args = Arguments([]interface{}{"Hello World", 123, true})
	var diff string
	var count int
	diff, count = args.Diff([]interface{}{"Hello World", 456, "false"})

	assert.Equal(t, 2, count)
	assert.Contains(t, diff, `(int=456) != (int=123)`)
	assert.Contains(t, diff, `(string=false) != (bool=true)`)

}

func Test_Arguments_Diff_DifferentNumberOfArgs(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	var diff string
	var count int
	diff, count = args.Diff([]interface{}{"string", 456, "false", "extra"})

	assert.Equal(t, 3, count)
	assert.Contains(t, diff, `(string=extra) != (Missing)`)

}

func Test_Arguments_Diff_WithAnythingArgument(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	var count int
	_, count = args.Diff([]interface{}{"string", Anything, true})

	assert.Equal(t, 0, count)

}

func Test_Arguments_Diff_WithAnythingArgument_InActualToo(t *testing.T) {

	var args = Arguments([]interface{}{"string", Anything, true})
	var count int
	_, count = args.Diff([]interface{}{"string", 123, true})

	assert.Equal(t, 0, count)

}

func Test_Arguments_Diff_WithAnythingOfTypeArgument(t *testing.T) {

	var args = Arguments([]interface{}{"string", AnythingOfType("int"), true})
	var count int
	_, count = args.Diff([]interface{}{"string", 123, true})

	assert.Equal(t, 0, count)

}

func Test_Arguments_Diff_WithAnythingOfTypeArgument_Failing(t *testing.T) {

	var args = Arguments([]interface{}{"string", AnythingOfType("string"), true})
	var count int
	var diff string
	diff, count = args.Diff([]interface{}{"string", 123, true})

	assert.Equal(t, 1, count)
	assert.Contains(t, diff, `string != type int - (int=123)`)

}

func Test_Arguments_Diff_WithIsTypeArgument(t *testing.T) {
	var args = Arguments([]interface{}{"string", IsType(0), true})
	var count int
	_, count = args.Diff([]interface{}{"string", 123, true})

	assert.Equal(t, 0, count)
}

func Test_Arguments_Diff_WithIsTypeArgument_Failing(t *testing.T) {
	var args = Arguments([]interface{}{"string", IsType(""), true})
	var count int
	var diff string
	diff, count = args.Diff([]interface{}{"string", 123, true})

	assert.Equal(t, 1, count)
	assert.Contains(t, diff, `string != type int - (int=123)`)
}

func Test_Arguments_Diff_WithArgMatcher(t *testing.T) {
	matchFn := func(a int) bool {
		return a == 123
	}
	var args = Arguments([]interface{}{"string", MatchedBy(matchFn), true})

	diff, count := args.Diff([]interface{}{"string", 124, true})
	assert.Equal(t, 1, count)
	assert.Contains(t, diff, `(int=124) not matched by func(int) bool`)

	diff, count = args.Diff([]interface{}{"string", false, true})
	assert.Equal(t, 1, count)
	assert.Contains(t, diff, `(bool=false) not matched by func(int) bool`)

	diff, count = args.Diff([]interface{}{"string", 123, false})
	assert.Equal(t, 1, count)
	assert.Contains(t, diff, `(int=123) matched by func(int) bool`)

	diff, count = args.Diff([]interface{}{"string", 123, true})
	assert.Equal(t, 0, count)
	assert.Contains(t, diff, `No differences.`)
}

func Test_Arguments_Assert(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})

	assert.True(t, args.Assert(t, "string", 123, true))

}

func Test_Arguments_String_Representation(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	assert.Equal(t, `string,int,bool`, args.String())

}

func Test_Arguments_String(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	assert.Equal(t, "string", args.String(0))

}

func Test_Arguments_Error(t *testing.T) {

	var err = errors.New("An Error")
	var args = Arguments([]interface{}{"string", 123, true, err})
	assert.Equal(t, err, args.Error(3))

}

func Test_Arguments_Error_Nil(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true, nil})
	assert.Equal(t, nil, args.Error(3))

}

func Test_Arguments_Int(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	assert.Equal(t, 123, args.Int(1))

}

func Test_Arguments_Bool(t *testing.T) {

	var args = Arguments([]interface{}{"string", 123, true})
	assert.Equal(t, true, args.Bool(2))

}

func Test_WaitUntil_Parallel(t *testing.T) {

	// make a test impl object
	var mockedService = new(TestExampleImplementation)

	ch1 := make(chan time.Time)
	ch2 := make(chan time.Time)

	mockedService.Mock.On("TheExampleMethod2", true).Return().WaitUntil(ch2).Run(func(args Arguments) {
		ch1 <- time.Now()
	})

	mockedService.Mock.On("TheExampleMethod2", false).Return().WaitUntil(ch1)

	// Lock both goroutines on the .WaitUntil method
	go func() {
		mockedService.TheExampleMethod2(false)
	}()
	go func() {
		mockedService.TheExampleMethod2(true)
	}()

	// Allow the first call to execute, so the second one executes afterwards
	ch2 <- time.Now()
}

func Test_MockMethodCalled(t *testing.T) {
	m := new(Mock)
	m.On("foo", "hello").Return("world")

	retArgs := m.MethodCalled("foo", "hello")
	require.True(t, len(retArgs) == 1)
	require.Equal(t, "world", retArgs[0])
	m.AssertExpectations(t)
}

func Test_MockMethodCalled_Panic(t *testing.T) {
	m := new(Mock)
	m.On("foo", "hello").Panic("world panics")

	require.PanicsWithValue(t, "world panics", func() { m.MethodCalled("foo", "hello") })
	m.AssertExpectations(t)
}

// Test to validate fix for racy concurrent call access in MethodCalled()
func Test_MockReturnAndCalledConcurrent(t *testing.T) {
	iterations := 1000
	m := &Mock{}
	call := m.On("ConcurrencyTestMethod")

	wg := sync.WaitGroup{}
	wg.Add(2)

	go func() {
		for i := 0; i < iterations; i++ {
			call.Return(10)
		}
		wg.Done()
	}()
	go func() {
		for i := 0; i < iterations; i++ {
			ConcurrencyTestMethod(m)
		}
		wg.Done()
	}()
	wg.Wait()
}

type timer struct{ Mock }

func (s *timer) GetTime(i int) string {
	return s.Called(i).Get(0).(string)
}

type tCustomLogger struct {
	*testing.T
	logs []string
	errs []string
}

func (tc *tCustomLogger) Logf(format string, args ...interface{}) {
	tc.T.Logf(format, args...)
	tc.logs = append(tc.logs, fmt.Sprintf(format, args...))
}

func (tc *tCustomLogger) Errorf(format string, args ...interface{}) {
	tc.errs = append(tc.errs, fmt.Sprintf(format, args...))
}

func (tc *tCustomLogger) FailNow() {}

func TestLoggingAssertExpectations(t *testing.T) {
	m := new(timer)
	m.On("GetTime", 0).Return("")
	tcl := &tCustomLogger{t, []string{}, []string{}}

	AssertExpectationsForObjects(tcl, m, new(TestExampleImplementation))

	require.Equal(t, 1, len(tcl.errs))
	assert.Regexp(t, regexp.MustCompile("(?s)FAIL: 0 out of 1 expectation\\(s\\) were met.*The code you are testing needs to make 1 more call\\(s\\).*"), tcl.errs[0])
	require.Equal(t, 2, len(tcl.logs))
	assert.Regexp(t, regexp.MustCompile("(?s)FAIL:\tGetTime\\(int\\).*"), tcl.logs[0])
	require.Equal(t, "Expectations didn't match for Mock: *mock.timer", tcl.logs[1])
}

func TestAfterTotalWaitTimeWhileExecution(t *testing.T) {
	waitDuration := 1
	total, waitMs := 5, time.Millisecond*time.Duration(waitDuration)
	aTimer := new(timer)
	for i := 0; i < total; i++ {
		aTimer.On("GetTime", i).After(waitMs).Return(fmt.Sprintf("Time%d", i)).Once()
	}
	time.Sleep(waitMs)
	start := time.Now()
	var results []string

	for i := 0; i < total; i++ {
		results = append(results, aTimer.GetTime(i))
	}

	end := time.Now()
	elapsedTime := end.Sub(start)
	assert.True(t, elapsedTime > waitMs, fmt.Sprintf("Total elapsed time:%v should be atleast greater than %v", elapsedTime, waitMs))
	assert.Equal(t, total, len(results))
	for i := range results {
		assert.Equal(t, fmt.Sprintf("Time%d", i), results[i], "Return value of method should be same")
	}
}

func TestArgumentMatcherToPrintMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			matchingExp := regexp.MustCompile(
				`\s+mock: Unexpected Method Call\s+-*\s+GetTime\(int\)\s+0: 1\s+The closest call I have is:\s+GetTime\(mock.argumentMatcher\)\s+0: mock.argumentMatcher\{.*?\}\s+Diff:.*\(int=1\) not matched by func\(int\) bool`)
			assert.Regexp(t, matchingExp, r)
		}
	}()

	m := new(timer)
	m.On("GetTime", MatchedBy(func(i int) bool { return false })).Return("SomeTime").Once()

	res := m.GetTime(1)
	require.Equal(t, "SomeTime", res)
	m.AssertExpectations(t)
}

func TestClosestCallMismatchedArgumentInformationShowsTheClosest(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			matchingExp := regexp.MustCompile(unexpectedCallRegex(`TheExampleMethod(int,int,int)`, `0: 1\s+1: 1\s+2: 2`, `0: 1\s+1: 1\s+2: 1`, `0: PASS:  \(int=1\) == \(int=1\)\s+1: PASS:  \(int=1\) == \(int=1\)\s+2: FAIL:  \(int=2\) != \(int=1\)`))
			assert.Regexp(t, matchingExp, r)
		}
	}()

	m := new(TestExampleImplementation)
	m.On("TheExampleMethod", 1, 1, 1).Return(1, nil).Once()
	m.On("TheExampleMethod", 2, 2, 2).Return(2, nil).Once()

	m.TheExampleMethod(1, 1, 2)
}

func TestClosestCallMismatchedArgumentValueInformation(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			matchingExp := regexp.MustCompile(unexpectedCallRegex(`GetTime(int)`, "0: 1", "0: 999", `0: FAIL:  \(int=1\) != \(int=999\)`))
			assert.Regexp(t, matchingExp, r)
		}
	}()

	m := new(timer)
	m.On("GetTime", 999).Return("SomeTime").Once()

	_ = m.GetTime(1)
}

func unexpectedCallRegex(method, calledArg, expectedArg, diff string) string {
	rMethod := regexp.QuoteMeta(method)
	return fmt.Sprintf(`\s+mock: Unexpected Method Call\s+-*\s+%s\s+%s\s+The closest call I have is:\s+%s\s+%s\s+Diff: %s`,
		rMethod, calledArg, rMethod, expectedArg, diff)
}

//go:noinline
func ConcurrencyTestMethod(m *Mock) {
	m.Called()
}
