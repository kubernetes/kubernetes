package mock

import (
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/pmezard/go-difflib/difflib"
	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
)

// TestingT is an interface wrapper around *testing.T
type TestingT interface {
	Logf(format string, args ...interface{})
	Errorf(format string, args ...interface{})
	FailNow()
}

/*
	Call
*/

// Call represents a method call and is used for setting expectations,
// as well as recording activity.
type Call struct {
	Parent *Mock

	// The name of the method that was or will be called.
	Method string

	// Holds the arguments of the method.
	Arguments Arguments

	// Holds the arguments that should be returned when
	// this method is called.
	ReturnArguments Arguments

	// Holds the caller info for the On() call
	callerInfo []string

	// The number of times to return the return arguments when setting
	// expectations. 0 means to always return the value.
	Repeatability int

	// Amount of times this call has been called
	totalCalls int

	// Call to this method can be optional
	optional bool

	// Holds a channel that will be used to block the Return until it either
	// receives a message or is closed. nil means it returns immediately.
	WaitFor <-chan time.Time

	waitTime time.Duration

	// Holds a handler used to manipulate arguments content that are passed by
	// reference. It's useful when mocking methods such as unmarshalers or
	// decoders.
	RunFn func(Arguments)

	// PanicMsg holds msg to be used to mock panic on the function call
	//  if the PanicMsg is set to a non nil string the function call will panic
	// irrespective of other settings
	PanicMsg *string
}

func newCall(parent *Mock, methodName string, callerInfo []string, methodArguments ...interface{}) *Call {
	return &Call{
		Parent:          parent,
		Method:          methodName,
		Arguments:       methodArguments,
		ReturnArguments: make([]interface{}, 0),
		callerInfo:      callerInfo,
		Repeatability:   0,
		WaitFor:         nil,
		RunFn:           nil,
		PanicMsg:        nil,
	}
}

func (c *Call) lock() {
	c.Parent.mutex.Lock()
}

func (c *Call) unlock() {
	c.Parent.mutex.Unlock()
}

// Return specifies the return arguments for the expectation.
//
//    Mock.On("DoSomething").Return(errors.New("failed"))
func (c *Call) Return(returnArguments ...interface{}) *Call {
	c.lock()
	defer c.unlock()

	c.ReturnArguments = returnArguments

	return c
}

// Panic specifies if the functon call should fail and the panic message
//
//    Mock.On("DoSomething").Panic("test panic")
func (c *Call) Panic(msg string) *Call {
	c.lock()
	defer c.unlock()

	c.PanicMsg = &msg

	return c
}

// Once indicates that that the mock should only return the value once.
//
//    Mock.On("MyMethod", arg1, arg2).Return(returnArg1, returnArg2).Once()
func (c *Call) Once() *Call {
	return c.Times(1)
}

// Twice indicates that that the mock should only return the value twice.
//
//    Mock.On("MyMethod", arg1, arg2).Return(returnArg1, returnArg2).Twice()
func (c *Call) Twice() *Call {
	return c.Times(2)
}

// Times indicates that that the mock should only return the indicated number
// of times.
//
//    Mock.On("MyMethod", arg1, arg2).Return(returnArg1, returnArg2).Times(5)
func (c *Call) Times(i int) *Call {
	c.lock()
	defer c.unlock()
	c.Repeatability = i
	return c
}

// WaitUntil sets the channel that will block the mock's return until its closed
// or a message is received.
//
//    Mock.On("MyMethod", arg1, arg2).WaitUntil(time.After(time.Second))
func (c *Call) WaitUntil(w <-chan time.Time) *Call {
	c.lock()
	defer c.unlock()
	c.WaitFor = w
	return c
}

// After sets how long to block until the call returns
//
//    Mock.On("MyMethod", arg1, arg2).After(time.Second)
func (c *Call) After(d time.Duration) *Call {
	c.lock()
	defer c.unlock()
	c.waitTime = d
	return c
}

// Run sets a handler to be called before returning. It can be used when
// mocking a method (such as an unmarshaler) that takes a pointer to a struct and
// sets properties in such struct
//
//    Mock.On("Unmarshal", AnythingOfType("*map[string]interface{}")).Return().Run(func(args Arguments) {
//    	arg := args.Get(0).(*map[string]interface{})
//    	arg["foo"] = "bar"
//    })
func (c *Call) Run(fn func(args Arguments)) *Call {
	c.lock()
	defer c.unlock()
	c.RunFn = fn
	return c
}

// Maybe allows the method call to be optional. Not calling an optional method
// will not cause an error while asserting expectations
func (c *Call) Maybe() *Call {
	c.lock()
	defer c.unlock()
	c.optional = true
	return c
}

// On chains a new expectation description onto the mocked interface. This
// allows syntax like.
//
//    Mock.
//       On("MyMethod", 1).Return(nil).
//       On("MyOtherMethod", 'a', 'b', 'c').Return(errors.New("Some Error"))
//go:noinline
func (c *Call) On(methodName string, arguments ...interface{}) *Call {
	return c.Parent.On(methodName, arguments...)
}

// Mock is the workhorse used to track activity on another object.
// For an example of its usage, refer to the "Example Usage" section at the top
// of this document.
type Mock struct {
	// Represents the calls that are expected of
	// an object.
	ExpectedCalls []*Call

	// Holds the calls that were made to this mocked object.
	Calls []Call

	// test is An optional variable that holds the test struct, to be used when an
	// invalid mock call was made.
	test TestingT

	// TestData holds any data that might be useful for testing.  Testify ignores
	// this data completely allowing you to do whatever you like with it.
	testData objx.Map

	mutex sync.Mutex
}

// TestData holds any data that might be useful for testing.  Testify ignores
// this data completely allowing you to do whatever you like with it.
func (m *Mock) TestData() objx.Map {

	if m.testData == nil {
		m.testData = make(objx.Map)
	}

	return m.testData
}

/*
	Setting expectations
*/

// Test sets the test struct variable of the mock object
func (m *Mock) Test(t TestingT) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.test = t
}

// fail fails the current test with the given formatted format and args.
// In case that a test was defined, it uses the test APIs for failing a test,
// otherwise it uses panic.
func (m *Mock) fail(format string, args ...interface{}) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.test == nil {
		panic(fmt.Sprintf(format, args...))
	}
	m.test.Errorf(format, args...)
	m.test.FailNow()
}

// On starts a description of an expectation of the specified method
// being called.
//
//     Mock.On("MyMethod", arg1, arg2)
func (m *Mock) On(methodName string, arguments ...interface{}) *Call {
	for _, arg := range arguments {
		if v := reflect.ValueOf(arg); v.Kind() == reflect.Func {
			panic(fmt.Sprintf("cannot use Func in expectations. Use mock.AnythingOfType(\"%T\")", arg))
		}
	}

	m.mutex.Lock()
	defer m.mutex.Unlock()
	c := newCall(m, methodName, assert.CallerInfo(), arguments...)
	m.ExpectedCalls = append(m.ExpectedCalls, c)
	return c
}

// /*
// 	Recording and responding to activity
// */

func (m *Mock) findExpectedCall(method string, arguments ...interface{}) (int, *Call) {
	var expectedCall *Call

	for i, call := range m.ExpectedCalls {
		if call.Method == method {
			_, diffCount := call.Arguments.Diff(arguments)
			if diffCount == 0 {
				expectedCall = call
				if call.Repeatability > -1 {
					return i, call
				}
			}
		}
	}

	return -1, expectedCall
}

type matchCandidate struct {
	call      *Call
	mismatch  string
	diffCount int
}

func (c matchCandidate) isBetterMatchThan(other matchCandidate) bool {
	if c.call == nil {
		return false
	}
	if other.call == nil {
		return true
	}

	if c.diffCount > other.diffCount {
		return false
	}
	if c.diffCount < other.diffCount {
		return true
	}

	if c.call.Repeatability > 0 && other.call.Repeatability <= 0 {
		return true
	}
	return false
}

func (m *Mock) findClosestCall(method string, arguments ...interface{}) (*Call, string) {
	var bestMatch matchCandidate

	for _, call := range m.expectedCalls() {
		if call.Method == method {

			errInfo, tempDiffCount := call.Arguments.Diff(arguments)
			tempCandidate := matchCandidate{
				call:      call,
				mismatch:  errInfo,
				diffCount: tempDiffCount,
			}
			if tempCandidate.isBetterMatchThan(bestMatch) {
				bestMatch = tempCandidate
			}
		}
	}

	return bestMatch.call, bestMatch.mismatch
}

func callString(method string, arguments Arguments, includeArgumentValues bool) string {

	var argValsString string
	if includeArgumentValues {
		var argVals []string
		for argIndex, arg := range arguments {
			argVals = append(argVals, fmt.Sprintf("%d: %#v", argIndex, arg))
		}
		argValsString = fmt.Sprintf("\n\t\t%s", strings.Join(argVals, "\n\t\t"))
	}

	return fmt.Sprintf("%s(%s)%s", method, arguments.String(), argValsString)
}

// Called tells the mock object that a method has been called, and gets an array
// of arguments to return.  Panics if the call is unexpected (i.e. not preceded by
// appropriate .On .Return() calls)
// If Call.WaitFor is set, blocks until the channel is closed or receives a message.
func (m *Mock) Called(arguments ...interface{}) Arguments {
	// get the calling function's name
	pc, _, _, ok := runtime.Caller(1)
	if !ok {
		panic("Couldn't get the caller information")
	}
	functionPath := runtime.FuncForPC(pc).Name()
	//Next four lines are required to use GCCGO function naming conventions.
	//For Ex:  github_com_docker_libkv_store_mock.WatchTree.pN39_github_com_docker_libkv_store_mock.Mock
	//uses interface information unlike golang github.com/docker/libkv/store/mock.(*Mock).WatchTree
	//With GCCGO we need to remove interface information starting from pN<dd>.
	re := regexp.MustCompile("\\.pN\\d+_")
	if re.MatchString(functionPath) {
		functionPath = re.Split(functionPath, -1)[0]
	}
	parts := strings.Split(functionPath, ".")
	functionName := parts[len(parts)-1]
	return m.MethodCalled(functionName, arguments...)
}

// MethodCalled tells the mock object that the given method has been called, and gets
// an array of arguments to return. Panics if the call is unexpected (i.e. not preceded
// by appropriate .On .Return() calls)
// If Call.WaitFor is set, blocks until the channel is closed or receives a message.
func (m *Mock) MethodCalled(methodName string, arguments ...interface{}) Arguments {
	m.mutex.Lock()
	//TODO: could combine expected and closes in single loop
	found, call := m.findExpectedCall(methodName, arguments...)

	if found < 0 {
		// expected call found but it has already been called with repeatable times
		if call != nil {
			m.mutex.Unlock()
			m.fail("\nassert: mock: The method has been called over %d times.\n\tEither do one more Mock.On(\"%s\").Return(...), or remove extra call.\n\tThis call was unexpected:\n\t\t%s\n\tat: %s", call.totalCalls, methodName, callString(methodName, arguments, true), assert.CallerInfo())
		}
		// we have to fail here - because we don't know what to do
		// as the return arguments.  This is because:
		//
		//   a) this is a totally unexpected call to this method,
		//   b) the arguments are not what was expected, or
		//   c) the developer has forgotten to add an accompanying On...Return pair.
		closestCall, mismatch := m.findClosestCall(methodName, arguments...)
		m.mutex.Unlock()

		if closestCall != nil {
			m.fail("\n\nmock: Unexpected Method Call\n-----------------------------\n\n%s\n\nThe closest call I have is: \n\n%s\n\n%s\nDiff: %s",
				callString(methodName, arguments, true),
				callString(methodName, closestCall.Arguments, true),
				diffArguments(closestCall.Arguments, arguments),
				strings.TrimSpace(mismatch),
			)
		} else {
			m.fail("\nassert: mock: I don't know what to return because the method call was unexpected.\n\tEither do Mock.On(\"%s\").Return(...) first, or remove the %s() call.\n\tThis method was unexpected:\n\t\t%s\n\tat: %s", methodName, methodName, callString(methodName, arguments, true), assert.CallerInfo())
		}
	}

	if call.Repeatability == 1 {
		call.Repeatability = -1
	} else if call.Repeatability > 1 {
		call.Repeatability--
	}
	call.totalCalls++

	// add the call
	m.Calls = append(m.Calls, *newCall(m, methodName, assert.CallerInfo(), arguments...))
	m.mutex.Unlock()

	// block if specified
	if call.WaitFor != nil {
		<-call.WaitFor
	} else {
		time.Sleep(call.waitTime)
	}

	m.mutex.Lock()
	panicMsg := call.PanicMsg
	m.mutex.Unlock()
	if panicMsg != nil {
		panic(*panicMsg)
	}

	m.mutex.Lock()
	runFn := call.RunFn
	m.mutex.Unlock()

	if runFn != nil {
		runFn(arguments)
	}

	m.mutex.Lock()
	returnArgs := call.ReturnArguments
	m.mutex.Unlock()

	return returnArgs
}

/*
	Assertions
*/

type assertExpectationser interface {
	AssertExpectations(TestingT) bool
}

// AssertExpectationsForObjects asserts that everything specified with On and Return
// of the specified objects was in fact called as expected.
//
// Calls may have occurred in any order.
func AssertExpectationsForObjects(t TestingT, testObjects ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	for _, obj := range testObjects {
		if m, ok := obj.(Mock); ok {
			t.Logf("Deprecated mock.AssertExpectationsForObjects(myMock.Mock) use mock.AssertExpectationsForObjects(myMock)")
			obj = &m
		}
		m := obj.(assertExpectationser)
		if !m.AssertExpectations(t) {
			t.Logf("Expectations didn't match for Mock: %+v", reflect.TypeOf(m))
			return false
		}
	}
	return true
}

// AssertExpectations asserts that everything specified with On and Return was
// in fact called as expected.  Calls may have occurred in any order.
func (m *Mock) AssertExpectations(t TestingT) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()
	var somethingMissing bool
	var failedExpectations int

	// iterate through each expectation
	expectedCalls := m.expectedCalls()
	for _, expectedCall := range expectedCalls {
		if !expectedCall.optional && !m.methodWasCalled(expectedCall.Method, expectedCall.Arguments) && expectedCall.totalCalls == 0 {
			somethingMissing = true
			failedExpectations++
			t.Logf("FAIL:\t%s(%s)\n\t\tat: %s", expectedCall.Method, expectedCall.Arguments.String(), expectedCall.callerInfo)
		} else {
			if expectedCall.Repeatability > 0 {
				somethingMissing = true
				failedExpectations++
				t.Logf("FAIL:\t%s(%s)\n\t\tat: %s", expectedCall.Method, expectedCall.Arguments.String(), expectedCall.callerInfo)
			} else {
				t.Logf("PASS:\t%s(%s)", expectedCall.Method, expectedCall.Arguments.String())
			}
		}
	}

	if somethingMissing {
		t.Errorf("FAIL: %d out of %d expectation(s) were met.\n\tThe code you are testing needs to make %d more call(s).\n\tat: %s", len(expectedCalls)-failedExpectations, len(expectedCalls), failedExpectations, assert.CallerInfo())
	}

	return !somethingMissing
}

// AssertNumberOfCalls asserts that the method was called expectedCalls times.
func (m *Mock) AssertNumberOfCalls(t TestingT, methodName string, expectedCalls int) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()
	var actualCalls int
	for _, call := range m.calls() {
		if call.Method == methodName {
			actualCalls++
		}
	}
	return assert.Equal(t, expectedCalls, actualCalls, fmt.Sprintf("Expected number of calls (%d) does not match the actual number of calls (%d).", expectedCalls, actualCalls))
}

// AssertCalled asserts that the method was called.
// It can produce a false result when an argument is a pointer type and the underlying value changed after calling the mocked method.
func (m *Mock) AssertCalled(t TestingT, methodName string, arguments ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()
	if !m.methodWasCalled(methodName, arguments) {
		var calledWithArgs []string
		for _, call := range m.calls() {
			calledWithArgs = append(calledWithArgs, fmt.Sprintf("%v", call.Arguments))
		}
		if len(calledWithArgs) == 0 {
			return assert.Fail(t, "Should have called with given arguments",
				fmt.Sprintf("Expected %q to have been called with:\n%v\nbut no actual calls happened", methodName, arguments))
		}
		return assert.Fail(t, "Should have called with given arguments",
			fmt.Sprintf("Expected %q to have been called with:\n%v\nbut actual calls were:\n        %v", methodName, arguments, strings.Join(calledWithArgs, "\n")))
	}
	return true
}

// AssertNotCalled asserts that the method was not called.
// It can produce a false result when an argument is a pointer type and the underlying value changed after calling the mocked method.
func (m *Mock) AssertNotCalled(t TestingT, methodName string, arguments ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()
	if m.methodWasCalled(methodName, arguments) {
		return assert.Fail(t, "Should not have called with given arguments",
			fmt.Sprintf("Expected %q to not have been called with:\n%v\nbut actually it was.", methodName, arguments))
	}
	return true
}

// IsMethodCallable checking that the method can be called
// If the method was called more than `Repeatability` return false
func (m *Mock) IsMethodCallable(t TestingT, methodName string, arguments ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for _, v := range m.ExpectedCalls {
		if v.Method != methodName {
			continue
		}
		if len(arguments) != len(v.Arguments) {
			continue
		}
		if v.Repeatability < v.totalCalls {
			continue
		}
		if isArgsEqual(v.Arguments, arguments) {
			return true
		}
	}
	return false
}

// isArgsEqual compares arguments
func isArgsEqual(expected Arguments, args []interface{}) bool {
	if len(expected) != len(args) {
		return false
	}
	for i, v := range args {
		if !reflect.DeepEqual(expected[i], v) {
			return false
		}
	}
	return true
}

func (m *Mock) methodWasCalled(methodName string, expected []interface{}) bool {
	for _, call := range m.calls() {
		if call.Method == methodName {

			_, differences := Arguments(expected).Diff(call.Arguments)

			if differences == 0 {
				// found the expected call
				return true
			}

		}
	}
	// we didn't find the expected call
	return false
}

func (m *Mock) expectedCalls() []*Call {
	return append([]*Call{}, m.ExpectedCalls...)
}

func (m *Mock) calls() []Call {
	return append([]Call{}, m.Calls...)
}

/*
	Arguments
*/

// Arguments holds an array of method arguments or return values.
type Arguments []interface{}

const (
	// Anything is used in Diff and Assert when the argument being tested
	// shouldn't be taken into consideration.
	Anything = "mock.Anything"
)

// AnythingOfTypeArgument is a string that contains the type of an argument
// for use when type checking.  Used in Diff and Assert.
type AnythingOfTypeArgument string

// AnythingOfType returns an AnythingOfTypeArgument object containing the
// name of the type to check for.  Used in Diff and Assert.
//
// For example:
//	Assert(t, AnythingOfType("string"), AnythingOfType("int"))
func AnythingOfType(t string) AnythingOfTypeArgument {
	return AnythingOfTypeArgument(t)
}

// IsTypeArgument is a struct that contains the type of an argument
// for use when type checking.  This is an alternative to AnythingOfType.
// Used in Diff and Assert.
type IsTypeArgument struct {
	t interface{}
}

// IsType returns an IsTypeArgument object containing the type to check for.
// You can provide a zero-value of the type to check.  This is an
// alternative to AnythingOfType.  Used in Diff and Assert.
//
// For example:
// Assert(t, IsType(""), IsType(0))
func IsType(t interface{}) *IsTypeArgument {
	return &IsTypeArgument{t: t}
}

// argumentMatcher performs custom argument matching, returning whether or
// not the argument is matched by the expectation fixture function.
type argumentMatcher struct {
	// fn is a function which accepts one argument, and returns a bool.
	fn reflect.Value
}

func (f argumentMatcher) Matches(argument interface{}) bool {
	expectType := f.fn.Type().In(0)
	expectTypeNilSupported := false
	switch expectType.Kind() {
	case reflect.Interface, reflect.Chan, reflect.Func, reflect.Map, reflect.Slice, reflect.Ptr:
		expectTypeNilSupported = true
	}

	argType := reflect.TypeOf(argument)
	var arg reflect.Value
	if argType == nil {
		arg = reflect.New(expectType).Elem()
	} else {
		arg = reflect.ValueOf(argument)
	}

	if argType == nil && !expectTypeNilSupported {
		panic(errors.New("attempting to call matcher with nil for non-nil expected type"))
	}
	if argType == nil || argType.AssignableTo(expectType) {
		result := f.fn.Call([]reflect.Value{arg})
		return result[0].Bool()
	}
	return false
}

func (f argumentMatcher) String() string {
	return fmt.Sprintf("func(%s) bool", f.fn.Type().In(0).Name())
}

// MatchedBy can be used to match a mock call based on only certain properties
// from a complex struct or some calculation. It takes a function that will be
// evaluated with the called argument and will return true when there's a match
// and false otherwise.
//
// Example:
// m.On("Do", MatchedBy(func(req *http.Request) bool { return req.Host == "example.com" }))
//
// |fn|, must be a function accepting a single argument (of the expected type)
// which returns a bool. If |fn| doesn't match the required signature,
// MatchedBy() panics.
func MatchedBy(fn interface{}) argumentMatcher {
	fnType := reflect.TypeOf(fn)

	if fnType.Kind() != reflect.Func {
		panic(fmt.Sprintf("assert: arguments: %s is not a func", fn))
	}
	if fnType.NumIn() != 1 {
		panic(fmt.Sprintf("assert: arguments: %s does not take exactly one argument", fn))
	}
	if fnType.NumOut() != 1 || fnType.Out(0).Kind() != reflect.Bool {
		panic(fmt.Sprintf("assert: arguments: %s does not return a bool", fn))
	}

	return argumentMatcher{fn: reflect.ValueOf(fn)}
}

// Get Returns the argument at the specified index.
func (args Arguments) Get(index int) interface{} {
	if index+1 > len(args) {
		panic(fmt.Sprintf("assert: arguments: Cannot call Get(%d) because there are %d argument(s).", index, len(args)))
	}
	return args[index]
}

// Is gets whether the objects match the arguments specified.
func (args Arguments) Is(objects ...interface{}) bool {
	for i, obj := range args {
		if obj != objects[i] {
			return false
		}
	}
	return true
}

// Diff gets a string describing the differences between the arguments
// and the specified objects.
//
// Returns the diff string and number of differences found.
func (args Arguments) Diff(objects []interface{}) (string, int) {
	//TODO: could return string as error and nil for No difference

	var output = "\n"
	var differences int

	var maxArgCount = len(args)
	if len(objects) > maxArgCount {
		maxArgCount = len(objects)
	}

	for i := 0; i < maxArgCount; i++ {
		var actual, expected interface{}
		var actualFmt, expectedFmt string

		if len(objects) <= i {
			actual = "(Missing)"
			actualFmt = "(Missing)"
		} else {
			actual = objects[i]
			actualFmt = fmt.Sprintf("(%[1]T=%[1]v)", actual)
		}

		if len(args) <= i {
			expected = "(Missing)"
			expectedFmt = "(Missing)"
		} else {
			expected = args[i]
			expectedFmt = fmt.Sprintf("(%[1]T=%[1]v)", expected)
		}

		if matcher, ok := expected.(argumentMatcher); ok {
			if matcher.Matches(actual) {
				output = fmt.Sprintf("%s\t%d: PASS:  %s matched by %s\n", output, i, actualFmt, matcher)
			} else {
				differences++
				output = fmt.Sprintf("%s\t%d: FAIL:  %s not matched by %s\n", output, i, actualFmt, matcher)
			}
		} else if reflect.TypeOf(expected) == reflect.TypeOf((*AnythingOfTypeArgument)(nil)).Elem() {

			// type checking
			if reflect.TypeOf(actual).Name() != string(expected.(AnythingOfTypeArgument)) && reflect.TypeOf(actual).String() != string(expected.(AnythingOfTypeArgument)) {
				// not match
				differences++
				output = fmt.Sprintf("%s\t%d: FAIL:  type %s != type %s - %s\n", output, i, expected, reflect.TypeOf(actual).Name(), actualFmt)
			}

		} else if reflect.TypeOf(expected) == reflect.TypeOf((*IsTypeArgument)(nil)) {
			t := expected.(*IsTypeArgument).t
			if reflect.TypeOf(t) != reflect.TypeOf(actual) {
				differences++
				output = fmt.Sprintf("%s\t%d: FAIL:  type %s != type %s - %s\n", output, i, reflect.TypeOf(t).Name(), reflect.TypeOf(actual).Name(), actualFmt)
			}
		} else {

			// normal checking

			if assert.ObjectsAreEqual(expected, Anything) || assert.ObjectsAreEqual(actual, Anything) || assert.ObjectsAreEqual(actual, expected) {
				// match
				output = fmt.Sprintf("%s\t%d: PASS:  %s == %s\n", output, i, actualFmt, expectedFmt)
			} else {
				// not match
				differences++
				output = fmt.Sprintf("%s\t%d: FAIL:  %s != %s\n", output, i, actualFmt, expectedFmt)
			}
		}

	}

	if differences == 0 {
		return "No differences.", differences
	}

	return output, differences

}

// Assert compares the arguments with the specified objects and fails if
// they do not exactly match.
func (args Arguments) Assert(t TestingT, objects ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}

	// get the differences
	diff, diffCount := args.Diff(objects)

	if diffCount == 0 {
		return true
	}

	// there are differences... report them...
	t.Logf(diff)
	t.Errorf("%sArguments do not match.", assert.CallerInfo())

	return false

}

// String gets the argument at the specified index. Panics if there is no argument, or
// if the argument is of the wrong type.
//
// If no index is provided, String() returns a complete string representation
// of the arguments.
func (args Arguments) String(indexOrNil ...int) string {

	if len(indexOrNil) == 0 {
		// normal String() method - return a string representation of the args
		var argsStr []string
		for _, arg := range args {
			argsStr = append(argsStr, fmt.Sprintf("%T", arg)) // handles nil nicely
		}
		return strings.Join(argsStr, ",")
	} else if len(indexOrNil) == 1 {
		// Index has been specified - get the argument at that index
		var index = indexOrNil[0]
		var s string
		var ok bool
		if s, ok = args.Get(index).(string); !ok {
			panic(fmt.Sprintf("assert: arguments: String(%d) failed because object wasn't correct type: %s", index, args.Get(index)))
		}
		return s
	}

	panic(fmt.Sprintf("assert: arguments: Wrong number of arguments passed to String.  Must be 0 or 1, not %d", len(indexOrNil)))

}

// Int gets the argument at the specified index. Panics if there is no argument, or
// if the argument is of the wrong type.
func (args Arguments) Int(index int) int {
	var s int
	var ok bool
	if s, ok = args.Get(index).(int); !ok {
		panic(fmt.Sprintf("assert: arguments: Int(%d) failed because object wasn't correct type: %v", index, args.Get(index)))
	}
	return s
}

// Error gets the argument at the specified index. Panics if there is no argument, or
// if the argument is of the wrong type.
func (args Arguments) Error(index int) error {
	obj := args.Get(index)
	var s error
	var ok bool
	if obj == nil {
		return nil
	}
	if s, ok = obj.(error); !ok {
		panic(fmt.Sprintf("assert: arguments: Error(%d) failed because object wasn't correct type: %v", index, args.Get(index)))
	}
	return s
}

// Bool gets the argument at the specified index. Panics if there is no argument, or
// if the argument is of the wrong type.
func (args Arguments) Bool(index int) bool {
	var s bool
	var ok bool
	if s, ok = args.Get(index).(bool); !ok {
		panic(fmt.Sprintf("assert: arguments: Bool(%d) failed because object wasn't correct type: %v", index, args.Get(index)))
	}
	return s
}

func typeAndKind(v interface{}) (reflect.Type, reflect.Kind) {
	t := reflect.TypeOf(v)
	k := t.Kind()

	if k == reflect.Ptr {
		t = t.Elem()
		k = t.Kind()
	}
	return t, k
}

func diffArguments(expected Arguments, actual Arguments) string {
	if len(expected) != len(actual) {
		return fmt.Sprintf("Provided %v arguments, mocked for %v arguments", len(expected), len(actual))
	}

	for x := range expected {
		if diffString := diff(expected[x], actual[x]); diffString != "" {
			return fmt.Sprintf("Difference found in argument %v:\n\n%s", x, diffString)
		}
	}

	return ""
}

// diff returns a diff of both values as long as both are of the same type and
// are a struct, map, slice or array. Otherwise it returns an empty string.
func diff(expected interface{}, actual interface{}) string {
	if expected == nil || actual == nil {
		return ""
	}

	et, ek := typeAndKind(expected)
	at, _ := typeAndKind(actual)

	if et != at {
		return ""
	}

	if ek != reflect.Struct && ek != reflect.Map && ek != reflect.Slice && ek != reflect.Array {
		return ""
	}

	e := spewConfig.Sdump(expected)
	a := spewConfig.Sdump(actual)

	diff, _ := difflib.GetUnifiedDiffString(difflib.UnifiedDiff{
		A:        difflib.SplitLines(e),
		B:        difflib.SplitLines(a),
		FromFile: "Expected",
		FromDate: "",
		ToFile:   "Actual",
		ToDate:   "",
		Context:  1,
	})

	return diff
}

var spewConfig = spew.ConfigState{
	Indent:                  " ",
	DisablePointerAddresses: true,
	DisableCapacities:       true,
	SortKeys:                true,
}

type tHelper interface {
	Helper()
}
