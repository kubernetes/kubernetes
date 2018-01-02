package check

import (
	"fmt"
	"reflect"
	"regexp"
)

// -----------------------------------------------------------------------
// CommentInterface and Commentf helper, to attach extra information to checks.

type comment struct {
	format string
	args   []interface{}
}

// Commentf returns an infomational value to use with Assert or Check calls.
// If the checker test fails, the provided arguments will be passed to
// fmt.Sprintf, and will be presented next to the logged failure.
//
// For example:
//
//     c.Assert(v, Equals, 42, Commentf("Iteration #%d failed.", i))
//
// Note that if the comment is constant, a better option is to
// simply use a normal comment right above or next to the line, as
// it will also get printed with any errors:
//
//     c.Assert(l, Equals, 8192) // Ensure buffer size is correct (bug #123)
//
func Commentf(format string, args ...interface{}) CommentInterface {
	return &comment{format, args}
}

// CommentInterface must be implemented by types that attach extra
// information to failed checks. See the Commentf function for details.
type CommentInterface interface {
	CheckCommentString() string
}

func (c *comment) CheckCommentString() string {
	return fmt.Sprintf(c.format, c.args...)
}

// -----------------------------------------------------------------------
// The Checker interface.

// The Checker interface must be provided by checkers used with
// the Assert and Check verification methods.
type Checker interface {
	Info() *CheckerInfo
	Check(params []interface{}, names []string) (result bool, error string)
}

// See the Checker interface.
type CheckerInfo struct {
	Name   string
	Params []string
}

func (info *CheckerInfo) Info() *CheckerInfo {
	return info
}

// -----------------------------------------------------------------------
// Not checker logic inverter.

// The Not checker inverts the logic of the provided checker.  The
// resulting checker will succeed where the original one failed, and
// vice-versa.
//
// For example:
//
//     c.Assert(a, Not(Equals), b)
//
func Not(checker Checker) Checker {
	return &notChecker{checker}
}

type notChecker struct {
	sub Checker
}

func (checker *notChecker) Info() *CheckerInfo {
	info := *checker.sub.Info()
	info.Name = "Not(" + info.Name + ")"
	return &info
}

func (checker *notChecker) Check(params []interface{}, names []string) (result bool, error string) {
	result, error = checker.sub.Check(params, names)
	result = !result
	return
}

// -----------------------------------------------------------------------
// IsNil checker.

type isNilChecker struct {
	*CheckerInfo
}

// The IsNil checker tests whether the obtained value is nil.
//
// For example:
//
//    c.Assert(err, IsNil)
//
var IsNil Checker = &isNilChecker{
	&CheckerInfo{Name: "IsNil", Params: []string{"value"}},
}

func (checker *isNilChecker) Check(params []interface{}, names []string) (result bool, error string) {
	return isNil(params[0]), ""
}

func isNil(obtained interface{}) (result bool) {
	if obtained == nil {
		result = true
	} else {
		switch v := reflect.ValueOf(obtained); v.Kind() {
		case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
			return v.IsNil()
		}
	}
	return
}

// -----------------------------------------------------------------------
// NotNil checker. Alias for Not(IsNil), since it's so common.

type notNilChecker struct {
	*CheckerInfo
}

// The NotNil checker verifies that the obtained value is not nil.
//
// For example:
//
//     c.Assert(iface, NotNil)
//
// This is an alias for Not(IsNil), made available since it's a
// fairly common check.
//
var NotNil Checker = &notNilChecker{
	&CheckerInfo{Name: "NotNil", Params: []string{"value"}},
}

func (checker *notNilChecker) Check(params []interface{}, names []string) (result bool, error string) {
	return !isNil(params[0]), ""
}

// -----------------------------------------------------------------------
// Equals checker.

type equalsChecker struct {
	*CheckerInfo
}

// The Equals checker verifies that the obtained value is equal to
// the expected value, according to usual Go semantics for ==.
//
// For example:
//
//     c.Assert(value, Equals, 42)
//
var Equals Checker = &equalsChecker{
	&CheckerInfo{Name: "Equals", Params: []string{"obtained", "expected"}},
}

func (checker *equalsChecker) Check(params []interface{}, names []string) (result bool, error string) {
	defer func() {
		if v := recover(); v != nil {
			result = false
			error = fmt.Sprint(v)
		}
	}()
	return params[0] == params[1], ""
}

// -----------------------------------------------------------------------
// DeepEquals checker.

type deepEqualsChecker struct {
	*CheckerInfo
}

// The DeepEquals checker verifies that the obtained value is deep-equal to
// the expected value.  The check will work correctly even when facing
// slices, interfaces, and values of different types (which always fail
// the test).
//
// For example:
//
//     c.Assert(value, DeepEquals, 42)
//     c.Assert(array, DeepEquals, []string{"hi", "there"})
//
var DeepEquals Checker = &deepEqualsChecker{
	&CheckerInfo{Name: "DeepEquals", Params: []string{"obtained", "expected"}},
}

func (checker *deepEqualsChecker) Check(params []interface{}, names []string) (result bool, error string) {
	return reflect.DeepEqual(params[0], params[1]), ""
}

// -----------------------------------------------------------------------
// HasLen checker.

type hasLenChecker struct {
	*CheckerInfo
}

// The HasLen checker verifies that the obtained value has the
// provided length. In many cases this is superior to using Equals
// in conjuction with the len function because in case the check
// fails the value itself will be printed, instead of its length,
// providing more details for figuring the problem.
//
// For example:
//
//     c.Assert(list, HasLen, 5)
//
var HasLen Checker = &hasLenChecker{
	&CheckerInfo{Name: "HasLen", Params: []string{"obtained", "n"}},
}

func (checker *hasLenChecker) Check(params []interface{}, names []string) (result bool, error string) {
	n, ok := params[1].(int)
	if !ok {
		return false, "n must be an int"
	}
	value := reflect.ValueOf(params[0])
	switch value.Kind() {
	case reflect.Map, reflect.Array, reflect.Slice, reflect.Chan, reflect.String:
	default:
		return false, "obtained value type has no length"
	}
	return value.Len() == n, ""
}

// -----------------------------------------------------------------------
// ErrorMatches checker.

type errorMatchesChecker struct {
	*CheckerInfo
}

// The ErrorMatches checker verifies that the error value
// is non nil and matches the regular expression provided.
//
// For example:
//
//     c.Assert(err, ErrorMatches, "perm.*denied")
//
var ErrorMatches Checker = errorMatchesChecker{
	&CheckerInfo{Name: "ErrorMatches", Params: []string{"value", "regex"}},
}

func (checker errorMatchesChecker) Check(params []interface{}, names []string) (result bool, errStr string) {
	if params[0] == nil {
		return false, "Error value is nil"
	}
	err, ok := params[0].(error)
	if !ok {
		return false, "Value is not an error"
	}
	params[0] = err.Error()
	names[0] = "error"
	return matches(params[0], params[1])
}

// -----------------------------------------------------------------------
// Matches checker.

type matchesChecker struct {
	*CheckerInfo
}

// The Matches checker verifies that the string provided as the obtained
// value (or the string resulting from obtained.String()) matches the
// regular expression provided.
//
// For example:
//
//     c.Assert(err, Matches, "perm.*denied")
//
var Matches Checker = &matchesChecker{
	&CheckerInfo{Name: "Matches", Params: []string{"value", "regex"}},
}

func (checker *matchesChecker) Check(params []interface{}, names []string) (result bool, error string) {
	return matches(params[0], params[1])
}

func matches(value, regex interface{}) (result bool, error string) {
	reStr, ok := regex.(string)
	if !ok {
		return false, "Regex must be a string"
	}
	valueStr, valueIsStr := value.(string)
	if !valueIsStr {
		if valueWithStr, valueHasStr := value.(fmt.Stringer); valueHasStr {
			valueStr, valueIsStr = valueWithStr.String(), true
		}
	}
	if valueIsStr {
		matches, err := regexp.MatchString("^"+reStr+"$", valueStr)
		if err != nil {
			return false, "Can't compile regex: " + err.Error()
		}
		return matches, ""
	}
	return false, "Obtained value is not a string and has no .String()"
}

// -----------------------------------------------------------------------
// Panics checker.

type panicsChecker struct {
	*CheckerInfo
}

// The Panics checker verifies that calling the provided zero-argument
// function will cause a panic which is deep-equal to the provided value.
//
// For example:
//
//     c.Assert(func() { f(1, 2) }, Panics, &SomeErrorType{"BOOM"}).
//
//
var Panics Checker = &panicsChecker{
	&CheckerInfo{Name: "Panics", Params: []string{"function", "expected"}},
}

func (checker *panicsChecker) Check(params []interface{}, names []string) (result bool, error string) {
	f := reflect.ValueOf(params[0])
	if f.Kind() != reflect.Func || f.Type().NumIn() != 0 {
		return false, "Function must take zero arguments"
	}
	defer func() {
		// If the function has not panicked, then don't do the check.
		if error != "" {
			return
		}
		params[0] = recover()
		names[0] = "panic"
		result = reflect.DeepEqual(params[0], params[1])
	}()
	f.Call(nil)
	return false, "Function has not panicked"
}

type panicMatchesChecker struct {
	*CheckerInfo
}

// The PanicMatches checker verifies that calling the provided zero-argument
// function will cause a panic with an error value matching
// the regular expression provided.
//
// For example:
//
//     c.Assert(func() { f(1, 2) }, PanicMatches, `open.*: no such file or directory`).
//
//
var PanicMatches Checker = &panicMatchesChecker{
	&CheckerInfo{Name: "PanicMatches", Params: []string{"function", "expected"}},
}

func (checker *panicMatchesChecker) Check(params []interface{}, names []string) (result bool, errmsg string) {
	f := reflect.ValueOf(params[0])
	if f.Kind() != reflect.Func || f.Type().NumIn() != 0 {
		return false, "Function must take zero arguments"
	}
	defer func() {
		// If the function has not panicked, then don't do the check.
		if errmsg != "" {
			return
		}
		obtained := recover()
		names[0] = "panic"
		if e, ok := obtained.(error); ok {
			params[0] = e.Error()
		} else if _, ok := obtained.(string); ok {
			params[0] = obtained
		} else {
			errmsg = "Panic value is not a string or an error"
			return
		}
		result, errmsg = matches(params[0], params[1])
	}()
	f.Call(nil)
	return false, "Function has not panicked"
}

// -----------------------------------------------------------------------
// FitsTypeOf checker.

type fitsTypeChecker struct {
	*CheckerInfo
}

// The FitsTypeOf checker verifies that the obtained value is
// assignable to a variable with the same type as the provided
// sample value.
//
// For example:
//
//     c.Assert(value, FitsTypeOf, int64(0))
//     c.Assert(value, FitsTypeOf, os.Error(nil))
//
var FitsTypeOf Checker = &fitsTypeChecker{
	&CheckerInfo{Name: "FitsTypeOf", Params: []string{"obtained", "sample"}},
}

func (checker *fitsTypeChecker) Check(params []interface{}, names []string) (result bool, error string) {
	obtained := reflect.ValueOf(params[0])
	sample := reflect.ValueOf(params[1])
	if !obtained.IsValid() {
		return false, ""
	}
	if !sample.IsValid() {
		return false, "Invalid sample value"
	}
	return obtained.Type().AssignableTo(sample.Type()), ""
}

// -----------------------------------------------------------------------
// Implements checker.

type implementsChecker struct {
	*CheckerInfo
}

// The Implements checker verifies that the obtained value
// implements the interface specified via a pointer to an interface
// variable.
//
// For example:
//
//     var e os.Error
//     c.Assert(err, Implements, &e)
//
var Implements Checker = &implementsChecker{
	&CheckerInfo{Name: "Implements", Params: []string{"obtained", "ifaceptr"}},
}

func (checker *implementsChecker) Check(params []interface{}, names []string) (result bool, error string) {
	obtained := reflect.ValueOf(params[0])
	ifaceptr := reflect.ValueOf(params[1])
	if !obtained.IsValid() {
		return false, ""
	}
	if !ifaceptr.IsValid() || ifaceptr.Kind() != reflect.Ptr || ifaceptr.Elem().Kind() != reflect.Interface {
		return false, "ifaceptr should be a pointer to an interface variable"
	}
	return obtained.Type().Implements(ifaceptr.Elem().Type()), ""
}
