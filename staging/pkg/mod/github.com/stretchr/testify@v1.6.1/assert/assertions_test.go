package assert

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

var (
	i     interface{}
	zeros = []interface{}{
		false,
		byte(0),
		complex64(0),
		complex128(0),
		float32(0),
		float64(0),
		int(0),
		int8(0),
		int16(0),
		int32(0),
		int64(0),
		rune(0),
		uint(0),
		uint8(0),
		uint16(0),
		uint32(0),
		uint64(0),
		uintptr(0),
		"",
		[0]interface{}{},
		[]interface{}(nil),
		struct{ x int }{},
		(*interface{})(nil),
		(func())(nil),
		nil,
		interface{}(nil),
		map[interface{}]interface{}(nil),
		(chan interface{})(nil),
		(<-chan interface{})(nil),
		(chan<- interface{})(nil),
	}
	nonZeros = []interface{}{
		true,
		byte(1),
		complex64(1),
		complex128(1),
		float32(1),
		float64(1),
		int(1),
		int8(1),
		int16(1),
		int32(1),
		int64(1),
		rune(1),
		uint(1),
		uint8(1),
		uint16(1),
		uint32(1),
		uint64(1),
		uintptr(1),
		"s",
		[1]interface{}{1},
		[]interface{}{},
		struct{ x int }{1},
		(&i),
		(func() {}),
		interface{}(1),
		map[interface{}]interface{}{},
		(make(chan interface{})),
		(<-chan interface{})(make(chan interface{})),
		(chan<- interface{})(make(chan interface{})),
	}
)

// AssertionTesterInterface defines an interface to be used for testing assertion methods
type AssertionTesterInterface interface {
	TestMethod()
}

// AssertionTesterConformingObject is an object that conforms to the AssertionTesterInterface interface
type AssertionTesterConformingObject struct {
}

func (a *AssertionTesterConformingObject) TestMethod() {
}

// AssertionTesterNonConformingObject is an object that does not conform to the AssertionTesterInterface interface
type AssertionTesterNonConformingObject struct {
}

func TestObjectsAreEqual(t *testing.T) {

	if !ObjectsAreEqual("Hello World", "Hello World") {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(123, 123) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(123.5, 123.5) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual([]byte("Hello World"), []byte("Hello World")) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(nil, nil) {
		t.Error("objectsAreEqual should return true")
	}
	if ObjectsAreEqual(map[int]int{5: 10}, map[int]int{10: 20}) {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual('x', "x") {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual("x", 'x') {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual(0, 0.1) {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual(0.1, 0) {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual(time.Now, time.Now) {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual(func() {}, func() {}) {
		t.Error("objectsAreEqual should return false")
	}
	if ObjectsAreEqual(uint32(10), int32(10)) {
		t.Error("objectsAreEqual should return false")
	}
	if !ObjectsAreEqualValues(uint32(10), int32(10)) {
		t.Error("ObjectsAreEqualValues should return true")
	}
	if ObjectsAreEqualValues(0, nil) {
		t.Fail()
	}
	if ObjectsAreEqualValues(nil, 0) {
		t.Fail()
	}

}

func TestImplements(t *testing.T) {

	mockT := new(testing.T)

	if !Implements(mockT, (*AssertionTesterInterface)(nil), new(AssertionTesterConformingObject)) {
		t.Error("Implements method should return true: AssertionTesterConformingObject implements AssertionTesterInterface")
	}
	if Implements(mockT, (*AssertionTesterInterface)(nil), new(AssertionTesterNonConformingObject)) {
		t.Error("Implements method should return false: AssertionTesterNonConformingObject does not implements AssertionTesterInterface")
	}
	if Implements(mockT, (*AssertionTesterInterface)(nil), nil) {
		t.Error("Implements method should return false: nil does not implement AssertionTesterInterface")
	}

}

func TestIsType(t *testing.T) {

	mockT := new(testing.T)

	if !IsType(mockT, new(AssertionTesterConformingObject), new(AssertionTesterConformingObject)) {
		t.Error("IsType should return true: AssertionTesterConformingObject is the same type as AssertionTesterConformingObject")
	}
	if IsType(mockT, new(AssertionTesterConformingObject), new(AssertionTesterNonConformingObject)) {
		t.Error("IsType should return false: AssertionTesterConformingObject is not the same type as AssertionTesterNonConformingObject")
	}

}

type myType string

func TestEqual(t *testing.T) {

	mockT := new(testing.T)

	if !Equal(mockT, "Hello World", "Hello World") {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, 123, 123) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, 123.5, 123.5) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, []byte("Hello World"), []byte("Hello World")) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, nil, nil) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, int32(123), int32(123)) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, uint64(123), uint64(123)) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, myType("1"), myType("1")) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, &struct{}{}, &struct{}{}) {
		t.Error("Equal should return true (pointer equality is based on equality of underlying value)")
	}
	var m map[string]interface{}
	if Equal(mockT, m["bar"], "something") {
		t.Error("Equal should return false")
	}
	if Equal(mockT, myType("1"), myType("2")) {
		t.Error("Equal should return false")
	}
	// A case that might be confusing, especially with numeric literals
	if Equal(mockT, 10, uint(10)) {
		t.Error("Equal should return false")
	}
}

func ptr(i int) *int {
	return &i
}

func TestSame(t *testing.T) {

	mockT := new(testing.T)

	if Same(mockT, ptr(1), ptr(1)) {
		t.Error("Same should return false")
	}
	if Same(mockT, 1, 1) {
		t.Error("Same should return false")
	}
	p := ptr(2)
	if Same(mockT, p, *p) {
		t.Error("Same should return false")
	}
	if !Same(mockT, p, p) {
		t.Error("Same should return true")
	}
}

func TestNotSame(t *testing.T) {

	mockT := new(testing.T)

	if !NotSame(mockT, ptr(1), ptr(1)) {
		t.Error("NotSame should return true; different pointers")
	}
	if !NotSame(mockT, 1, 1) {
		t.Error("NotSame should return true; constant inputs")
	}
	p := ptr(2)
	if !NotSame(mockT, p, *p) {
		t.Error("NotSame should return true; mixed-type inputs")
	}
	if NotSame(mockT, p, p) {
		t.Error("NotSame should return false")
	}
}

func Test_samePointers(t *testing.T) {
	p := ptr(2)

	type args struct {
		first  interface{}
		second interface{}
	}
	tests := []struct {
		name      string
		args      args
		assertion BoolAssertionFunc
	}{
		{
			name:      "1 != 2",
			args:      args{first: 1, second: 2},
			assertion: False,
		},
		{
			name:      "1 != 1 (not same ptr)",
			args:      args{first: 1, second: 1},
			assertion: False,
		},
		{
			name:      "ptr(1) == ptr(1)",
			args:      args{first: p, second: p},
			assertion: True,
		},
		{
			name:      "int(1) != float32(1)",
			args:      args{first: int(1), second: float32(1)},
			assertion: False,
		},
		{
			name:      "array != slice",
			args:      args{first: [2]int{1, 2}, second: []int{1, 2}},
			assertion: False,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, samePointers(tt.args.first, tt.args.second))
		})
	}
}

// bufferT implements TestingT. Its implementation of Errorf writes the output that would be produced by
// testing.T.Errorf to an internal bytes.Buffer.
type bufferT struct {
	buf bytes.Buffer
}

func (t *bufferT) Errorf(format string, args ...interface{}) {
	// implementation of decorate is copied from testing.T
	decorate := func(s string) string {
		_, file, line, ok := runtime.Caller(3) // decorate + log + public function.
		if ok {
			// Truncate file name at last file name separator.
			if index := strings.LastIndex(file, "/"); index >= 0 {
				file = file[index+1:]
			} else if index = strings.LastIndex(file, "\\"); index >= 0 {
				file = file[index+1:]
			}
		} else {
			file = "???"
			line = 1
		}
		buf := new(bytes.Buffer)
		// Every line is indented at least one tab.
		buf.WriteByte('\t')
		fmt.Fprintf(buf, "%s:%d: ", file, line)
		lines := strings.Split(s, "\n")
		if l := len(lines); l > 1 && lines[l-1] == "" {
			lines = lines[:l-1]
		}
		for i, line := range lines {
			if i > 0 {
				// Second and subsequent lines are indented an extra tab.
				buf.WriteString("\n\t\t")
			}
			buf.WriteString(line)
		}
		buf.WriteByte('\n')
		return buf.String()
	}
	t.buf.WriteString(decorate(fmt.Sprintf(format, args...)))
}

func TestStringEqual(t *testing.T) {
	for i, currCase := range []struct {
		equalWant  string
		equalGot   string
		msgAndArgs []interface{}
		want       string
	}{
		{equalWant: "hi, \nmy name is", equalGot: "what,\nmy name is", want: "\tassertions.go:\\d+: \n\t+Error Trace:\t\n\t+Error:\\s+Not equal:\\s+\n\\s+expected: \"hi, \\\\nmy name is\"\n\\s+actual\\s+: \"what,\\\\nmy name is\"\n\\s+Diff:\n\\s+-+ Expected\n\\s+\\++ Actual\n\\s+@@ -1,2 \\+1,2 @@\n\\s+-hi, \n\\s+\\+what,\n\\s+my name is"},
	} {
		mockT := &bufferT{}
		Equal(mockT, currCase.equalWant, currCase.equalGot, currCase.msgAndArgs...)
		Regexp(t, regexp.MustCompile(currCase.want), mockT.buf.String(), "Case %d", i)
	}
}

func TestEqualFormatting(t *testing.T) {
	for i, currCase := range []struct {
		equalWant  string
		equalGot   string
		msgAndArgs []interface{}
		want       string
	}{
		{equalWant: "want", equalGot: "got", want: "\tassertions.go:\\d+: \n\t+Error Trace:\t\n\t+Error:\\s+Not equal:\\s+\n\\s+expected: \"want\"\n\\s+actual\\s+: \"got\"\n\\s+Diff:\n\\s+-+ Expected\n\\s+\\++ Actual\n\\s+@@ -1 \\+1 @@\n\\s+-want\n\\s+\\+got\n"},
		{equalWant: "want", equalGot: "got", msgAndArgs: []interface{}{"hello, %v!", "world"}, want: "\tassertions.go:[0-9]+: \n\t+Error Trace:\t\n\t+Error:\\s+Not equal:\\s+\n\\s+expected: \"want\"\n\\s+actual\\s+: \"got\"\n\\s+Diff:\n\\s+-+ Expected\n\\s+\\++ Actual\n\\s+@@ -1 \\+1 @@\n\\s+-want\n\\s+\\+got\n\\s+Messages:\\s+hello, world!\n"},
		{equalWant: "want", equalGot: "got", msgAndArgs: []interface{}{123}, want: "\tassertions.go:[0-9]+: \n\t+Error Trace:\t\n\t+Error:\\s+Not equal:\\s+\n\\s+expected: \"want\"\n\\s+actual\\s+: \"got\"\n\\s+Diff:\n\\s+-+ Expected\n\\s+\\++ Actual\n\\s+@@ -1 \\+1 @@\n\\s+-want\n\\s+\\+got\n\\s+Messages:\\s+123\n"},
		{equalWant: "want", equalGot: "got", msgAndArgs: []interface{}{struct{ a string }{"hello"}}, want: "\tassertions.go:[0-9]+: \n\t+Error Trace:\t\n\t+Error:\\s+Not equal:\\s+\n\\s+expected: \"want\"\n\\s+actual\\s+: \"got\"\n\\s+Diff:\n\\s+-+ Expected\n\\s+\\++ Actual\n\\s+@@ -1 \\+1 @@\n\\s+-want\n\\s+\\+got\n\\s+Messages:\\s+{a:hello}\n"},
	} {
		mockT := &bufferT{}
		Equal(mockT, currCase.equalWant, currCase.equalGot, currCase.msgAndArgs...)
		Regexp(t, regexp.MustCompile(currCase.want), mockT.buf.String(), "Case %d", i)
	}
}

func TestFormatUnequalValues(t *testing.T) {
	expected, actual := formatUnequalValues("foo", "bar")
	Equal(t, `"foo"`, expected, "value should not include type")
	Equal(t, `"bar"`, actual, "value should not include type")

	expected, actual = formatUnequalValues(123, 123)
	Equal(t, `123`, expected, "value should not include type")
	Equal(t, `123`, actual, "value should not include type")

	expected, actual = formatUnequalValues(int64(123), int32(123))
	Equal(t, `int64(123)`, expected, "value should include type")
	Equal(t, `int32(123)`, actual, "value should include type")

	expected, actual = formatUnequalValues(int64(123), nil)
	Equal(t, `int64(123)`, expected, "value should include type")
	Equal(t, `<nil>(<nil>)`, actual, "value should include type")

	type testStructType struct {
		Val string
	}

	expected, actual = formatUnequalValues(&testStructType{Val: "test"}, &testStructType{Val: "test"})
	Equal(t, `&assert.testStructType{Val:"test"}`, expected, "value should not include type annotation")
	Equal(t, `&assert.testStructType{Val:"test"}`, actual, "value should not include type annotation")
}

func TestNotNil(t *testing.T) {

	mockT := new(testing.T)

	if !NotNil(mockT, new(AssertionTesterConformingObject)) {
		t.Error("NotNil should return true: object is not nil")
	}
	if NotNil(mockT, nil) {
		t.Error("NotNil should return false: object is nil")
	}
	if NotNil(mockT, (*struct{})(nil)) {
		t.Error("NotNil should return false: object is (*struct{})(nil)")
	}

}

func TestNil(t *testing.T) {

	mockT := new(testing.T)

	if !Nil(mockT, nil) {
		t.Error("Nil should return true: object is nil")
	}
	if !Nil(mockT, (*struct{})(nil)) {
		t.Error("Nil should return true: object is (*struct{})(nil)")
	}
	if Nil(mockT, new(AssertionTesterConformingObject)) {
		t.Error("Nil should return false: object is not nil")
	}

}

func TestTrue(t *testing.T) {

	mockT := new(testing.T)

	if !True(mockT, true) {
		t.Error("True should return true")
	}
	if True(mockT, false) {
		t.Error("True should return false")
	}

}

func TestFalse(t *testing.T) {

	mockT := new(testing.T)

	if !False(mockT, false) {
		t.Error("False should return true")
	}
	if False(mockT, true) {
		t.Error("False should return false")
	}

}

func TestExactly(t *testing.T) {

	mockT := new(testing.T)

	a := float32(1)
	b := float64(1)
	c := float32(1)
	d := float32(2)

	if Exactly(mockT, a, b) {
		t.Error("Exactly should return false")
	}
	if Exactly(mockT, a, d) {
		t.Error("Exactly should return false")
	}
	if !Exactly(mockT, a, c) {
		t.Error("Exactly should return true")
	}

	if Exactly(mockT, nil, a) {
		t.Error("Exactly should return false")
	}
	if Exactly(mockT, a, nil) {
		t.Error("Exactly should return false")
	}

}

func TestNotEqual(t *testing.T) {

	mockT := new(testing.T)

	if !NotEqual(mockT, "Hello World", "Hello World!") {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, 123, 1234) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, 123.5, 123.55) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, []byte("Hello World"), []byte("Hello World!")) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, nil, new(AssertionTesterConformingObject)) {
		t.Error("NotEqual should return true")
	}
	funcA := func() int { return 23 }
	funcB := func() int { return 42 }
	if NotEqual(mockT, funcA, funcB) {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, nil, nil) {
		t.Error("NotEqual should return false")
	}

	if NotEqual(mockT, "Hello World", "Hello World") {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, 123, 123) {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, 123.5, 123.5) {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, []byte("Hello World"), []byte("Hello World")) {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, new(AssertionTesterConformingObject), new(AssertionTesterConformingObject)) {
		t.Error("NotEqual should return false")
	}
	if NotEqual(mockT, &struct{}{}, &struct{}{}) {
		t.Error("NotEqual should return false")
	}

	// A case that might be confusing, especially with numeric literals
	if !NotEqual(mockT, 10, uint(10)) {
		t.Error("NotEqual should return false")
	}
}

func TestNotEqualValues(t *testing.T) {

	mockT := new(testing.T)

	// Same tests as NotEqual since they behave the same when types are irrelevant
	if !NotEqualValues(mockT, "Hello World", "Hello World!") {
		t.Error("NotEqualValues should return true")
	}
	if !NotEqualValues(mockT, 123, 1234) {
		t.Error("NotEqualValues should return true")
	}
	if !NotEqualValues(mockT, 123.5, 123.55) {
		t.Error("NotEqualValues should return true")
	}
	if !NotEqualValues(mockT, []byte("Hello World"), []byte("Hello World!")) {
		t.Error("NotEqualValues should return true")
	}
	if !NotEqualValues(mockT, nil, new(AssertionTesterConformingObject)) {
		t.Error("NotEqualValues should return true")
	}
	if NotEqualValues(mockT, nil, nil) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, "Hello World", "Hello World") {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, 123, 123) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, 123.5, 123.5) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, []byte("Hello World"), []byte("Hello World")) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, new(AssertionTesterConformingObject), new(AssertionTesterConformingObject)) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, &struct{}{}, &struct{}{}) {
		t.Error("NotEqualValues should return false")
	}

	// Special cases where NotEqualValues behaves differently
	funcA := func() int { return 23 }
	funcB := func() int { return 42 }
	if !NotEqualValues(mockT, funcA, funcB) {
		t.Error("NotEqualValues should return true")
	}
	if !NotEqualValues(mockT, int(10), int(11)) {
		t.Error("NotEqualValues should return true")
	}
	if NotEqualValues(mockT, int(10), uint(10)) {
		t.Error("NotEqualValues should return false")
	}
	if NotEqualValues(mockT, struct{}{}, struct{}{}) {
		t.Error("NotEqualValues should return false")
	}
}

type A struct {
	Name, Value string
}

func TestContains(t *testing.T) {

	mockT := new(testing.T)
	list := []string{"Foo", "Bar"}
	complexList := []*A{
		{"b", "c"},
		{"d", "e"},
		{"g", "h"},
		{"j", "k"},
	}
	simpleMap := map[interface{}]interface{}{"Foo": "Bar"}

	if !Contains(mockT, "Hello World", "Hello") {
		t.Error("Contains should return true: \"Hello World\" contains \"Hello\"")
	}
	if Contains(mockT, "Hello World", "Salut") {
		t.Error("Contains should return false: \"Hello World\" does not contain \"Salut\"")
	}

	if !Contains(mockT, list, "Bar") {
		t.Error("Contains should return true: \"[\"Foo\", \"Bar\"]\" contains \"Bar\"")
	}
	if Contains(mockT, list, "Salut") {
		t.Error("Contains should return false: \"[\"Foo\", \"Bar\"]\" does not contain \"Salut\"")
	}
	if !Contains(mockT, complexList, &A{"g", "h"}) {
		t.Error("Contains should return true: complexList contains {\"g\", \"h\"}")
	}
	if Contains(mockT, complexList, &A{"g", "e"}) {
		t.Error("Contains should return false: complexList contains {\"g\", \"e\"}")
	}
	if Contains(mockT, complexList, &A{"g", "e"}) {
		t.Error("Contains should return false: complexList contains {\"g\", \"e\"}")
	}
	if !Contains(mockT, simpleMap, "Foo") {
		t.Error("Contains should return true: \"{\"Foo\": \"Bar\"}\" contains \"Foo\"")
	}
	if Contains(mockT, simpleMap, "Bar") {
		t.Error("Contains should return false: \"{\"Foo\": \"Bar\"}\" does not contains \"Bar\"")
	}
}

func TestContainsFailMessage(t *testing.T) {

	mockT := new(mockTestingT)

	Contains(mockT, "Hello World", errors.New("Hello"))
	expectedFail := "\"Hello World\" does not contain &errors.errorString{s:\"Hello\"}"
	actualFail := mockT.errorString()
	if !strings.Contains(actualFail, expectedFail) {
		t.Errorf("Contains failure should include %q but was %q", expectedFail, actualFail)
	}
}

func TestNotContains(t *testing.T) {

	mockT := new(testing.T)
	list := []string{"Foo", "Bar"}
	simpleMap := map[interface{}]interface{}{"Foo": "Bar"}

	if !NotContains(mockT, "Hello World", "Hello!") {
		t.Error("NotContains should return true: \"Hello World\" does not contain \"Hello!\"")
	}
	if NotContains(mockT, "Hello World", "Hello") {
		t.Error("NotContains should return false: \"Hello World\" contains \"Hello\"")
	}

	if !NotContains(mockT, list, "Foo!") {
		t.Error("NotContains should return true: \"[\"Foo\", \"Bar\"]\" does not contain \"Foo!\"")
	}
	if NotContains(mockT, list, "Foo") {
		t.Error("NotContains should return false: \"[\"Foo\", \"Bar\"]\" contains \"Foo\"")
	}
	if NotContains(mockT, simpleMap, "Foo") {
		t.Error("Contains should return true: \"{\"Foo\": \"Bar\"}\" contains \"Foo\"")
	}
	if !NotContains(mockT, simpleMap, "Bar") {
		t.Error("Contains should return false: \"{\"Foo\": \"Bar\"}\" does not contains \"Bar\"")
	}
}

func TestSubset(t *testing.T) {
	mockT := new(testing.T)

	if !Subset(mockT, []int{1, 2, 3}, nil) {
		t.Error("Subset should return true: given subset is nil")
	}
	if !Subset(mockT, []int{1, 2, 3}, []int{}) {
		t.Error("Subset should return true: any set contains the nil set")
	}
	if !Subset(mockT, []int{1, 2, 3}, []int{1, 2}) {
		t.Error("Subset should return true: [1, 2, 3] contains [1, 2]")
	}
	if !Subset(mockT, []int{1, 2, 3}, []int{1, 2, 3}) {
		t.Error("Subset should return true: [1, 2, 3] contains [1, 2, 3]")
	}
	if !Subset(mockT, []string{"hello", "world"}, []string{"hello"}) {
		t.Error("Subset should return true: [\"hello\", \"world\"] contains [\"hello\"]")
	}

	if Subset(mockT, []string{"hello", "world"}, []string{"hello", "testify"}) {
		t.Error("Subset should return false: [\"hello\", \"world\"] does not contain [\"hello\", \"testify\"]")
	}
	if Subset(mockT, []int{1, 2, 3}, []int{4, 5}) {
		t.Error("Subset should return false: [1, 2, 3] does not contain [4, 5]")
	}
	if Subset(mockT, []int{1, 2, 3}, []int{1, 5}) {
		t.Error("Subset should return false: [1, 2, 3] does not contain [1, 5]")
	}
}

func TestNotSubset(t *testing.T) {
	mockT := new(testing.T)

	if NotSubset(mockT, []int{1, 2, 3}, nil) {
		t.Error("NotSubset should return false: given subset is nil")
	}
	if NotSubset(mockT, []int{1, 2, 3}, []int{}) {
		t.Error("NotSubset should return false: any set contains the nil set")
	}
	if NotSubset(mockT, []int{1, 2, 3}, []int{1, 2}) {
		t.Error("NotSubset should return false: [1, 2, 3] contains [1, 2]")
	}
	if NotSubset(mockT, []int{1, 2, 3}, []int{1, 2, 3}) {
		t.Error("NotSubset should return false: [1, 2, 3] contains [1, 2, 3]")
	}
	if NotSubset(mockT, []string{"hello", "world"}, []string{"hello"}) {
		t.Error("NotSubset should return false: [\"hello\", \"world\"] contains [\"hello\"]")
	}

	if !NotSubset(mockT, []string{"hello", "world"}, []string{"hello", "testify"}) {
		t.Error("NotSubset should return true: [\"hello\", \"world\"] does not contain [\"hello\", \"testify\"]")
	}
	if !NotSubset(mockT, []int{1, 2, 3}, []int{4, 5}) {
		t.Error("NotSubset should return true: [1, 2, 3] does not contain [4, 5]")
	}
	if !NotSubset(mockT, []int{1, 2, 3}, []int{1, 5}) {
		t.Error("NotSubset should return true: [1, 2, 3] does not contain [1, 5]")
	}
}

func TestNotSubsetNil(t *testing.T) {
	mockT := new(testing.T)
	NotSubset(mockT, []string{"foo"}, nil)
	if !mockT.Failed() {
		t.Error("NotSubset on nil set should have failed the test")
	}
}

func Test_includeElement(t *testing.T) {

	list1 := []string{"Foo", "Bar"}
	list2 := []int{1, 2}
	simpleMap := map[interface{}]interface{}{"Foo": "Bar"}

	ok, found := includeElement("Hello World", "World")
	True(t, ok)
	True(t, found)

	ok, found = includeElement(list1, "Foo")
	True(t, ok)
	True(t, found)

	ok, found = includeElement(list1, "Bar")
	True(t, ok)
	True(t, found)

	ok, found = includeElement(list2, 1)
	True(t, ok)
	True(t, found)

	ok, found = includeElement(list2, 2)
	True(t, ok)
	True(t, found)

	ok, found = includeElement(list1, "Foo!")
	True(t, ok)
	False(t, found)

	ok, found = includeElement(list2, 3)
	True(t, ok)
	False(t, found)

	ok, found = includeElement(list2, "1")
	True(t, ok)
	False(t, found)

	ok, found = includeElement(simpleMap, "Foo")
	True(t, ok)
	True(t, found)

	ok, found = includeElement(simpleMap, "Bar")
	True(t, ok)
	False(t, found)

	ok, found = includeElement(1433, "1")
	False(t, ok)
	False(t, found)
}

func TestElementsMatch(t *testing.T) {
	mockT := new(testing.T)

	if !ElementsMatch(mockT, nil, nil) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{}, []int{}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{1}, []int{1}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{1, 1}, []int{1, 1}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{1, 2}, []int{1, 2}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{1, 2}, []int{2, 1}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, [2]int{1, 2}, [2]int{2, 1}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []string{"hello", "world"}, []string{"world", "hello"}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []string{"hello", "hello"}, []string{"hello", "hello"}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []string{"hello", "hello", "world"}, []string{"hello", "world", "hello"}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, [3]string{"hello", "hello", "world"}, [3]string{"hello", "world", "hello"}) {
		t.Error("ElementsMatch should return true")
	}
	if !ElementsMatch(mockT, []int{}, nil) {
		t.Error("ElementsMatch should return true")
	}

	if ElementsMatch(mockT, []int{1}, []int{1, 1}) {
		t.Error("ElementsMatch should return false")
	}
	if ElementsMatch(mockT, []int{1, 2}, []int{2, 2}) {
		t.Error("ElementsMatch should return false")
	}
	if ElementsMatch(mockT, []string{"hello", "hello"}, []string{"hello"}) {
		t.Error("ElementsMatch should return false")
	}
}

func TestDiffLists(t *testing.T) {
	tests := []struct {
		name   string
		listA  interface{}
		listB  interface{}
		extraA []interface{}
		extraB []interface{}
	}{
		{
			name:   "equal empty",
			listA:  []string{},
			listB:  []string{},
			extraA: nil,
			extraB: nil,
		},
		{
			name:   "equal same order",
			listA:  []string{"hello", "world"},
			listB:  []string{"hello", "world"},
			extraA: nil,
			extraB: nil,
		},
		{
			name:   "equal different order",
			listA:  []string{"hello", "world"},
			listB:  []string{"world", "hello"},
			extraA: nil,
			extraB: nil,
		},
		{
			name:   "extra A",
			listA:  []string{"hello", "hello", "world"},
			listB:  []string{"hello", "world"},
			extraA: []interface{}{"hello"},
			extraB: nil,
		},
		{
			name:   "extra A twice",
			listA:  []string{"hello", "hello", "hello", "world"},
			listB:  []string{"hello", "world"},
			extraA: []interface{}{"hello", "hello"},
			extraB: nil,
		},
		{
			name:   "extra B",
			listA:  []string{"hello", "world"},
			listB:  []string{"hello", "hello", "world"},
			extraA: nil,
			extraB: []interface{}{"hello"},
		},
		{
			name:   "extra B twice",
			listA:  []string{"hello", "world"},
			listB:  []string{"hello", "hello", "world", "hello"},
			extraA: nil,
			extraB: []interface{}{"hello", "hello"},
		},
		{
			name:   "integers 1",
			listA:  []int{1, 2, 3, 4, 5},
			listB:  []int{5, 4, 3, 2, 1},
			extraA: nil,
			extraB: nil,
		},
		{
			name:   "integers 2",
			listA:  []int{1, 2, 1, 2, 1},
			listB:  []int{2, 1, 2, 1, 2},
			extraA: []interface{}{1},
			extraB: []interface{}{2},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			actualExtraA, actualExtraB := diffLists(test.listA, test.listB)
			Equal(t, test.extraA, actualExtraA, "extra A does not match for listA=%v listB=%v",
				test.listA, test.listB)
			Equal(t, test.extraB, actualExtraB, "extra B does not match for listA=%v listB=%v",
				test.listA, test.listB)
		})
	}
}

func TestCondition(t *testing.T) {
	mockT := new(testing.T)

	if !Condition(mockT, func() bool { return true }, "Truth") {
		t.Error("Condition should return true")
	}

	if Condition(mockT, func() bool { return false }, "Lie") {
		t.Error("Condition should return false")
	}

}

func TestDidPanic(t *testing.T) {

	if funcDidPanic, _, _ := didPanic(func() {
		panic("Panic!")
	}); !funcDidPanic {
		t.Error("didPanic should return true")
	}

	if funcDidPanic, _, _ := didPanic(func() {
	}); funcDidPanic {
		t.Error("didPanic should return false")
	}

}

func TestPanics(t *testing.T) {

	mockT := new(testing.T)

	if !Panics(mockT, func() {
		panic("Panic!")
	}) {
		t.Error("Panics should return true")
	}

	if Panics(mockT, func() {
	}) {
		t.Error("Panics should return false")
	}

}

func TestPanicsWithValue(t *testing.T) {

	mockT := new(testing.T)

	if !PanicsWithValue(mockT, "Panic!", func() {
		panic("Panic!")
	}) {
		t.Error("PanicsWithValue should return true")
	}

	if PanicsWithValue(mockT, "Panic!", func() {
	}) {
		t.Error("PanicsWithValue should return false")
	}

	if PanicsWithValue(mockT, "at the disco", func() {
		panic("Panic!")
	}) {
		t.Error("PanicsWithValue should return false")
	}
}

func TestPanicsWithError(t *testing.T) {

	mockT := new(testing.T)

	if !PanicsWithError(mockT, "panic", func() {
		panic(errors.New("panic"))
	}) {
		t.Error("PanicsWithError should return true")
	}

	if PanicsWithError(mockT, "Panic!", func() {
	}) {
		t.Error("PanicsWithError should return false")
	}

	if PanicsWithError(mockT, "at the disco", func() {
		panic(errors.New("panic"))
	}) {
		t.Error("PanicsWithError should return false")
	}

	if PanicsWithError(mockT, "Panic!", func() {
		panic("panic")
	}) {
		t.Error("PanicsWithError should return false")
	}
}

func TestNotPanics(t *testing.T) {

	mockT := new(testing.T)

	if !NotPanics(mockT, func() {
	}) {
		t.Error("NotPanics should return true")
	}

	if NotPanics(mockT, func() {
		panic("Panic!")
	}) {
		t.Error("NotPanics should return false")
	}

}

func TestNoError(t *testing.T) {

	mockT := new(testing.T)

	// start with a nil error
	var err error

	True(t, NoError(mockT, err), "NoError should return True for nil arg")

	// now set an error
	err = errors.New("some error")

	False(t, NoError(mockT, err), "NoError with error should return False")

	// returning an empty error interface
	err = func() error {
		var err *customError
		return err
	}()

	if err == nil { // err is not nil here!
		t.Errorf("Error should be nil due to empty interface: %s", err)
	}

	False(t, NoError(mockT, err), "NoError should fail with empty error interface")
}

type customError struct{}

func (*customError) Error() string { return "fail" }

func TestError(t *testing.T) {

	mockT := new(testing.T)

	// start with a nil error
	var err error

	False(t, Error(mockT, err), "Error should return False for nil arg")

	// now set an error
	err = errors.New("some error")

	True(t, Error(mockT, err), "Error with error should return True")

	// go vet check
	True(t, Errorf(mockT, err, "example with %s", "formatted message"), "Errorf with error should rturn True")

	// returning an empty error interface
	err = func() error {
		var err *customError
		return err
	}()

	if err == nil { // err is not nil here!
		t.Errorf("Error should be nil due to empty interface: %s", err)
	}

	True(t, Error(mockT, err), "Error should pass with empty error interface")
}

func TestEqualError(t *testing.T) {
	mockT := new(testing.T)

	// start with a nil error
	var err error
	False(t, EqualError(mockT, err, ""),
		"EqualError should return false for nil arg")

	// now set an error
	err = errors.New("some error")
	False(t, EqualError(mockT, err, "Not some error"),
		"EqualError should return false for different error string")
	True(t, EqualError(mockT, err, "some error"),
		"EqualError should return true")
}

func Test_isEmpty(t *testing.T) {

	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}

	True(t, isEmpty(""))
	True(t, isEmpty(nil))
	True(t, isEmpty([]string{}))
	True(t, isEmpty(0))
	True(t, isEmpty(int32(0)))
	True(t, isEmpty(int64(0)))
	True(t, isEmpty(false))
	True(t, isEmpty(map[string]string{}))
	True(t, isEmpty(new(time.Time)))
	True(t, isEmpty(time.Time{}))
	True(t, isEmpty(make(chan struct{})))
	False(t, isEmpty("something"))
	False(t, isEmpty(errors.New("something")))
	False(t, isEmpty([]string{"something"}))
	False(t, isEmpty(1))
	False(t, isEmpty(true))
	False(t, isEmpty(map[string]string{"Hello": "World"}))
	False(t, isEmpty(chWithValue))

}

func TestEmpty(t *testing.T) {

	mockT := new(testing.T)
	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}
	var tiP *time.Time
	var tiNP time.Time
	var s *string
	var f *os.File
	sP := &s
	x := 1
	xP := &x

	type TString string
	type TStruct struct {
		x int
	}

	True(t, Empty(mockT, ""), "Empty string is empty")
	True(t, Empty(mockT, nil), "Nil is empty")
	True(t, Empty(mockT, []string{}), "Empty string array is empty")
	True(t, Empty(mockT, 0), "Zero int value is empty")
	True(t, Empty(mockT, false), "False value is empty")
	True(t, Empty(mockT, make(chan struct{})), "Channel without values is empty")
	True(t, Empty(mockT, s), "Nil string pointer is empty")
	True(t, Empty(mockT, f), "Nil os.File pointer is empty")
	True(t, Empty(mockT, tiP), "Nil time.Time pointer is empty")
	True(t, Empty(mockT, tiNP), "time.Time is empty")
	True(t, Empty(mockT, TStruct{}), "struct with zero values is empty")
	True(t, Empty(mockT, TString("")), "empty aliased string is empty")
	True(t, Empty(mockT, sP), "ptr to nil value is empty")

	False(t, Empty(mockT, "something"), "Non Empty string is not empty")
	False(t, Empty(mockT, errors.New("something")), "Non nil object is not empty")
	False(t, Empty(mockT, []string{"something"}), "Non empty string array is not empty")
	False(t, Empty(mockT, 1), "Non-zero int value is not empty")
	False(t, Empty(mockT, true), "True value is not empty")
	False(t, Empty(mockT, chWithValue), "Channel with values is not empty")
	False(t, Empty(mockT, TStruct{x: 1}), "struct with initialized values is empty")
	False(t, Empty(mockT, TString("abc")), "non-empty aliased string is empty")
	False(t, Empty(mockT, xP), "ptr to non-nil value is not empty")
}

func TestNotEmpty(t *testing.T) {

	mockT := new(testing.T)
	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}

	False(t, NotEmpty(mockT, ""), "Empty string is empty")
	False(t, NotEmpty(mockT, nil), "Nil is empty")
	False(t, NotEmpty(mockT, []string{}), "Empty string array is empty")
	False(t, NotEmpty(mockT, 0), "Zero int value is empty")
	False(t, NotEmpty(mockT, false), "False value is empty")
	False(t, NotEmpty(mockT, make(chan struct{})), "Channel without values is empty")

	True(t, NotEmpty(mockT, "something"), "Non Empty string is not empty")
	True(t, NotEmpty(mockT, errors.New("something")), "Non nil object is not empty")
	True(t, NotEmpty(mockT, []string{"something"}), "Non empty string array is not empty")
	True(t, NotEmpty(mockT, 1), "Non-zero int value is not empty")
	True(t, NotEmpty(mockT, true), "True value is not empty")
	True(t, NotEmpty(mockT, chWithValue), "Channel with values is not empty")
}

func Test_getLen(t *testing.T) {
	falseCases := []interface{}{
		nil,
		0,
		true,
		false,
		'A',
		struct{}{},
	}
	for _, v := range falseCases {
		ok, l := getLen(v)
		False(t, ok, "Expected getLen fail to get length of %#v", v)
		Equal(t, 0, l, "getLen should return 0 for %#v", v)
	}

	ch := make(chan int, 5)
	ch <- 1
	ch <- 2
	ch <- 3
	trueCases := []struct {
		v interface{}
		l int
	}{
		{[]int{1, 2, 3}, 3},
		{[...]int{1, 2, 3}, 3},
		{"ABC", 3},
		{map[int]int{1: 2, 2: 4, 3: 6}, 3},
		{ch, 3},

		{[]int{}, 0},
		{map[int]int{}, 0},
		{make(chan int), 0},

		{[]int(nil), 0},
		{map[int]int(nil), 0},
		{(chan int)(nil), 0},
	}

	for _, c := range trueCases {
		ok, l := getLen(c.v)
		True(t, ok, "Expected getLen success to get length of %#v", c.v)
		Equal(t, c.l, l)
	}
}

func TestLen(t *testing.T) {
	mockT := new(testing.T)

	False(t, Len(mockT, nil, 0), "nil does not have length")
	False(t, Len(mockT, 0, 0), "int does not have length")
	False(t, Len(mockT, true, 0), "true does not have length")
	False(t, Len(mockT, false, 0), "false does not have length")
	False(t, Len(mockT, 'A', 0), "Rune does not have length")
	False(t, Len(mockT, struct{}{}, 0), "Struct does not have length")

	ch := make(chan int, 5)
	ch <- 1
	ch <- 2
	ch <- 3

	cases := []struct {
		v interface{}
		l int
	}{
		{[]int{1, 2, 3}, 3},
		{[...]int{1, 2, 3}, 3},
		{"ABC", 3},
		{map[int]int{1: 2, 2: 4, 3: 6}, 3},
		{ch, 3},

		{[]int{}, 0},
		{map[int]int{}, 0},
		{make(chan int), 0},

		{[]int(nil), 0},
		{map[int]int(nil), 0},
		{(chan int)(nil), 0},
	}

	for _, c := range cases {
		True(t, Len(mockT, c.v, c.l), "%#v have %d items", c.v, c.l)
	}

	cases = []struct {
		v interface{}
		l int
	}{
		{[]int{1, 2, 3}, 4},
		{[...]int{1, 2, 3}, 2},
		{"ABC", 2},
		{map[int]int{1: 2, 2: 4, 3: 6}, 4},
		{ch, 2},

		{[]int{}, 1},
		{map[int]int{}, 1},
		{make(chan int), 1},

		{[]int(nil), 1},
		{map[int]int(nil), 1},
		{(chan int)(nil), 1},
	}

	for _, c := range cases {
		False(t, Len(mockT, c.v, c.l), "%#v have %d items", c.v, c.l)
	}
}

func TestWithinDuration(t *testing.T) {

	mockT := new(testing.T)
	a := time.Now()
	b := a.Add(10 * time.Second)

	True(t, WithinDuration(mockT, a, b, 10*time.Second), "A 10s difference is within a 10s time difference")
	True(t, WithinDuration(mockT, b, a, 10*time.Second), "A 10s difference is within a 10s time difference")

	False(t, WithinDuration(mockT, a, b, 9*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, 9*time.Second), "A 10s difference is not within a 9s time difference")

	False(t, WithinDuration(mockT, a, b, -9*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, -9*time.Second), "A 10s difference is not within a 9s time difference")

	False(t, WithinDuration(mockT, a, b, -11*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, -11*time.Second), "A 10s difference is not within a 9s time difference")
}

func TestInDelta(t *testing.T) {
	mockT := new(testing.T)

	True(t, InDelta(mockT, 1.001, 1, 0.01), "|1.001 - 1| <= 0.01")
	True(t, InDelta(mockT, 1, 1.001, 0.01), "|1 - 1.001| <= 0.01")
	True(t, InDelta(mockT, 1, 2, 1), "|1 - 2| <= 1")
	False(t, InDelta(mockT, 1, 2, 0.5), "Expected |1 - 2| <= 0.5 to fail")
	False(t, InDelta(mockT, 2, 1, 0.5), "Expected |2 - 1| <= 0.5 to fail")
	False(t, InDelta(mockT, "", nil, 1), "Expected non numerals to fail")
	False(t, InDelta(mockT, 42, math.NaN(), 0.01), "Expected NaN for actual to fail")
	False(t, InDelta(mockT, math.NaN(), 42, 0.01), "Expected NaN for expected to fail")

	cases := []struct {
		a, b  interface{}
		delta float64
	}{
		{uint(2), uint(1), 1},
		{uint8(2), uint8(1), 1},
		{uint16(2), uint16(1), 1},
		{uint32(2), uint32(1), 1},
		{uint64(2), uint64(1), 1},

		{int(2), int(1), 1},
		{int8(2), int8(1), 1},
		{int16(2), int16(1), 1},
		{int32(2), int32(1), 1},
		{int64(2), int64(1), 1},

		{float32(2), float32(1), 1},
		{float64(2), float64(1), 1},
	}

	for _, tc := range cases {
		True(t, InDelta(mockT, tc.a, tc.b, tc.delta), "Expected |%V - %V| <= %v", tc.a, tc.b, tc.delta)
	}
}

func TestInDeltaSlice(t *testing.T) {
	mockT := new(testing.T)

	True(t, InDeltaSlice(mockT,
		[]float64{1.001, 0.999},
		[]float64{1, 1},
		0.1), "{1.001, 0.009} is element-wise close to {1, 1} in delta=0.1")

	True(t, InDeltaSlice(mockT,
		[]float64{1, 2},
		[]float64{0, 3},
		1), "{1, 2} is element-wise close to {0, 3} in delta=1")

	False(t, InDeltaSlice(mockT,
		[]float64{1, 2},
		[]float64{0, 3},
		0.1), "{1, 2} is not element-wise close to {0, 3} in delta=0.1")

	False(t, InDeltaSlice(mockT, "", nil, 1), "Expected non numeral slices to fail")
}

func TestInDeltaMapValues(t *testing.T) {
	mockT := new(testing.T)

	for _, tc := range []struct {
		title  string
		expect interface{}
		actual interface{}
		f      func(TestingT, bool, ...interface{}) bool
		delta  float64
	}{
		{
			title: "Within delta",
			expect: map[string]float64{
				"foo": 1.0,
				"bar": 2.0,
			},
			actual: map[string]float64{
				"foo": 1.01,
				"bar": 1.99,
			},
			delta: 0.1,
			f:     True,
		},
		{
			title: "Within delta",
			expect: map[int]float64{
				1: 1.0,
				2: 2.0,
			},
			actual: map[int]float64{
				1: 1.0,
				2: 1.99,
			},
			delta: 0.1,
			f:     True,
		},
		{
			title: "Different number of keys",
			expect: map[int]float64{
				1: 1.0,
				2: 2.0,
			},
			actual: map[int]float64{
				1: 1.0,
			},
			delta: 0.1,
			f:     False,
		},
		{
			title: "Within delta with zero value",
			expect: map[string]float64{
				"zero": 0.0,
			},
			actual: map[string]float64{
				"zero": 0.0,
			},
			delta: 0.1,
			f:     True,
		},
		{
			title: "With missing key with zero value",
			expect: map[string]float64{
				"zero": 0.0,
				"foo":  0.0,
			},
			actual: map[string]float64{
				"zero": 0.0,
				"bar":  0.0,
			},
			f: False,
		},
	} {
		tc.f(t, InDeltaMapValues(mockT, tc.expect, tc.actual, tc.delta), tc.title+"\n"+diff(tc.expect, tc.actual))
	}
}

func TestInEpsilon(t *testing.T) {
	mockT := new(testing.T)

	cases := []struct {
		a, b    interface{}
		epsilon float64
	}{
		{uint8(2), uint16(2), .001},
		{2.1, 2.2, 0.1},
		{2.2, 2.1, 0.1},
		{-2.1, -2.2, 0.1},
		{-2.2, -2.1, 0.1},
		{uint64(100), uint8(101), 0.01},
		{0.1, -0.1, 2},
		{0.1, 0, 2},
		{time.Second, time.Second + time.Millisecond, 0.002},
	}

	for _, tc := range cases {
		True(t, InEpsilon(t, tc.a, tc.b, tc.epsilon, "Expected %V and %V to have a relative difference of %v", tc.a, tc.b, tc.epsilon), "test: %q", tc)
	}

	cases = []struct {
		a, b    interface{}
		epsilon float64
	}{
		{uint8(2), int16(-2), .001},
		{uint64(100), uint8(102), 0.01},
		{2.1, 2.2, 0.001},
		{2.2, 2.1, 0.001},
		{2.1, -2.2, 1},
		{2.1, "bla-bla", 0},
		{0.1, -0.1, 1.99},
		{0, 0.1, 2}, // expected must be different to zero
		{time.Second, time.Second + 10*time.Millisecond, 0.002},
		{math.NaN(), 0, 1},
		{0, math.NaN(), 1},
		{0, 0, math.NaN()},
	}

	for _, tc := range cases {
		False(t, InEpsilon(mockT, tc.a, tc.b, tc.epsilon, "Expected %V and %V to have a relative difference of %v", tc.a, tc.b, tc.epsilon))
	}

}

func TestInEpsilonSlice(t *testing.T) {
	mockT := new(testing.T)

	True(t, InEpsilonSlice(mockT,
		[]float64{2.2, 2.0},
		[]float64{2.1, 2.1},
		0.06), "{2.2, 2.0} is element-wise close to {2.1, 2.1} in espilon=0.06")

	False(t, InEpsilonSlice(mockT,
		[]float64{2.2, 2.0},
		[]float64{2.1, 2.1},
		0.04), "{2.2, 2.0} is not element-wise close to {2.1, 2.1} in espilon=0.04")

	False(t, InEpsilonSlice(mockT, "", nil, 1), "Expected non numeral slices to fail")
}

func TestRegexp(t *testing.T) {
	mockT := new(testing.T)

	cases := []struct {
		rx, str string
	}{
		{"^start", "start of the line"},
		{"end$", "in the end"},
		{"[0-9]{3}[.-]?[0-9]{2}[.-]?[0-9]{2}", "My phone number is 650.12.34"},
	}

	for _, tc := range cases {
		True(t, Regexp(mockT, tc.rx, tc.str))
		True(t, Regexp(mockT, regexp.MustCompile(tc.rx), tc.str))
		False(t, NotRegexp(mockT, tc.rx, tc.str))
		False(t, NotRegexp(mockT, regexp.MustCompile(tc.rx), tc.str))
	}

	cases = []struct {
		rx, str string
	}{
		{"^asdfastart", "Not the start of the line"},
		{"end$", "in the end."},
		{"[0-9]{3}[.-]?[0-9]{2}[.-]?[0-9]{2}", "My phone number is 650.12a.34"},
	}

	for _, tc := range cases {
		False(t, Regexp(mockT, tc.rx, tc.str), "Expected \"%s\" to not match \"%s\"", tc.rx, tc.str)
		False(t, Regexp(mockT, regexp.MustCompile(tc.rx), tc.str))
		True(t, NotRegexp(mockT, tc.rx, tc.str))
		True(t, NotRegexp(mockT, regexp.MustCompile(tc.rx), tc.str))
	}
}

func testAutogeneratedFunction() {
	defer func() {
		if err := recover(); err == nil {
			panic("did not panic")
		}
		CallerInfo()
	}()
	t := struct {
		io.Closer
	}{}
	var c io.Closer
	c = t
	c.Close()
}

func TestCallerInfoWithAutogeneratedFunctions(t *testing.T) {
	NotPanics(t, func() {
		testAutogeneratedFunction()
	})
}

func TestZero(t *testing.T) {
	mockT := new(testing.T)

	for _, test := range zeros {
		True(t, Zero(mockT, test, "%#v is not the %v zero value", test, reflect.TypeOf(test)))
	}

	for _, test := range nonZeros {
		False(t, Zero(mockT, test, "%#v is not the %v zero value", test, reflect.TypeOf(test)))
	}
}

func TestNotZero(t *testing.T) {
	mockT := new(testing.T)

	for _, test := range zeros {
		False(t, NotZero(mockT, test, "%#v is not the %v zero value", test, reflect.TypeOf(test)))
	}

	for _, test := range nonZeros {
		True(t, NotZero(mockT, test, "%#v is not the %v zero value", test, reflect.TypeOf(test)))
	}
}

func TestFileExists(t *testing.T) {
	mockT := new(testing.T)
	True(t, FileExists(mockT, "assertions.go"))

	mockT = new(testing.T)
	False(t, FileExists(mockT, "random_file"))

	mockT = new(testing.T)
	False(t, FileExists(mockT, "../_codegen"))

	var tempFiles []string

	link, err := getTempSymlinkPath("assertions.go")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	True(t, FileExists(mockT, link))

	link, err = getTempSymlinkPath("non_existent_file")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	True(t, FileExists(mockT, link))

	errs := cleanUpTempFiles(tempFiles)
	if len(errs) > 0 {
		t.Fatal("could not clean up temporary files")
	}
}

func TestNoFileExists(t *testing.T) {
	mockT := new(testing.T)
	False(t, NoFileExists(mockT, "assertions.go"))

	mockT = new(testing.T)
	True(t, NoFileExists(mockT, "non_existent_file"))

	mockT = new(testing.T)
	True(t, NoFileExists(mockT, "../_codegen"))

	var tempFiles []string

	link, err := getTempSymlinkPath("assertions.go")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	False(t, NoFileExists(mockT, link))

	link, err = getTempSymlinkPath("non_existent_file")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	False(t, NoFileExists(mockT, link))

	errs := cleanUpTempFiles(tempFiles)
	if len(errs) > 0 {
		t.Fatal("could not clean up temporary files")
	}
}

func getTempSymlinkPath(file string) (string, error) {
	link := file + "_symlink"
	err := os.Symlink(file, link)
	return link, err
}

func cleanUpTempFiles(paths []string) []error {
	var res []error
	for _, path := range paths {
		err := os.Remove(path)
		if err != nil {
			res = append(res, err)
		}
	}
	return res
}

func TestDirExists(t *testing.T) {
	mockT := new(testing.T)
	False(t, DirExists(mockT, "assertions.go"))

	mockT = new(testing.T)
	False(t, DirExists(mockT, "non_existent_dir"))

	mockT = new(testing.T)
	True(t, DirExists(mockT, "../_codegen"))

	var tempFiles []string

	link, err := getTempSymlinkPath("assertions.go")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	False(t, DirExists(mockT, link))

	link, err = getTempSymlinkPath("non_existent_dir")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	False(t, DirExists(mockT, link))

	errs := cleanUpTempFiles(tempFiles)
	if len(errs) > 0 {
		t.Fatal("could not clean up temporary files")
	}
}

func TestNoDirExists(t *testing.T) {
	mockT := new(testing.T)
	True(t, NoDirExists(mockT, "assertions.go"))

	mockT = new(testing.T)
	True(t, NoDirExists(mockT, "non_existent_dir"))

	mockT = new(testing.T)
	False(t, NoDirExists(mockT, "../_codegen"))

	var tempFiles []string

	link, err := getTempSymlinkPath("assertions.go")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	True(t, NoDirExists(mockT, link))

	link, err = getTempSymlinkPath("non_existent_dir")
	if err != nil {
		t.Fatal("could not create temp symlink, err:", err)
	}
	tempFiles = append(tempFiles, link)
	mockT = new(testing.T)
	True(t, NoDirExists(mockT, link))

	errs := cleanUpTempFiles(tempFiles)
	if len(errs) > 0 {
		t.Fatal("could not clean up temporary files")
	}
}

func TestJSONEq_EqualSONString(t *testing.T) {
	mockT := new(testing.T)
	True(t, JSONEq(mockT, `{"hello": "world", "foo": "bar"}`, `{"hello": "world", "foo": "bar"}`))
}

func TestJSONEq_EquivalentButNotEqual(t *testing.T) {
	mockT := new(testing.T)
	True(t, JSONEq(mockT, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`))
}

func TestJSONEq_HashOfArraysAndHashes(t *testing.T) {
	mockT := new(testing.T)
	True(t, JSONEq(mockT, "{\r\n\t\"numeric\": 1.5,\r\n\t\"array\": [{\"foo\": \"bar\"}, 1, \"string\", [\"nested\", \"array\", 5.5]],\r\n\t\"hash\": {\"nested\": \"hash\", \"nested_slice\": [\"this\", \"is\", \"nested\"]},\r\n\t\"string\": \"foo\"\r\n}",
		"{\r\n\t\"numeric\": 1.5,\r\n\t\"hash\": {\"nested\": \"hash\", \"nested_slice\": [\"this\", \"is\", \"nested\"]},\r\n\t\"string\": \"foo\",\r\n\t\"array\": [{\"foo\": \"bar\"}, 1, \"string\", [\"nested\", \"array\", 5.5]]\r\n}"))
}

func TestJSONEq_Array(t *testing.T) {
	mockT := new(testing.T)
	True(t, JSONEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `["foo", {"nested": "hash", "hello": "world"}]`))
}

func TestJSONEq_HashAndArrayNotEquivalent(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `{"foo": "bar", {"nested": "hash", "hello": "world"}}`))
}

func TestJSONEq_HashesNotEquivalent(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, `{"foo": "bar"}`, `{"foo": "bar", "hello": "world"}`))
}

func TestJSONEq_ActualIsNotJSON(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, `{"foo": "bar"}`, "Not JSON"))
}

func TestJSONEq_ExpectedIsNotJSON(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, "Not JSON", `{"foo": "bar", "hello": "world"}`))
}

func TestJSONEq_ExpectedAndActualNotJSON(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, "Not JSON", "Not JSON"))
}

func TestJSONEq_ArraysOfDifferentOrder(t *testing.T) {
	mockT := new(testing.T)
	False(t, JSONEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `[{ "hello": "world", "nested": "hash"}, "foo"]`))
}

func TestYAMLEq_EqualYAMLString(t *testing.T) {
	mockT := new(testing.T)
	True(t, YAMLEq(mockT, `{"hello": "world", "foo": "bar"}`, `{"hello": "world", "foo": "bar"}`))
}

func TestYAMLEq_EquivalentButNotEqual(t *testing.T) {
	mockT := new(testing.T)
	True(t, YAMLEq(mockT, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`))
}

func TestYAMLEq_HashOfArraysAndHashes(t *testing.T) {
	mockT := new(testing.T)
	expected := `
numeric: 1.5
array:
  - foo: bar
  - 1
  - "string"
  - ["nested", "array", 5.5]
hash:
  nested: hash
  nested_slice: [this, is, nested]
string: "foo"
`

	actual := `
numeric: 1.5
hash:
  nested: hash
  nested_slice: [this, is, nested]
string: "foo"
array:
  - foo: bar
  - 1
  - "string"
  - ["nested", "array", 5.5]
`
	True(t, YAMLEq(mockT, expected, actual))
}

func TestYAMLEq_Array(t *testing.T) {
	mockT := new(testing.T)
	True(t, YAMLEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `["foo", {"nested": "hash", "hello": "world"}]`))
}

func TestYAMLEq_HashAndArrayNotEquivalent(t *testing.T) {
	mockT := new(testing.T)
	False(t, YAMLEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `{"foo": "bar", {"nested": "hash", "hello": "world"}}`))
}

func TestYAMLEq_HashesNotEquivalent(t *testing.T) {
	mockT := new(testing.T)
	False(t, YAMLEq(mockT, `{"foo": "bar"}`, `{"foo": "bar", "hello": "world"}`))
}

func TestYAMLEq_ActualIsSimpleString(t *testing.T) {
	mockT := new(testing.T)
	False(t, YAMLEq(mockT, `{"foo": "bar"}`, "Simple String"))
}

func TestYAMLEq_ExpectedIsSimpleString(t *testing.T) {
	mockT := new(testing.T)
	False(t, YAMLEq(mockT, "Simple String", `{"foo": "bar", "hello": "world"}`))
}

func TestYAMLEq_ExpectedAndActualSimpleString(t *testing.T) {
	mockT := new(testing.T)
	True(t, YAMLEq(mockT, "Simple String", "Simple String"))
}

func TestYAMLEq_ArraysOfDifferentOrder(t *testing.T) {
	mockT := new(testing.T)
	False(t, YAMLEq(mockT, `["foo", {"hello": "world", "nested": "hash"}]`, `[{ "hello": "world", "nested": "hash"}, "foo"]`))
}

type diffTestingStruct struct {
	A string
	B int
}

func (d *diffTestingStruct) String() string {
	return d.A
}

func TestDiff(t *testing.T) {
	expected := `

Diff:
--- Expected
+++ Actual
@@ -1,3 +1,3 @@
 (struct { foo string }) {
- foo: (string) (len=5) "hello"
+ foo: (string) (len=3) "bar"
 }
`
	actual := diff(
		struct{ foo string }{"hello"},
		struct{ foo string }{"bar"},
	)
	Equal(t, expected, actual)

	expected = `

Diff:
--- Expected
+++ Actual
@@ -2,5 +2,5 @@
  (int) 1,
- (int) 2,
  (int) 3,
- (int) 4
+ (int) 5,
+ (int) 7
 }
`
	actual = diff(
		[]int{1, 2, 3, 4},
		[]int{1, 3, 5, 7},
	)
	Equal(t, expected, actual)

	expected = `

Diff:
--- Expected
+++ Actual
@@ -2,4 +2,4 @@
  (int) 1,
- (int) 2,
- (int) 3
+ (int) 3,
+ (int) 5
 }
`
	actual = diff(
		[]int{1, 2, 3, 4}[0:3],
		[]int{1, 3, 5, 7}[0:3],
	)
	Equal(t, expected, actual)

	expected = `

Diff:
--- Expected
+++ Actual
@@ -1,6 +1,6 @@
 (map[string]int) (len=4) {
- (string) (len=4) "four": (int) 4,
+ (string) (len=4) "five": (int) 5,
  (string) (len=3) "one": (int) 1,
- (string) (len=5) "three": (int) 3,
- (string) (len=3) "two": (int) 2
+ (string) (len=5) "seven": (int) 7,
+ (string) (len=5) "three": (int) 3
 }
`

	actual = diff(
		map[string]int{"one": 1, "two": 2, "three": 3, "four": 4},
		map[string]int{"one": 1, "three": 3, "five": 5, "seven": 7},
	)
	Equal(t, expected, actual)

	expected = `

Diff:
--- Expected
+++ Actual
@@ -1,3 +1,3 @@
 (*errors.errorString)({
- s: (string) (len=19) "some expected error"
+ s: (string) (len=12) "actual error"
 })
`

	actual = diff(
		errors.New("some expected error"),
		errors.New("actual error"),
	)
	Equal(t, expected, actual)

	expected = `

Diff:
--- Expected
+++ Actual
@@ -2,3 +2,3 @@
  A: (string) (len=11) "some string",
- B: (int) 10
+ B: (int) 15
 }
`

	actual = diff(
		diffTestingStruct{A: "some string", B: 10},
		diffTestingStruct{A: "some string", B: 15},
	)
	Equal(t, expected, actual)
}

func TestTimeEqualityErrorFormatting(t *testing.T) {
	mockT := new(mockTestingT)

	Equal(mockT, time.Second*2, time.Millisecond)

	expectedErr := "\\s+Error Trace:\\s+Error:\\s+Not equal:\\s+\n\\s+expected: 2s\n\\s+actual\\s+: 1ms\n"
	Regexp(t, regexp.MustCompile(expectedErr), mockT.errorString())
}

func TestDiffEmptyCases(t *testing.T) {
	Equal(t, "", diff(nil, nil))
	Equal(t, "", diff(struct{ foo string }{}, nil))
	Equal(t, "", diff(nil, struct{ foo string }{}))
	Equal(t, "", diff(1, 2))
	Equal(t, "", diff(1, 2))
	Equal(t, "", diff([]int{1}, []bool{true}))
}

// Ensure there are no data races
func TestDiffRace(t *testing.T) {
	t.Parallel()

	expected := map[string]string{
		"a": "A",
		"b": "B",
		"c": "C",
	}

	actual := map[string]string{
		"d": "D",
		"e": "E",
		"f": "F",
	}

	// run diffs in parallel simulating tests with t.Parallel()
	numRoutines := 10
	rChans := make([]chan string, numRoutines)
	for idx := range rChans {
		rChans[idx] = make(chan string)
		go func(ch chan string) {
			defer close(ch)
			ch <- diff(expected, actual)
		}(rChans[idx])
	}

	for _, ch := range rChans {
		for msg := range ch {
			NotZero(t, msg) // dummy assert
		}
	}
}

type mockTestingT struct {
	errorFmt string
	args     []interface{}
}

func (m *mockTestingT) errorString() string {
	return fmt.Sprintf(m.errorFmt, m.args...)
}

func (m *mockTestingT) Errorf(format string, args ...interface{}) {
	m.errorFmt = format
	m.args = args
}

func TestFailNowWithPlainTestingT(t *testing.T) {
	mockT := &mockTestingT{}

	Panics(t, func() {
		FailNow(mockT, "failed")
	}, "should panic since mockT is missing FailNow()")
}

type mockFailNowTestingT struct {
}

func (m *mockFailNowTestingT) Errorf(format string, args ...interface{}) {}

func (m *mockFailNowTestingT) FailNow() {}

func TestFailNowWithFullTestingT(t *testing.T) {
	mockT := &mockFailNowTestingT{}

	NotPanics(t, func() {
		FailNow(mockT, "failed")
	}, "should call mockT.FailNow() rather than panicking")
}

func TestBytesEqual(t *testing.T) {
	var cases = []struct {
		a, b []byte
	}{
		{make([]byte, 2), make([]byte, 2)},
		{make([]byte, 2), make([]byte, 2, 3)},
		{nil, make([]byte, 0)},
	}
	for i, c := range cases {
		Equal(t, reflect.DeepEqual(c.a, c.b), ObjectsAreEqual(c.a, c.b), "case %d failed", i+1)
	}
}

func BenchmarkBytesEqual(b *testing.B) {
	const size = 1024 * 8
	s := make([]byte, size)
	for i := range s {
		s[i] = byte(i % 255)
	}
	s2 := make([]byte, size)
	copy(s2, s)

	mockT := &mockFailNowTestingT{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Equal(mockT, s, s2)
	}
}

func BenchmarkNotNil(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NotNil(b, b)
	}
}

func ExampleComparisonAssertionFunc() {
	t := &testing.T{} // provided by test

	adder := func(x, y int) int {
		return x + y
	}

	type args struct {
		x int
		y int
	}

	tests := []struct {
		name      string
		args      args
		expect    int
		assertion ComparisonAssertionFunc
	}{
		{"2+2=4", args{2, 2}, 4, Equal},
		{"2+2!=5", args{2, 2}, 5, NotEqual},
		{"2+3==5", args{2, 3}, 5, Exactly},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, tt.expect, adder(tt.args.x, tt.args.y))
		})
	}
}

func TestComparisonAssertionFunc(t *testing.T) {
	type iface interface {
		Name() string
	}

	tests := []struct {
		name      string
		expect    interface{}
		got       interface{}
		assertion ComparisonAssertionFunc
	}{
		{"implements", (*iface)(nil), t, Implements},
		{"isType", (*testing.T)(nil), t, IsType},
		{"equal", t, t, Equal},
		{"equalValues", t, t, EqualValues},
		{"notEqualValues", t, nil, NotEqualValues},
		{"exactly", t, t, Exactly},
		{"notEqual", t, nil, NotEqual},
		{"notContains", []int{1, 2, 3}, 4, NotContains},
		{"subset", []int{1, 2, 3, 4}, []int{2, 3}, Subset},
		{"notSubset", []int{1, 2, 3, 4}, []int{0, 3}, NotSubset},
		{"elementsMatch", []byte("abc"), []byte("bac"), ElementsMatch},
		{"regexp", "^t.*y$", "testify", Regexp},
		{"notRegexp", "^t.*y$", "Testify", NotRegexp},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, tt.expect, tt.got)
		})
	}
}

func ExampleValueAssertionFunc() {
	t := &testing.T{} // provided by test

	dumbParse := func(input string) interface{} {
		var x interface{}
		json.Unmarshal([]byte(input), &x)
		return x
	}

	tests := []struct {
		name      string
		arg       string
		assertion ValueAssertionFunc
	}{
		{"true is not nil", "true", NotNil},
		{"empty string is nil", "", Nil},
		{"zero is not nil", "0", NotNil},
		{"zero is zero", "0", Zero},
		{"false is zero", "false", Zero},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, dumbParse(tt.arg))
		})
	}
}

func TestValueAssertionFunc(t *testing.T) {
	tests := []struct {
		name      string
		value     interface{}
		assertion ValueAssertionFunc
	}{
		{"notNil", true, NotNil},
		{"nil", nil, Nil},
		{"empty", []int{}, Empty},
		{"notEmpty", []int{1}, NotEmpty},
		{"zero", false, Zero},
		{"notZero", 42, NotZero},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, tt.value)
		})
	}
}

func ExampleBoolAssertionFunc() {
	t := &testing.T{} // provided by test

	isOkay := func(x int) bool {
		return x >= 42
	}

	tests := []struct {
		name      string
		arg       int
		assertion BoolAssertionFunc
	}{
		{"-1 is bad", -1, False},
		{"42 is good", 42, True},
		{"41 is bad", 41, False},
		{"45 is cool", 45, True},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, isOkay(tt.arg))
		})
	}
}

func TestBoolAssertionFunc(t *testing.T) {
	tests := []struct {
		name      string
		value     bool
		assertion BoolAssertionFunc
	}{
		{"true", true, True},
		{"false", false, False},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, tt.value)
		})
	}
}

func ExampleErrorAssertionFunc() {
	t := &testing.T{} // provided by test

	dumbParseNum := func(input string, v interface{}) error {
		return json.Unmarshal([]byte(input), v)
	}

	tests := []struct {
		name      string
		arg       string
		assertion ErrorAssertionFunc
	}{
		{"1.2 is number", "1.2", NoError},
		{"1.2.3 not number", "1.2.3", Error},
		{"true is not number", "true", Error},
		{"3 is number", "3", NoError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var x float64
			tt.assertion(t, dumbParseNum(tt.arg, &x))
		})
	}
}

func TestErrorAssertionFunc(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		assertion ErrorAssertionFunc
	}{
		{"noError", nil, NoError},
		{"error", errors.New("whoops"), Error},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assertion(t, tt.err)
		})
	}
}

func TestEventuallyFalse(t *testing.T) {
	mockT := new(testing.T)

	condition := func() bool {
		return false
	}

	False(t, Eventually(mockT, condition, 100*time.Millisecond, 20*time.Millisecond))
}

func TestEventuallyTrue(t *testing.T) {
	state := 0
	condition := func() bool {
		defer func() {
			state += 1
		}()
		return state == 2
	}

	True(t, Eventually(t, condition, 100*time.Millisecond, 20*time.Millisecond))
}

func TestNeverFalse(t *testing.T) {
	condition := func() bool {
		return false
	}

	True(t, Never(t, condition, 100*time.Millisecond, 20*time.Millisecond))
}

func TestNeverTrue(t *testing.T) {
	mockT := new(testing.T)
	state := 0
	condition := func() bool {
		defer func() {
			state = state + 1
		}()
		return state == 2
	}

	False(t, Never(mockT, condition, 100*time.Millisecond, 20*time.Millisecond))
}

func TestEventuallyIssue805(t *testing.T) {
	mockT := new(testing.T)

	NotPanics(t, func() {
		condition := func() bool { <-time.After(time.Millisecond); return true }
		False(t, Eventually(mockT, condition, time.Millisecond, time.Microsecond))
	})
}

func Test_validateEqualArgs(t *testing.T) {
	if validateEqualArgs(func() {}, func() {}) == nil {
		t.Error("non-nil functions should error")
	}

	if validateEqualArgs(func() {}, func() {}) == nil {
		t.Error("non-nil functions should error")
	}

	if validateEqualArgs(nil, nil) != nil {
		t.Error("nil functions are equal")
	}
}

func Test_truncatingFormat(t *testing.T) {

	original := strings.Repeat("a", bufio.MaxScanTokenSize-102)
	result := truncatingFormat(original)
	Equal(t, fmt.Sprintf("%#v", original), result, "string should not be truncated")

	original = original + "x"
	result = truncatingFormat(original)
	NotEqual(t, fmt.Sprintf("%#v", original), result, "string should have been truncated.")

	if !strings.HasSuffix(result, "<... truncated>") {
		t.Error("truncated string should have <... truncated> suffix")
	}
}
