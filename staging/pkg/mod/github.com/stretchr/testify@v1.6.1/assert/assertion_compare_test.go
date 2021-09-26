package assert

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

func TestCompare(t *testing.T) {
	for _, currCase := range []struct {
		less    interface{}
		greater interface{}
		cType   string
	}{
		{less: "a", greater: "b", cType: "string"},
		{less: int(1), greater: int(2), cType: "int"},
		{less: int8(1), greater: int8(2), cType: "int8"},
		{less: int16(1), greater: int16(2), cType: "int16"},
		{less: int32(1), greater: int32(2), cType: "int32"},
		{less: int64(1), greater: int64(2), cType: "int64"},
		{less: uint8(1), greater: uint8(2), cType: "uint8"},
		{less: uint16(1), greater: uint16(2), cType: "uint16"},
		{less: uint32(1), greater: uint32(2), cType: "uint32"},
		{less: uint64(1), greater: uint64(2), cType: "uint64"},
		{less: float32(1.23), greater: float32(2.34), cType: "float32"},
		{less: float64(1.23), greater: float64(2.34), cType: "float64"},
	} {
		resLess, isComparable := compare(currCase.less, currCase.greater, reflect.ValueOf(currCase.less).Kind())
		if !isComparable {
			t.Error("object should be comparable for type " + currCase.cType)
		}

		if resLess != compareLess {
			t.Errorf("object less should be less than greater for type " + currCase.cType)
		}

		resGreater, isComparable := compare(currCase.greater, currCase.less, reflect.ValueOf(currCase.less).Kind())
		if !isComparable {
			t.Error("object are comparable for type " + currCase.cType)
		}

		if resGreater != compareGreater {
			t.Errorf("object greater should be greater than less for type " + currCase.cType)
		}

		resEqual, isComparable := compare(currCase.less, currCase.less, reflect.ValueOf(currCase.less).Kind())
		if !isComparable {
			t.Error("object are comparable for type " + currCase.cType)
		}

		if resEqual != 0 {
			t.Errorf("objects should be equal for type " + currCase.cType)
		}
	}
}

type outputT struct {
	buf *bytes.Buffer
}

// Implements TestingT
func (t *outputT) Errorf(format string, args ...interface{}) {
	s := fmt.Sprintf(format, args...)
	t.buf.WriteString(s)
}

func TestGreater(t *testing.T) {
	mockT := new(testing.T)

	if !Greater(mockT, 2, 1) {
		t.Error("Greater should return true")
	}

	if Greater(mockT, 1, 1) {
		t.Error("Greater should return false")
	}

	if Greater(mockT, 1, 2) {
		t.Error("Greater should return false")
	}

	// Check error report
	for _, currCase := range []struct {
		less    interface{}
		greater interface{}
		msg     string
	}{
		{less: "a", greater: "b", msg: `"a" is not greater than "b"`},
		{less: int(1), greater: int(2), msg: `"1" is not greater than "2"`},
		{less: int8(1), greater: int8(2), msg: `"1" is not greater than "2"`},
		{less: int16(1), greater: int16(2), msg: `"1" is not greater than "2"`},
		{less: int32(1), greater: int32(2), msg: `"1" is not greater than "2"`},
		{less: int64(1), greater: int64(2), msg: `"1" is not greater than "2"`},
		{less: uint8(1), greater: uint8(2), msg: `"1" is not greater than "2"`},
		{less: uint16(1), greater: uint16(2), msg: `"1" is not greater than "2"`},
		{less: uint32(1), greater: uint32(2), msg: `"1" is not greater than "2"`},
		{less: uint64(1), greater: uint64(2), msg: `"1" is not greater than "2"`},
		{less: float32(1.23), greater: float32(2.34), msg: `"1.23" is not greater than "2.34"`},
		{less: float64(1.23), greater: float64(2.34), msg: `"1.23" is not greater than "2.34"`},
	} {
		out := &outputT{buf: bytes.NewBuffer(nil)}
		False(t, Greater(out, currCase.less, currCase.greater))
		Contains(t, string(out.buf.Bytes()), currCase.msg)
	}
}

func TestGreaterOrEqual(t *testing.T) {
	mockT := new(testing.T)

	if !GreaterOrEqual(mockT, 2, 1) {
		t.Error("GreaterOrEqual should return true")
	}

	if !GreaterOrEqual(mockT, 1, 1) {
		t.Error("GreaterOrEqual should return true")
	}

	if GreaterOrEqual(mockT, 1, 2) {
		t.Error("GreaterOrEqual should return false")
	}

	// Check error report
	for _, currCase := range []struct {
		less    interface{}
		greater interface{}
		msg     string
	}{
		{less: "a", greater: "b", msg: `"a" is not greater than or equal to "b"`},
		{less: int(1), greater: int(2), msg: `"1" is not greater than or equal to "2"`},
		{less: int8(1), greater: int8(2), msg: `"1" is not greater than or equal to "2"`},
		{less: int16(1), greater: int16(2), msg: `"1" is not greater than or equal to "2"`},
		{less: int32(1), greater: int32(2), msg: `"1" is not greater than or equal to "2"`},
		{less: int64(1), greater: int64(2), msg: `"1" is not greater than or equal to "2"`},
		{less: uint8(1), greater: uint8(2), msg: `"1" is not greater than or equal to "2"`},
		{less: uint16(1), greater: uint16(2), msg: `"1" is not greater than or equal to "2"`},
		{less: uint32(1), greater: uint32(2), msg: `"1" is not greater than or equal to "2"`},
		{less: uint64(1), greater: uint64(2), msg: `"1" is not greater than or equal to "2"`},
		{less: float32(1.23), greater: float32(2.34), msg: `"1.23" is not greater than or equal to "2.34"`},
		{less: float64(1.23), greater: float64(2.34), msg: `"1.23" is not greater than or equal to "2.34"`},
	} {
		out := &outputT{buf: bytes.NewBuffer(nil)}
		False(t, GreaterOrEqual(out, currCase.less, currCase.greater))
		Contains(t, string(out.buf.Bytes()), currCase.msg)
	}
}

func TestLess(t *testing.T) {
	mockT := new(testing.T)

	if !Less(mockT, 1, 2) {
		t.Error("Less should return true")
	}

	if Less(mockT, 1, 1) {
		t.Error("Less should return false")
	}

	if Less(mockT, 2, 1) {
		t.Error("Less should return false")
	}

	// Check error report
	for _, currCase := range []struct {
		less    interface{}
		greater interface{}
		msg     string
	}{
		{less: "a", greater: "b", msg: `"b" is not less than "a"`},
		{less: int(1), greater: int(2), msg: `"2" is not less than "1"`},
		{less: int8(1), greater: int8(2), msg: `"2" is not less than "1"`},
		{less: int16(1), greater: int16(2), msg: `"2" is not less than "1"`},
		{less: int32(1), greater: int32(2), msg: `"2" is not less than "1"`},
		{less: int64(1), greater: int64(2), msg: `"2" is not less than "1"`},
		{less: uint8(1), greater: uint8(2), msg: `"2" is not less than "1"`},
		{less: uint16(1), greater: uint16(2), msg: `"2" is not less than "1"`},
		{less: uint32(1), greater: uint32(2), msg: `"2" is not less than "1"`},
		{less: uint64(1), greater: uint64(2), msg: `"2" is not less than "1"`},
		{less: float32(1.23), greater: float32(2.34), msg: `"2.34" is not less than "1.23"`},
		{less: float64(1.23), greater: float64(2.34), msg: `"2.34" is not less than "1.23"`},
	} {
		out := &outputT{buf: bytes.NewBuffer(nil)}
		False(t, Less(out, currCase.greater, currCase.less))
		Contains(t, string(out.buf.Bytes()), currCase.msg)
	}
}

func TestLessOrEqual(t *testing.T) {
	mockT := new(testing.T)

	if !LessOrEqual(mockT, 1, 2) {
		t.Error("LessOrEqual should return true")
	}

	if !LessOrEqual(mockT, 1, 1) {
		t.Error("LessOrEqual should return true")
	}

	if LessOrEqual(mockT, 2, 1) {
		t.Error("LessOrEqual should return false")
	}

	// Check error report
	for _, currCase := range []struct {
		less    interface{}
		greater interface{}
		msg     string
	}{
		{less: "a", greater: "b", msg: `"b" is not less than or equal to "a"`},
		{less: int(1), greater: int(2), msg: `"2" is not less than or equal to "1"`},
		{less: int8(1), greater: int8(2), msg: `"2" is not less than or equal to "1"`},
		{less: int16(1), greater: int16(2), msg: `"2" is not less than or equal to "1"`},
		{less: int32(1), greater: int32(2), msg: `"2" is not less than or equal to "1"`},
		{less: int64(1), greater: int64(2), msg: `"2" is not less than or equal to "1"`},
		{less: uint8(1), greater: uint8(2), msg: `"2" is not less than or equal to "1"`},
		{less: uint16(1), greater: uint16(2), msg: `"2" is not less than or equal to "1"`},
		{less: uint32(1), greater: uint32(2), msg: `"2" is not less than or equal to "1"`},
		{less: uint64(1), greater: uint64(2), msg: `"2" is not less than or equal to "1"`},
		{less: float32(1.23), greater: float32(2.34), msg: `"2.34" is not less than or equal to "1.23"`},
		{less: float64(1.23), greater: float64(2.34), msg: `"2.34" is not less than or equal to "1.23"`},
	} {
		out := &outputT{buf: bytes.NewBuffer(nil)}
		False(t, LessOrEqual(out, currCase.greater, currCase.less))
		Contains(t, string(out.buf.Bytes()), currCase.msg)
	}
}

func Test_compareTwoValuesDifferentValuesTypes(t *testing.T) {
	mockT := new(testing.T)

	for _, currCase := range []struct {
		v1            interface{}
		v2            interface{}
		compareResult bool
	}{
		{v1: 123, v2: "abc"},
		{v1: "abc", v2: 123456},
		{v1: float64(12), v2: "123"},
		{v1: "float(12)", v2: float64(1)},
	} {
		compareResult := compareTwoValues(mockT, currCase.v1, currCase.v2, []CompareType{compareLess, compareEqual, compareGreater}, "testFailMessage")
		False(t, compareResult)
	}
}

func Test_compareTwoValuesNotComparableValues(t *testing.T) {
	mockT := new(testing.T)

	type CompareStruct struct {
	}

	for _, currCase := range []struct {
		v1 interface{}
		v2 interface{}
	}{
		{v1: CompareStruct{}, v2: CompareStruct{}},
		{v1: map[string]int{}, v2: map[string]int{}},
		{v1: make([]int, 5, 5), v2: make([]int, 5, 5)},
	} {
		compareResult := compareTwoValues(mockT, currCase.v1, currCase.v2, []CompareType{compareLess, compareEqual, compareGreater}, "testFailMessage")
		False(t, compareResult)
	}
}

func Test_compareTwoValuesCorrectCompareResult(t *testing.T) {
	mockT := new(testing.T)

	for _, currCase := range []struct {
		v1           interface{}
		v2           interface{}
		compareTypes []CompareType
	}{
		{v1: 1, v2: 2, compareTypes: []CompareType{compareLess}},
		{v1: 1, v2: 2, compareTypes: []CompareType{compareLess, compareEqual}},
		{v1: 2, v2: 2, compareTypes: []CompareType{compareGreater, compareEqual}},
		{v1: 2, v2: 2, compareTypes: []CompareType{compareEqual}},
		{v1: 2, v2: 1, compareTypes: []CompareType{compareEqual, compareGreater}},
		{v1: 2, v2: 1, compareTypes: []CompareType{compareGreater}},
	} {
		compareResult := compareTwoValues(mockT, currCase.v1, currCase.v2, currCase.compareTypes, "testFailMessage")
		True(t, compareResult)
	}
}

func Test_containsValue(t *testing.T) {
	for _, currCase := range []struct {
		values []CompareType
		value  CompareType
		result bool
	}{
		{values: []CompareType{compareGreater}, value: compareGreater, result: true},
		{values: []CompareType{compareGreater, compareLess}, value: compareGreater, result: true},
		{values: []CompareType{compareGreater, compareLess}, value: compareLess, result: true},
		{values: []CompareType{compareGreater, compareLess}, value: compareEqual, result: false},
	} {
		compareResult := containsValue(currCase.values, currCase.value)
		Equal(t, currCase.result, compareResult)
	}
}
