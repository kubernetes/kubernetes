package govalidator

import (
	"fmt"
	"testing"
)

func TestToInt(t *testing.T) {
	tests := []string{"1000", "-123", "abcdef", "100000000000000000000000000000000000000000000"}
	expected := []int64{1000, -123, 0, 0}
	for i := 0; i < len(tests); i++ {
		result, _ := ToInt(tests[i])
		if result != expected[i] {
			t.Log("Case ", i, ": expected ", expected[i], " when result is ", result)
			t.FailNow()
		}
	}
}

func TestToBoolean(t *testing.T) {
	tests := []string{"true", "1", "True", "false", "0", "abcdef"}
	expected := []bool{true, true, true, false, false, false}
	for i := 0; i < len(tests); i++ {
		res, _ := ToBoolean(tests[i])
		if res != expected[i] {
			t.Log("Case ", i, ": expected ", expected[i], " when result is ", res)
			t.FailNow()
		}
	}
}

func toString(t *testing.T, test interface{}, expected string) {
	res := ToString(test)
	if res != expected {
		t.Log("Case ToString: expected ", expected, " when result is ", res)
		t.FailNow()
	}
}

func TestToString(t *testing.T) {
	toString(t, "str123", "str123")
	toString(t, 123, "123")
	toString(t, 12.3, "12.3")
	toString(t, true, "true")
	toString(t, 1.5+10i, "(1.5+10i)")
	// Sprintf function not guarantee that maps with equal keys always will be equal in string  representation
	//toString(t, struct{ Keys map[int]int }{Keys: map[int]int{1: 2, 3: 4}}, "{map[1:2 3:4]}")
}

func TestToFloat(t *testing.T) {
	tests := []string{"", "123", "-.01", "10.", "string", "1.23e3", ".23e10"}
	expected := []float64{0, 123, -0.01, 10.0, 0, 1230, 0.23e10}
	for i := 0; i < len(tests); i++ {
		res, _ := ToFloat(tests[i])
		if res != expected[i] {
			t.Log("Case ", i, ": expected ", expected[i], " when result is ", res)
			t.FailNow()
		}
	}
}

func TestToJSON(t *testing.T) {
	tests := []interface{}{"test", map[string]string{"a": "b", "b": "c"}, func() error { return fmt.Errorf("Error") }}
	expected := [][]string{
		[]string{"\"test\"", "<nil>"},
		[]string{"{\"a\":\"b\",\"b\":\"c\"}", "<nil>"},
		[]string{"", "json: unsupported type: func() error"},
	}
	for i, test := range tests {
		actual, err := ToJSON(test)
		if actual != expected[i][0] {
			t.Errorf("Expected toJSON(%v) to return '%v', got '%v'", test, expected[i][0], actual)
		}
		if fmt.Sprintf("%v", err) != expected[i][1] {
			t.Errorf("Expected error returned from toJSON(%v) to return '%v', got '%v'", test, expected[i][1], fmt.Sprintf("%v", err))
		}
	}
}
