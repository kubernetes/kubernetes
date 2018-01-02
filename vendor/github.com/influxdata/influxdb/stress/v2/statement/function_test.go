package statement

import (
	"testing"
)

func TestNewStrRandStringer(t *testing.T) {
	function := newStrRandFunction()
	strRandStringer := function.NewStringer(10)
	s := strRandStringer()
	if len(s) != function.Argument {
		t.Errorf("Expected: %v\nGot: %v\n", function.Argument, len(s))
	}
}

func TestNewIntIncStringer(t *testing.T) {
	function := newIntIncFunction()
	intIncStringer := function.NewStringer(10)
	s := intIncStringer()
	if s != "0i" {
		t.Errorf("Expected: 0i\nGot: %v\n", s)
	}
}

func TestNewIntRandStringer(t *testing.T) {
	function := newIntRandFunction()
	intRandStringer := function.NewStringer(10)
	s := intRandStringer()
	if parseInt(s[:len(s)-1]) > function.Argument {
		t.Errorf("Expected value below: %v\nGot value: %v\n", function.Argument, s)
	}
}

func TestNewFloatIncStringer(t *testing.T) {
	function := newFloatIncFunction()
	floatIncStringer := function.NewStringer(10)
	s := floatIncStringer()
	if parseFloat(s) != function.Argument {
		t.Errorf("Expected value: %v\nGot: %v\n", function.Argument, s)
	}
}
func TestNewFloatRandStringer(t *testing.T) {
	function := newFloatRandFunction()
	floatRandStringer := function.NewStringer(10)
	s := floatRandStringer()
	if parseFloat(s) > function.Argument {
		t.Errorf("Expected value below: %v\nGot value: %v\n", function.Argument, s)
	}
}

func TestStringersEval(t *testing.T) {
	// Make the *Function(s)
	strRandFunction := newStrRandFunction()
	intIncFunction := newIntIncFunction()
	intRandFunction := newIntRandFunction()
	floatIncFunction := newFloatIncFunction()
	floatRandFunction := newFloatRandFunction()
	// Make the *Stringer(s)
	strRandStringer := strRandFunction.NewStringer(10)
	intIncStringer := intIncFunction.NewStringer(10)
	intRandStringer := intRandFunction.NewStringer(10)
	floatIncStringer := floatIncFunction.NewStringer(10)
	floatRandStringer := floatRandFunction.NewStringer(10)
	// Make the *Stringers
	stringers := Stringers([]Stringer{strRandStringer, intIncStringer, intRandStringer, floatIncStringer, floatRandStringer})
	// Spoff the Time function
	// Call *Stringers.Eval
	values := stringers.Eval(spoofTime)
	// Check the strRandFunction
	if len(values[0].(string)) != strRandFunction.Argument {
		t.Errorf("Expected: %v\nGot: %v\n", strRandFunction.Argument, len(values[0].(string)))
	}
	// Check the intIncFunction
	if values[1].(string) != "0i" {
		t.Errorf("Expected: 0i\nGot: %v\n", values[1].(string))
	}
	// Check the intRandFunction
	s := values[2].(string)
	if parseInt(s[:len(s)-1]) > intRandFunction.Argument {
		t.Errorf("Expected value below: %v\nGot value: %v\n", intRandFunction.Argument, s)
	}
	// Check the floatIncFunction
	if parseFloat(values[3].(string)) != floatIncFunction.Argument {
		t.Errorf("Expected value: %v\nGot: %v\n", floatIncFunction.Argument, values[3])
	}
	// Check the floatRandFunction
	if parseFloat(values[4].(string)) > floatRandFunction.Argument {
		t.Errorf("Expected value below: %v\nGot value: %v\n", floatRandFunction.Argument, values[4])
	}
	// Check the spoofTime func
	if values[5] != 8 {

	}
}

func spoofTime() int64 {
	return int64(8)
}

func newStrRandFunction() *Function {
	return &Function{
		Type:     "str",
		Fn:       "rand",
		Argument: 8,
		Count:    1000,
	}
}

func newIntIncFunction() *Function {
	return &Function{
		Type:     "int",
		Fn:       "inc",
		Argument: 0,
		Count:    0,
	}
}

func newIntRandFunction() *Function {
	return &Function{
		Type:     "int",
		Fn:       "rand",
		Argument: 100,
		Count:    1000,
	}
}

func newFloatIncFunction() *Function {
	return &Function{
		Type:     "float",
		Fn:       "inc",
		Argument: 0,
		Count:    1000,
	}
}

func newFloatRandFunction() *Function {
	return &Function{
		Type:     "float",
		Fn:       "rand",
		Argument: 100,
		Count:    1000,
	}
}
