package tmp

import (
	"testing"
)

type UselessStruct struct {
	ImportantField string
	T              *testing.T
}

var testFunc = func(t *testing.T, arg *string) {}

func assertEqual(t *testing.T, arg1, arg2 interface{}) {
	if arg1 != arg2 {
		t.Fail()
	}
}

func TestSomethingImportant(t *testing.T) {
	whatever := &UselessStruct{
		T:              t,
		ImportantField: "SECRET_PASSWORD",
	}
	something := &UselessStruct{ImportantField: "string value"}
	assertEqual(t, whatever.ImportantField, "SECRET_PASSWORD")
	assertEqual(t, something.ImportantField, "string value")

	var foo = func(t *testing.T) {}
	foo(t)

	strp := "something"
	testFunc(t, &strp)
	t.Fail()
}

func Test3Things(t *testing.T) {
	if 3 != 3 {
		t.Fail()
	}
}
