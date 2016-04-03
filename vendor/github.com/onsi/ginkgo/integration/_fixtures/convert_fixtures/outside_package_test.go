package tmp_test

import (
	"testing"
)

type UselessStruct struct {
	ImportantField string
}

func TestSomethingImportant(t *testing.T) {
	whatever := &UselessStruct{}
	if whatever.ImportantField != "SECRET_PASSWORD" {
		t.Fail()
	}
}
