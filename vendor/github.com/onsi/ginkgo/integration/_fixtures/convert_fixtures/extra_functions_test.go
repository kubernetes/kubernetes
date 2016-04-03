package tmp

import (
	"testing"
)

func TestSomethingLessImportant(t *testing.T) {
	strp := "hello!"
	somethingImportant(t, &strp)
}

func somethingImportant(t *testing.T, message *string) {
	t.Log("Something important happened in a test: " + *message)
}
