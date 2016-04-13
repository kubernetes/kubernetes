package nested

import (
	"testing"
)

func TestSomethingLessImportant(t *testing.T) {
	whatever := &UselessStruct{}
	t.Fail(whatever.ImportantField != "SECRET_PASSWORD")
}
