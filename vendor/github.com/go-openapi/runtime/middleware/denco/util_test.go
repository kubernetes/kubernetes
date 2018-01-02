package denco_test

import (
	"reflect"
	"testing"

	"github.com/go-openapi/runtime/middleware/denco"
)

func TestNextSeparator(t *testing.T) {
	for _, testcase := range []struct {
		path     string
		start    int
		expected interface{}
	}{
		{"/path/to/route", 0, 0},
		{"/path/to/route", 1, 5},
		{"/path/to/route", 9, 14},
		{"/path.html", 1, 10},
		{"/foo/bar.html", 1, 4},
		{"/foo/bar.html/baz.png", 5, 13},
		{"/foo/bar.html/baz.png", 14, 21},
		{"path#", 0, 4},
	} {
		actual := denco.NextSeparator(testcase.path, testcase.start)
		expected := testcase.expected
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("path = %q, start = %v expect %v, but %v", testcase.path, testcase.start, expected, actual)
		}
	}
}
