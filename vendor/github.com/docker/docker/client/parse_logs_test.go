package client

import (
	"reflect"
	"testing"

	"github.com/pkg/errors"
)

func TestParseLogDetails(t *testing.T) {
	testCases := []struct {
		line     string
		expected map[string]string
		err      error
	}{
		{"key=value", map[string]string{"key": "value"}, nil},
		{"key1=value1,key2=value2", map[string]string{"key1": "value1", "key2": "value2"}, nil},
		{"key+with+spaces=value%3Dequals,asdf%2C=", map[string]string{"key with spaces": "value=equals", "asdf,": ""}, nil},
		{"key=,=nothing", map[string]string{"key": "", "": "nothing"}, nil},
		{"=", map[string]string{"": ""}, nil},
		{"errors", nil, errors.New("invalid details format")},
	}
	for _, tc := range testCases {
		tc := tc // capture range variable
		t.Run(tc.line, func(t *testing.T) {
			t.Parallel()
			res, err := ParseLogDetails(tc.line)
			if err != nil && (err.Error() != tc.err.Error()) {
				t.Fatalf("unexpected error parsing logs:\nExpected:\n\t%v\nActual:\n\t%v", tc.err, err)
			}
			if !reflect.DeepEqual(tc.expected, res) {
				t.Errorf("result does not match expected:\nExpected:\n\t%#v\nActual:\n\t%#v", tc.expected, res)
			}
		})
	}
}
