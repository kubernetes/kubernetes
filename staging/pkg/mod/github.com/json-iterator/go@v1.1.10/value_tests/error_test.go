package test

import (
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"reflect"
	"testing"
)

func Test_errorInput(t *testing.T) {
	for _, testCase := range unmarshalCases {
		if testCase.obj != nil {
			continue
		}
		valType := reflect.TypeOf(testCase.ptr).Elem()
		t.Run(valType.String(), func(t *testing.T) {
			for _, data := range []string{
				`x`,
				`n`,
				`nul`,
				`{x}`,
				`{"x"}`,
				`{"x": "y"x}`,
				`{"x": "y"`,
				`{"x": "y", "a"}`,
				`[`,
				`[{"x": "y"}`,
			} {
				ptrVal := reflect.New(valType)
				ptr := ptrVal.Interface()
				err := jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal([]byte(data), ptr)
				require.Error(t, err, "on input %q", data)
			}
		})
	}
}
