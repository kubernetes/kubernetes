package test

import (
	"encoding/json"
	"fmt"
	"github.com/json-iterator/go"
	"github.com/modern-go/reflect2"
	"github.com/stretchr/testify/require"
	"testing"
)

type unmarshalCase struct {
	obj      func() interface{}
	ptr      interface{}
	input    string
	selected bool
}

var unmarshalCases []unmarshalCase

var marshalCases = []interface{}{
	nil,
}

type selectedMarshalCase struct {
	marshalCase interface{}
}

func Test_unmarshal(t *testing.T) {
	for _, testCase := range unmarshalCases {
		if testCase.selected {
			unmarshalCases = []unmarshalCase{testCase}
			break
		}
	}
	for i, testCase := range unmarshalCases {
		t.Run(fmt.Sprintf("[%v]%s", i, testCase.input), func(t *testing.T) {
			should := require.New(t)
			var obj1 interface{}
			var obj2 interface{}
			if testCase.obj != nil {
				obj1 = testCase.obj()
				obj2 = testCase.obj()
			} else {
				valType := reflect2.TypeOfPtr(testCase.ptr).Elem()
				obj1 = valType.New()
				obj2 = valType.New()
			}
			err1 := json.Unmarshal([]byte(testCase.input), obj1)
			should.NoError(err1, "json")
			err2 := jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal([]byte(testCase.input), obj2)
			should.NoError(err2, "jsoniter")
			should.Equal(obj1, obj2)
		})
	}
}

func Test_marshal(t *testing.T) {
	for _, testCase := range marshalCases {
		selectedMarshalCase, found := testCase.(selectedMarshalCase)
		if found {
			marshalCases = []interface{}{selectedMarshalCase.marshalCase}
			break
		}
	}
	for i, testCase := range marshalCases {
		var name string
		if testCase != nil {
			name = fmt.Sprintf("[%v]%v/%s", i, testCase, reflect2.TypeOf(testCase).String())
		}
		t.Run(name, func(t *testing.T) {
			should := require.New(t)
			output1, err1 := json.Marshal(testCase)
			should.NoError(err1, "json")
			output2, err2 := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(testCase)
			should.NoError(err2, "jsoniter")
			should.Equal(string(output1), string(output2))
		})
	}
}
