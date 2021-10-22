package test

import (
	"encoding/json"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

func Test_use_number_for_unmarshal(t *testing.T) {
	should := require.New(t)
	api := jsoniter.Config{UseNumber: true}.Froze()
	var obj interface{}
	should.Nil(api.UnmarshalFromString("123", &obj))
	should.Equal(json.Number("123"), obj)
}

func Test_customize_float_marshal(t *testing.T) {
	should := require.New(t)
	json := jsoniter.Config{MarshalFloatWith6Digits: true}.Froze()
	str, err := json.MarshalToString(float32(1.23456789))
	should.Nil(err)
	should.Equal("1.234568", str)
}

func Test_customize_tag_key(t *testing.T) {

	type TestObject struct {
		Field string `orm:"field"`
	}

	should := require.New(t)
	json := jsoniter.Config{TagKey: "orm"}.Froze()
	str, err := json.MarshalToString(TestObject{"hello"})
	should.Nil(err)
	should.Equal(`{"field":"hello"}`, str)
}

func Test_read_large_number_as_interface(t *testing.T) {
	should := require.New(t)
	var val interface{}
	err := jsoniter.Config{UseNumber: true}.Froze().UnmarshalFromString(`123456789123456789123456789`, &val)
	should.Nil(err)
	output, err := jsoniter.MarshalToString(val)
	should.Nil(err)
	should.Equal(`123456789123456789123456789`, output)
}

type caseSensitiveStruct struct {
	A string `json:"a"`
	B string `json:"b,omitempty"`
	C *C     `json:"C,omitempty"`
}

type C struct {
	D int64 `json:"D,omitempty"`
	E *E    `json:"e,omitempty"`
}

type E struct {
	F string `json:"F,omitempty"`
}

func Test_CaseSensitive(t *testing.T) {
	should := require.New(t)

	testCases := []struct {
		input          string
		expectedOutput string
		caseSensitive  bool
	}{
		{
			input:          `{"A":"foo","B":"bar"}`,
			expectedOutput: `{"a":"foo","b":"bar"}`,
			caseSensitive:  false,
		},
		{
			input:          `{"a":"foo","b":"bar"}`,
			expectedOutput: `{"a":"foo","b":"bar"}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"foo","b":"bar","C":{"D":10}}`,
			expectedOutput: `{"a":"foo","b":"bar","C":{"D":10}}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"foo","B":"bar","c":{"d":10}}`,
			expectedOutput: `{"a":"foo"}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"foo","C":{"d":10}}`,
			expectedOutput: `{"a":"foo","C":{}}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"foo","C":{"D":10,"e":{"f":"baz"}}}`,
			expectedOutput: `{"a":"foo","C":{"D":10,"e":{}}}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"foo","C":{"D":10,"e":{"F":"baz"}}}`,
			expectedOutput: `{"a":"foo","C":{"D":10,"e":{"F":"baz"}}}`,
			caseSensitive:  true,
		},
		{
			input:          `{"A":"foo","c":{"d":10,"E":{"f":"baz"}}}`,
			expectedOutput: `{"a":"foo","C":{"D":10,"e":{"F":"baz"}}}`,
			caseSensitive:  false,
		},
	}

	for _, tc := range testCases {
		val := caseSensitiveStruct{}
		err := jsoniter.Config{CaseSensitive: tc.caseSensitive}.Froze().UnmarshalFromString(tc.input, &val)
		should.Nil(err)

		output, err := jsoniter.MarshalToString(val)
		should.Nil(err)
		should.Equal(tc.expectedOutput, output)
	}
}

type structWithElevenFields struct {
	A string `json:"A,omitempty"`
	B string `json:"B,omitempty"`
	C string `json:"C,omitempty"`
	D string `json:"d,omitempty"`
	E string `json:"e,omitempty"`
	F string `json:"f,omitempty"`
	G string `json:"g,omitempty"`
	H string `json:"h,omitempty"`
	I string `json:"i,omitempty"`
	J string `json:"j,omitempty"`
	K string `json:"k,omitempty"`
}

func Test_CaseSensitive_MoreThanTenFields(t *testing.T) {
	should := require.New(t)

	testCases := []struct {
		input          string
		expectedOutput string
		caseSensitive  bool
	}{
		{
			input:          `{"A":"1","B":"2","C":"3","d":"4","e":"5","f":"6","g":"7","h":"8","i":"9","j":"10","k":"11"}`,
			expectedOutput: `{"A":"1","B":"2","C":"3","d":"4","e":"5","f":"6","g":"7","h":"8","i":"9","j":"10","k":"11"}`,
			caseSensitive:  true,
		},
		{
			input:          `{"a":"1","b":"2","c":"3","D":"4","E":"5","F":"6"}`,
			expectedOutput: `{"A":"1","B":"2","C":"3","d":"4","e":"5","f":"6"}`,
			caseSensitive:  false,
		},
		{
			input:          `{"A":"1","b":"2","d":"4","E":"5"}`,
			expectedOutput: `{"A":"1","d":"4"}`,
			caseSensitive:  true,
		},
	}

	for _, tc := range testCases {
		val := structWithElevenFields{}
		err := jsoniter.Config{CaseSensitive: tc.caseSensitive}.Froze().UnmarshalFromString(tc.input, &val)
		should.Nil(err)

		output, err := jsoniter.MarshalToString(val)
		should.Nil(err)
		should.Equal(tc.expectedOutput, output)
	}
}

type onlyTaggedFieldStruct struct {
	A      string `json:"a"`
	B      string
	FSimpl F `json:"f_simpl"`
	ISimpl I
	FPtr   *F `json:"f_ptr"`
	IPtr   *I
	F
	*I
}

type F struct {
	G string `json:"g"`
	H string
}

type I struct {
	J string `json:"j"`
	K string
}

func Test_OnlyTaggedField(t *testing.T) {
	should := require.New(t)

	obj := onlyTaggedFieldStruct{
		A:      "a",
		B:      "b",
		FSimpl: F{G: "g", H: "h"},
		ISimpl: I{J: "j", K: "k"},
		FPtr:   &F{G: "g", H: "h"},
		IPtr:   &I{J: "j", K: "k"},
		F:      F{G: "g", H: "h"},
		I:      &I{J: "j", K: "k"},
	}

	output, err := jsoniter.Config{OnlyTaggedField: true}.Froze().Marshal(obj)
	should.Nil(err)

	m := make(map[string]interface{})
	err = jsoniter.Unmarshal(output, &m)
	should.Nil(err)

	should.Equal(map[string]interface{}{
		"a": "a",
		"f_simpl": map[string]interface{}{
			"g": "g",
		},
		"f_ptr": map[string]interface{}{
			"g": "g",
		},
		"g": "g",
		"j": "j",
	}, m)
}
