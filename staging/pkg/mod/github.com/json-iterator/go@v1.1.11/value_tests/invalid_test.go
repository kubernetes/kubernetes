package test

import (
	"bytes"
	"encoding/json"
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func Test_missing_object_end(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Metric string                 `json:"metric"`
		Tags   map[string]interface{} `json:"tags"`
	}
	obj := TestObject{}
	should.NotNil(jsoniter.UnmarshalFromString(`{"metric": "sys.777","tags": {"a":"123"}`, &obj))
}

func Test_missing_array_end(t *testing.T) {
	should := require.New(t)
	should.NotNil(jsoniter.UnmarshalFromString(`[1,2,3`, &[]int{}))
}

func Test_invalid_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[]"))
	should.Equal(jsoniter.InvalidValue, any.Get(0.3).ValueType())
	// is nil correct ?
	should.Equal(nil, any.Get(0.3).GetInterface())

	any = any.Get(0.3)
	should.Equal(false, any.ToBool())
	should.Equal(int(0), any.ToInt())
	should.Equal(int32(0), any.ToInt32())
	should.Equal(int64(0), any.ToInt64())
	should.Equal(uint(0), any.ToUint())
	should.Equal(uint32(0), any.ToUint32())
	should.Equal(uint64(0), any.ToUint64())
	should.Equal(float32(0), any.ToFloat32())
	should.Equal(float64(0), any.ToFloat64())
	should.Equal("", any.ToString())

	should.Equal(jsoniter.InvalidValue, any.Get(0.1).Get(1).ValueType())
}

func Test_invalid_struct_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{54, 141, 30}
	obj := TestObject{}
	should.NotNil(jsoniter.Unmarshal(input, &obj))
}

func Test_invalid_slice_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{93}
	obj := []string{}
	should.NotNil(jsoniter.Unmarshal(input, &obj))
}

func Test_invalid_array_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{93}
	obj := [0]string{}
	should.NotNil(jsoniter.Unmarshal(input, &obj))
}

func Test_invalid_float(t *testing.T) {
	inputs := []string{
		`1.e1`, // dot without following digit
		`1.`,   // dot can not be the last char
		``,     // empty number
		`01`,   // extra leading zero
		`-`,    // negative without digit
		`--`,   // double negative
		`--2`,  // double negative
	}
	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			should := require.New(t)
			iter := jsoniter.ParseString(jsoniter.ConfigDefault, input+",")
			iter.Skip()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
			v := float64(0)
			should.NotNil(json.Unmarshal([]byte(input), &v))
			iter = jsoniter.ParseString(jsoniter.ConfigDefault, input+",")
			iter.ReadFloat64()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
			iter = jsoniter.ParseString(jsoniter.ConfigDefault, input+",")
			iter.ReadFloat32()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
		})
	}
}

func Test_chan(t *testing.T) {
	type TestObject struct {
		MyChan  chan bool
		MyField int
	}

	obj := TestObject{}

	t.Run("Encode channel", func(t *testing.T) {
		should := require.New(t)
		str, err := jsoniter.Marshal(obj)
		should.NotNil(err)
		should.Nil(str)
	})

	t.Run("Encode channel using compatible configuration", func(t *testing.T) {
		should := require.New(t)
		str, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(obj)
		should.NotNil(err)
		should.Nil(str)
	})
}

func Test_invalid_in_map(t *testing.T) {
	testMap := map[string]interface{}{"chan": make(chan interface{})}

	t.Run("Encode map with invalid content", func(t *testing.T) {
		should := require.New(t)
		str, err := jsoniter.Marshal(testMap)
		should.NotNil(err)
		should.Nil(str)
	})

	t.Run("Encode map with invalid content using compatible configuration", func(t *testing.T) {
		should := require.New(t)
		str, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(testMap)
		should.NotNil(err)
		should.Nil(str)
	})
}

func Test_invalid_number(t *testing.T) {
	type Message struct {
		Number int `json:"number"`
	}
	obj := Message{}
	decoder := jsoniter.ConfigCompatibleWithStandardLibrary.NewDecoder(bytes.NewBufferString(`{"number":"5"}`))
	err := decoder.Decode(&obj)
	invalidStr := err.Error()
	result, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(invalidStr)
	should := require.New(t)
	should.Nil(err)
	result2, err := json.Marshal(invalidStr)
	should.Nil(err)
	should.Equal(string(result2), string(result))
}

func Test_valid(t *testing.T) {
	should := require.New(t)
	should.True(jsoniter.Valid([]byte(`{}`)))
	should.False(jsoniter.Valid([]byte(`{`)))
}

func Test_nil_pointer(t *testing.T) {
	should := require.New(t)
	data := []byte(`{"A":0}`)
	type T struct {
		X int
	}
	var obj *T
	err := jsoniter.Unmarshal(data, obj)
	should.NotNil(err)
}

func Test_func_pointer_type(t *testing.T) {
	type TestObject2 struct {
		F func()
	}
	type TestObject1 struct {
		Obj *TestObject2
	}
	t.Run("encode null is valid", func(t *testing.T) {
		should := require.New(t)
		output, err := json.Marshal(TestObject1{})
		should.Nil(err)
		should.Equal(`{"Obj":null}`, string(output))
		output, err = jsoniter.Marshal(TestObject1{})
		should.Nil(err)
		should.Equal(`{"Obj":null}`, string(output))
	})
	t.Run("encode not null is invalid", func(t *testing.T) {
		should := require.New(t)
		_, err := json.Marshal(TestObject1{Obj: &TestObject2{}})
		should.NotNil(err)
		_, err = jsoniter.Marshal(TestObject1{Obj: &TestObject2{}})
		should.NotNil(err)
	})
	t.Run("decode null is valid", func(t *testing.T) {
		should := require.New(t)
		var obj TestObject1
		should.Nil(json.Unmarshal([]byte(`{"Obj":{"F": null}}`), &obj))
		should.Nil(jsoniter.Unmarshal([]byte(`{"Obj":{"F": null}}`), &obj))
	})
	t.Run("decode not null is invalid", func(t *testing.T) {
		should := require.New(t)
		var obj TestObject1
		should.NotNil(json.Unmarshal([]byte(`{"Obj":{"F": "hello"}}`), &obj))
		should.NotNil(jsoniter.Unmarshal([]byte(`{"Obj":{"F": "hello"}}`), &obj))
	})
}

func TestEOF(t *testing.T) {
	var s string
	err := jsoniter.ConfigCompatibleWithStandardLibrary.NewDecoder(&bytes.Buffer{}).Decode(&s)
	assert.Equal(t, io.EOF, err)
}

func TestDecodeErrorType(t *testing.T) {
	should := require.New(t)
	var err error
	should.Nil(jsoniter.Unmarshal([]byte("null"), &err))
	should.NotNil(jsoniter.Unmarshal([]byte("123"), &err))
}

func Test_decode_slash(t *testing.T) {
	should := require.New(t)
	var obj interface{}
	should.NotNil(json.Unmarshal([]byte("\\"), &obj))
	should.NotNil(jsoniter.UnmarshalFromString("\\", &obj))
}

func Test_NilInput(t *testing.T) {
	var jb []byte // nil
	var out string
	err := jsoniter.Unmarshal(jb, &out)
	if err == nil {
		t.Errorf("Expected error")
	}
}

func Test_EmptyInput(t *testing.T) {
	jb := []byte("")
	var out string
	err := jsoniter.Unmarshal(jb, &out)
	if err == nil {
		t.Errorf("Expected error")
	}
}

type Foo struct {
	A jsoniter.Any
}

func Test_nil_any(t *testing.T) {
	should := require.New(t)
	data, _ := jsoniter.Marshal(&Foo{})
	should.Equal(`{"A":null}`, string(data))
}
