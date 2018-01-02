package jsoniter

import (
	"bytes"
	"encoding/json"
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
	should.NotNil(UnmarshalFromString(`{"metric": "sys.777","tags": {"a":"123"}`, &obj))
}

func Test_missing_array_end(t *testing.T) {
	should := require.New(t)
	should.NotNil(UnmarshalFromString(`[1,2,3`, &[]int{}))
}

func Test_invalid_any(t *testing.T) {
	should := require.New(t)
	any := Get([]byte("[]"))
	should.Equal(InvalidValue, any.Get(0.3).ValueType())
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

	should.Equal(InvalidValue, any.Get(0.1).Get(1).ValueType())
}

func Test_invalid_struct_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{54, 141, 30}
	obj := TestObject{}
	should.NotNil(Unmarshal(input, &obj))
}

func Test_invalid_slice_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{93}
	obj := []string{}
	should.NotNil(Unmarshal(input, &obj))
}

func Test_invalid_array_input(t *testing.T) {
	should := require.New(t)
	type TestObject struct{}
	input := []byte{93}
	obj := [0]string{}
	should.NotNil(Unmarshal(input, &obj))
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
			iter := ParseString(ConfigDefault, input+",")
			iter.Skip()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
			v := float64(0)
			should.NotNil(json.Unmarshal([]byte(input), &v))
			iter = ParseString(ConfigDefault, input+",")
			iter.ReadFloat64()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
			iter = ParseString(ConfigDefault, input+",")
			iter.ReadFloat32()
			should.NotEqual(io.EOF, iter.Error)
			should.NotNil(iter.Error)
		})
	}
}

func Test_chan(t *testing.T) {
	t.Skip("do not support chan")

	type TestObject struct {
		MyChan  chan bool
		MyField int
	}

	should := require.New(t)
	obj := TestObject{}
	str, err := json.Marshal(obj)
	should.Nil(err)
	should.Equal(``, str)
}

func Test_invalid_number(t *testing.T) {
	type Message struct {
		Number int `json:"number"`
	}
	obj := Message{}
	decoder := ConfigCompatibleWithStandardLibrary.NewDecoder(bytes.NewBufferString(`{"number":"5"}`))
	err := decoder.Decode(&obj)
	invalidStr := err.Error()
	result, err := ConfigCompatibleWithStandardLibrary.Marshal(invalidStr)
	should := require.New(t)
	should.Nil(err)
	result2, err := json.Marshal(invalidStr)
	should.Nil(err)
	should.Equal(string(result2), string(result))
}

func Test_valid(t *testing.T) {
	should := require.New(t)
	should.True(Valid([]byte(`{}`)))
	should.False(Valid([]byte(`{`)))
}
