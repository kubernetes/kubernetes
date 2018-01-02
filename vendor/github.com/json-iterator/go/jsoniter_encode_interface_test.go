package jsoniter

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_encode_interface(t *testing.T) {
	should := require.New(t)
	var a interface{}
	a = int8(10)
	str, err := MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "10")
	a = float32(3)
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "3")
	a = map[string]interface{}{"abc": 1}
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, `{"abc":1}`)
	a = uintptr(1)
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "1")
	a = uint(1)
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "1")
	a = uint8(1)
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "1")
	a = json.RawMessage("abc")
	MarshalToString(a)
	str, err = MarshalToString(a)
	should.Nil(err)
	should.Equal(str, "abc")
}
