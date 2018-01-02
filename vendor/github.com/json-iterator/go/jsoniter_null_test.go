package jsoniter

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_read_null(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `null`)
	should.True(iter.ReadNil())
	iter = ParseString(ConfigDefault, `null`)
	should.Nil(iter.Read())
	iter = ParseString(ConfigDefault, `navy`)
	iter.Read()
	should.True(iter.Error != nil && iter.Error != io.EOF)
	iter = ParseString(ConfigDefault, `navy`)
	iter.ReadNil()
	should.True(iter.Error != nil && iter.Error != io.EOF)
}

func Test_write_null(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 4096)
	stream.WriteNil()
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("null", buf.String())
}

func Test_encode_null(t *testing.T) {
	should := require.New(t)
	str, err := MarshalToString(nil)
	should.Nil(err)
	should.Equal("null", str)
}

func Test_decode_null_object_field(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `[null,"a"]`)
	iter.ReadArray()
	if iter.ReadObject() != "" {
		t.FailNow()
	}
	iter.ReadArray()
	if iter.ReadString() != "a" {
		t.FailNow()
	}
	type TestObject struct {
		Field string
	}
	objs := []TestObject{}
	should.Nil(UnmarshalFromString("[null]", &objs))
	should.Len(objs, 1)
}

func Test_decode_null_array_element(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `[null,"a"]`)
	should.True(iter.ReadArray())
	should.True(iter.ReadNil())
	should.True(iter.ReadArray())
	should.Equal("a", iter.ReadString())
}

func Test_decode_null_array(t *testing.T) {
	should := require.New(t)
	arr := []string{}
	should.Nil(UnmarshalFromString("null", &arr))
	should.Nil(arr)
}

func Test_decode_null_map(t *testing.T) {
	should := require.New(t)
	arr := map[string]string{}
	should.Nil(UnmarshalFromString("null", &arr))
	should.Nil(arr)
}

func Test_decode_null_string(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `[null,"a"]`)
	should.True(iter.ReadArray())
	should.Equal("", iter.ReadString())
	should.True(iter.ReadArray())
	should.Equal("a", iter.ReadString())
}

func Test_decode_null_skip(t *testing.T) {
	iter := ParseString(ConfigDefault, `[null,"a"]`)
	iter.ReadArray()
	iter.Skip()
	iter.ReadArray()
	if iter.ReadString() != "a" {
		t.FailNow()
	}
}

func Test_encode_nil_map(t *testing.T) {
	should := require.New(t)
	type Ttest map[string]string
	var obj1 Ttest
	output, err := json.Marshal(obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = json.Marshal(&obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = Marshal(obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = Marshal(&obj1)
	should.Nil(err)
	should.Equal("null", string(output))
}

func Test_encode_nil_array(t *testing.T) {
	should := require.New(t)
	type Ttest []string
	var obj1 Ttest
	output, err := json.Marshal(obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = json.Marshal(&obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = Marshal(obj1)
	should.Nil(err)
	should.Equal("null", string(output))
	output, err = Marshal(&obj1)
	should.Nil(err)
	should.Equal("null", string(output))
}

func Test_decode_nil_num(t *testing.T) {
	type TestData struct {
		Field int `json:"field"`
	}
	should := require.New(t)

	data1 := []byte(`{"field": 42}`)
	data2 := []byte(`{"field": null}`)

	// Checking stdlib behavior as well
	obj2 := TestData{}
	err := json.Unmarshal(data1, &obj2)
	should.Equal(nil, err)
	should.Equal(42, obj2.Field)

	err = json.Unmarshal(data2, &obj2)
	should.Equal(nil, err)
	should.Equal(42, obj2.Field)

	obj := TestData{}

	err = Unmarshal(data1, &obj)
	should.Equal(nil, err)
	should.Equal(42, obj.Field)

	err = Unmarshal(data2, &obj)
	should.Equal(nil, err)
	should.Equal(42, obj.Field)
}
