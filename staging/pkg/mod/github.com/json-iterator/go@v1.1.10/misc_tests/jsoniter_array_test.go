package misc_tests

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

func Test_empty_array(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[]`)
	cont := iter.ReadArray()
	should.False(cont)
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `[]`)
	iter.ReadArrayCB(func(iter *jsoniter.Iterator) bool {
		should.FailNow("should not call")
		return true
	})
}

func Test_one_element(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[1]`)
	should.True(iter.ReadArray())
	should.Equal(1, iter.ReadInt())
	should.False(iter.ReadArray())
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `[1]`)
	iter.ReadArrayCB(func(iter *jsoniter.Iterator) bool {
		should.Equal(1, iter.ReadInt())
		return true
	})
}

func Test_two_elements(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[1,2]`)
	should.True(iter.ReadArray())
	should.Equal(int64(1), iter.ReadInt64())
	should.True(iter.ReadArray())
	should.Equal(int64(2), iter.ReadInt64())
	should.False(iter.ReadArray())
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `[1,2]`)
	should.Equal([]interface{}{float64(1), float64(2)}, iter.Read())
}

func Test_whitespace_in_head(t *testing.T) {
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, ` [1]`)
	cont := iter.ReadArray()
	if cont != true {
		t.FailNow()
	}
	if iter.ReadUint64() != 1 {
		t.FailNow()
	}
}

func Test_whitespace_after_array_start(t *testing.T) {
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[ 1]`)
	cont := iter.ReadArray()
	if cont != true {
		t.FailNow()
	}
	if iter.ReadUint64() != 1 {
		t.FailNow()
	}
}

func Test_whitespace_before_array_end(t *testing.T) {
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[1 ]`)
	cont := iter.ReadArray()
	if cont != true {
		t.FailNow()
	}
	if iter.ReadUint64() != 1 {
		t.FailNow()
	}
	cont = iter.ReadArray()
	if cont != false {
		t.FailNow()
	}
}

func Test_whitespace_before_comma(t *testing.T) {
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `[1 ,2]`)
	cont := iter.ReadArray()
	if cont != true {
		t.FailNow()
	}
	if iter.ReadUint64() != 1 {
		t.FailNow()
	}
	cont = iter.ReadArray()
	if cont != true {
		t.FailNow()
	}
	if iter.ReadUint64() != 2 {
		t.FailNow()
	}
	cont = iter.ReadArray()
	if cont != false {
		t.FailNow()
	}
}

func Test_write_array(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.Config{IndentionStep: 2}.Froze(), buf, 4096)
	stream.WriteArrayStart()
	stream.WriteInt(1)
	stream.WriteMore()
	stream.WriteInt(2)
	stream.WriteArrayEnd()
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("[\n  1,\n  2\n]", buf.String())
}

func Test_write_val_array(t *testing.T) {
	should := require.New(t)
	val := []int{1, 2, 3}
	str, err := jsoniter.MarshalToString(&val)
	should.Nil(err)
	should.Equal("[1,2,3]", str)
}

func Test_write_val_empty_array(t *testing.T) {
	should := require.New(t)
	val := []int{}
	str, err := jsoniter.MarshalToString(val)
	should.Nil(err)
	should.Equal("[]", str)
}

func Test_write_array_of_interface_in_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field  []interface{}
		Field2 string
	}
	val := TestObject{[]interface{}{1, 2}, ""}
	str, err := jsoniter.MarshalToString(val)
	should.Nil(err)
	should.Contains(str, `"Field":[1,2]`)
	should.Contains(str, `"Field2":""`)
}

func Test_encode_byte_array(t *testing.T) {
	should := require.New(t)
	bytes, err := json.Marshal([]byte{1, 2, 3})
	should.Nil(err)
	should.Equal(`"AQID"`, string(bytes))
	bytes, err = jsoniter.Marshal([]byte{1, 2, 3})
	should.Nil(err)
	should.Equal(`"AQID"`, string(bytes))
}

func Test_encode_empty_byte_array(t *testing.T) {
	should := require.New(t)
	bytes, err := json.Marshal([]byte{})
	should.Nil(err)
	should.Equal(`""`, string(bytes))
	bytes, err = jsoniter.Marshal([]byte{})
	should.Nil(err)
	should.Equal(`""`, string(bytes))
}

func Test_encode_nil_byte_array(t *testing.T) {
	should := require.New(t)
	var nilSlice []byte
	bytes, err := json.Marshal(nilSlice)
	should.Nil(err)
	should.Equal(`null`, string(bytes))
	bytes, err = jsoniter.Marshal(nilSlice)
	should.Nil(err)
	should.Equal(`null`, string(bytes))
}

func Test_decode_byte_array_from_base64(t *testing.T) {
	should := require.New(t)
	data := []byte{}
	err := json.Unmarshal([]byte(`"AQID"`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
	err = jsoniter.Unmarshal([]byte(`"AQID"`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
}

func Test_decode_byte_array_from_base64_with_newlines(t *testing.T) {
	should := require.New(t)
	data := []byte{}
	err := json.Unmarshal([]byte(`"A\rQ\nID"`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
	err = jsoniter.Unmarshal([]byte(`"A\rQ\nID"`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
}

func Test_decode_byte_array_from_array(t *testing.T) {
	should := require.New(t)
	data := []byte{}
	err := json.Unmarshal([]byte(`[1,2,3]`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
	err = jsoniter.Unmarshal([]byte(`[1,2,3]`), &data)
	should.Nil(err)
	should.Equal([]byte{1, 2, 3}, data)
}

func Test_decode_slice(t *testing.T) {
	should := require.New(t)
	slice := make([]string, 0, 5)
	jsoniter.UnmarshalFromString(`["hello", "world"]`, &slice)
	should.Equal([]string{"hello", "world"}, slice)
}

func Test_decode_large_slice(t *testing.T) {
	should := require.New(t)
	slice := make([]int, 0, 1)
	jsoniter.UnmarshalFromString(`[1,2,3,4,5,6,7,8,9]`, &slice)
	should.Equal([]int{1, 2, 3, 4, 5, 6, 7, 8, 9}, slice)
}

func Benchmark_jsoniter_array(b *testing.B) {
	b.ReportAllocs()
	input := []byte(`[1,2,3,4,5,6,7,8,9]`)
	iter := jsoniter.ParseBytes(jsoniter.ConfigDefault, input)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(input)
		for iter.ReadArray() {
			iter.ReadUint64()
		}
	}
}

func Benchmark_json_array(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := []interface{}{}
		json.Unmarshal([]byte(`[1,2,3]`), &result)
	}
}
