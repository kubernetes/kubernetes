package jsoniter

import (
	"bytes"
	"encoding/json"
	"github.com/stretchr/testify/require"
	"io/ioutil"
	"testing"
)

func Test_new_decoder(t *testing.T) {
	should := require.New(t)
	decoder1 := json.NewDecoder(bytes.NewBufferString(`[1][2]`))
	decoder2 := NewDecoder(bytes.NewBufferString(`[1][2]`))
	arr1 := []int{}
	should.Nil(decoder1.Decode(&arr1))
	should.Equal([]int{1}, arr1)
	arr2 := []int{}
	should.True(decoder1.More())
	buffered, _ := ioutil.ReadAll(decoder1.Buffered())
	should.Equal("[2]", string(buffered))
	should.Nil(decoder2.Decode(&arr2))
	should.Equal([]int{1}, arr2)
	should.True(decoder2.More())
	buffered, _ = ioutil.ReadAll(decoder2.Buffered())
	should.Equal("[2]", string(buffered))

	should.Nil(decoder1.Decode(&arr1))
	should.Equal([]int{2}, arr1)
	should.False(decoder1.More())
	should.Nil(decoder2.Decode(&arr2))
	should.Equal([]int{2}, arr2)
	should.False(decoder2.More())
}

func Test_use_number(t *testing.T) {
	should := require.New(t)
	decoder1 := json.NewDecoder(bytes.NewBufferString(`123`))
	decoder1.UseNumber()
	decoder2 := NewDecoder(bytes.NewBufferString(`123`))
	decoder2.UseNumber()
	var obj1 interface{}
	should.Nil(decoder1.Decode(&obj1))
	should.Equal(json.Number("123"), obj1)
	var obj2 interface{}
	should.Nil(decoder2.Decode(&obj2))
	should.Equal(json.Number("123"), obj2)
}

func Test_use_number_for_unmarshal(t *testing.T) {
	should := require.New(t)
	api := Config{UseNumber: true}.Froze()
	var obj interface{}
	should.Nil(api.UnmarshalFromString("123", &obj))
	should.Equal(json.Number("123"), obj)
}

func Test_marshal_indent(t *testing.T) {
	should := require.New(t)
	obj := struct {
		F1 int
		F2 []int
	}{1, []int{2, 3, 4}}
	output, err := json.MarshalIndent(obj, "", "  ")
	should.Nil(err)
	should.Equal("{\n  \"F1\": 1,\n  \"F2\": [\n    2,\n    3,\n    4\n  ]\n}", string(output))
	output, err = MarshalIndent(obj, "", "  ")
	should.Nil(err)
	should.Equal("{\n  \"F1\": 1,\n  \"F2\": [\n    2,\n    3,\n    4\n  ]\n}", string(output))
}

func Test_marshal_indent_map(t *testing.T) {
	should := require.New(t)
	obj := map[int]int{1: 2}
	output, err := json.MarshalIndent(obj, "", "  ")
	should.Nil(err)
	should.Equal("{\n  \"1\": 2\n}", string(output))
	output, err = MarshalIndent(obj, "", "  ")
	should.Nil(err)
	should.Equal("{\n  \"1\": 2\n}", string(output))
	output, err = ConfigCompatibleWithStandardLibrary.MarshalIndent(obj, "", "  ")
	should.Nil(err)
	should.Equal("{\n  \"1\": 2\n}", string(output))
}
