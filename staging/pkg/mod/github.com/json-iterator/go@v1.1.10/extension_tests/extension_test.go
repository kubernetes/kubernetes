package test

import (
	"github.com/json-iterator/go"
	"github.com/modern-go/reflect2"
	"github.com/stretchr/testify/require"
	"reflect"
	"strconv"
	"testing"
	"unsafe"
)

type TestObject1 struct {
	Field1 string
}

type testExtension struct {
	jsoniter.DummyExtension
}

func (extension *testExtension) UpdateStructDescriptor(structDescriptor *jsoniter.StructDescriptor) {
	if structDescriptor.Type.String() != "test.TestObject1" {
		return
	}
	binding := structDescriptor.GetField("Field1")
	binding.Encoder = &funcEncoder{fun: func(ptr unsafe.Pointer, stream *jsoniter.Stream) {
		str := *((*string)(ptr))
		val, _ := strconv.Atoi(str)
		stream.WriteInt(val)
	}}
	binding.Decoder = &funcDecoder{func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
		*((*string)(ptr)) = strconv.Itoa(iter.ReadInt())
	}}
	binding.ToNames = []string{"field-1"}
	binding.FromNames = []string{"field-1"}
}

func Test_customize_field_by_extension(t *testing.T) {
	should := require.New(t)
	cfg := jsoniter.Config{}.Froze()
	cfg.RegisterExtension(&testExtension{})
	obj := TestObject1{}
	err := cfg.UnmarshalFromString(`{"field-1": 100}`, &obj)
	should.Nil(err)
	should.Equal("100", obj.Field1)
	str, err := cfg.MarshalToString(obj)
	should.Nil(err)
	should.Equal(`{"field-1":100}`, str)
}

func Test_customize_map_key_encoder(t *testing.T) {
	should := require.New(t)
	cfg := jsoniter.Config{}.Froze()
	cfg.RegisterExtension(&testMapKeyExtension{})
	m := map[int]int{1: 2}
	output, err := cfg.MarshalToString(m)
	should.NoError(err)
	should.Equal(`{"2":2}`, output)
	m = map[int]int{}
	should.NoError(cfg.UnmarshalFromString(output, &m))
	should.Equal(map[int]int{1: 2}, m)
}

type testMapKeyExtension struct {
	jsoniter.DummyExtension
}

func (extension *testMapKeyExtension) CreateMapKeyEncoder(typ reflect2.Type) jsoniter.ValEncoder {
	if typ.Kind() == reflect.Int {
		return &funcEncoder{
			fun: func(ptr unsafe.Pointer, stream *jsoniter.Stream) {
				stream.WriteRaw(`"`)
				stream.WriteInt(*(*int)(ptr) + 1)
				stream.WriteRaw(`"`)
			},
		}
	}
	return nil
}

func (extension *testMapKeyExtension) CreateMapKeyDecoder(typ reflect2.Type) jsoniter.ValDecoder {
	if typ.Kind() == reflect.Int {
		return &funcDecoder{
			fun: func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
				i, err := strconv.Atoi(iter.ReadString())
				if err != nil {
					iter.ReportError("read map key", err.Error())
					return
				}
				i--
				*(*int)(ptr) = i
			},
		}
	}
	return nil
}

type funcDecoder struct {
	fun jsoniter.DecoderFunc
}

func (decoder *funcDecoder) Decode(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
	decoder.fun(ptr, iter)
}

type funcEncoder struct {
	fun         jsoniter.EncoderFunc
	isEmptyFunc func(ptr unsafe.Pointer) bool
}

func (encoder *funcEncoder) Encode(ptr unsafe.Pointer, stream *jsoniter.Stream) {
	encoder.fun(ptr, stream)
}

func (encoder *funcEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	if encoder.isEmptyFunc == nil {
		return false
	}
	return encoder.isEmptyFunc(ptr)
}
