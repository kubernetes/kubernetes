// +build go1.8

package jsoniter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_read_uint64_invalid(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, ",")
	iter.ReadUint64()
	should.NotNil(iter.Error)
}

func Test_read_int_from_null(t *testing.T) {

	type TestObject struct {
		F1  int8
		F2  int16
		F3  int32
		F4  int64
		F5  int
		F6  uint8
		F7  uint16
		F8  uint32
		F9  uint64
		F10 uint
		F11 float32
		F12 float64
		F13 uintptr
	}

	should := require.New(t)
	obj := TestObject{}
	err := Unmarshal([]byte(`{
	"f1":null,
	"f2":null,
	"f3":null,
	"f4":null,
	"f5":null,
	"f6":null,
	"f7":null,
	"f8":null,
	"f9":null,
	"f10":null,
	"f11":null,
	"f12":null,
	"f13":null
	}`), &obj)
	should.Nil(err)
}

func _int8(t *testing.T) {
	inputs := []string{`127`, `-128`}
	for _, input := range inputs {
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			expected, err := strconv.ParseInt(input, 10, 8)
			should.Nil(err)
			should.Equal(int8(expected), iter.ReadInt8())
		})
	}
}

func Test_read_int16(t *testing.T) {
	inputs := []string{`32767`, `-32768`}
	for _, input := range inputs {
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			expected, err := strconv.ParseInt(input, 10, 16)
			should.Nil(err)
			should.Equal(int16(expected), iter.ReadInt16())
		})
	}
}

func Test_read_int32(t *testing.T) {
	inputs := []string{`1`, `12`, `123`, `1234`, `12345`, `123456`, `2147483647`, `-2147483648`}
	for _, input := range inputs {
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			expected, err := strconv.ParseInt(input, 10, 32)
			should.Nil(err)
			should.Equal(int32(expected), iter.ReadInt32())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input), 2)
			expected, err := strconv.ParseInt(input, 10, 32)
			should.Nil(err)
			should.Equal(int32(expected), iter.ReadInt32())
		})
	}
}

func Test_read_int32_array(t *testing.T) {
	should := require.New(t)
	input := `[123,456,789]`
	val := make([]int32, 0)
	UnmarshalFromString(input, &val)
	should.Equal(3, len(val))
}

func Test_read_int64_array(t *testing.T) {
	should := require.New(t)
	input := `[123,456,789]`
	val := make([]int64, 0)
	UnmarshalFromString(input, &val)
	should.Equal(3, len(val))
}

func Test_read_int_overflow(t *testing.T) {
	should := require.New(t)
	inputArr := []string{"123451", "-123451"}
	for _, s := range inputArr {
		iter := ParseString(ConfigDefault, s)
		iter.ReadInt8()
		should.NotNil(iter.Error)

		iterU := ParseString(ConfigDefault, s)
		iterU.ReadUint8()
		should.NotNil(iterU.Error)

	}

	inputArr = []string{"12345678912", "-12345678912"}
	for _, s := range inputArr {
		iter := ParseString(ConfigDefault, s)
		iter.ReadInt16()
		should.NotNil(iter.Error)

		iterUint := ParseString(ConfigDefault, s)
		iterUint.ReadUint16()
		should.NotNil(iterUint.Error)
	}

	inputArr = []string{"3111111111", "-3111111111", "1234232323232323235678912", "-1234567892323232323212"}
	for _, s := range inputArr {
		iter := ParseString(ConfigDefault, s)
		iter.ReadInt32()
		should.NotNil(iter.Error)

		iterUint := ParseString(ConfigDefault, s)
		iterUint.ReadUint32()
		should.NotNil(iterUint.Error)
	}

	inputArr = []string{"9223372036854775811", "-9523372036854775807", "1234232323232323235678912", "-1234567892323232323212"}
	for _, s := range inputArr {
		iter := ParseString(ConfigDefault, s)
		iter.ReadInt64()
		should.NotNil(iter.Error)

		iterUint := ParseString(ConfigDefault, s)
		iterUint.ReadUint64()
		should.NotNil(iterUint.Error)
	}
}

func Test_read_int64(t *testing.T) {
	inputs := []string{`1`, `12`, `123`, `1234`, `12345`, `123456`, `9223372036854775807`, `-9223372036854775808`}
	for _, input := range inputs {
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			expected, err := strconv.ParseInt(input, 10, 64)
			should.Nil(err)
			should.Equal(expected, iter.ReadInt64())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input), 2)
			expected, err := strconv.ParseInt(input, 10, 64)
			should.Nil(err)
			should.Equal(expected, iter.ReadInt64())
		})
	}
}

func Test_read_int64_overflow(t *testing.T) {
	should := require.New(t)
	input := "123456789123456789123456789123456789,"
	iter := ParseString(ConfigDefault, input)
	iter.ReadInt64()
	should.NotNil(iter.Error)
}

func Test_wrap_int(t *testing.T) {
	should := require.New(t)
	str, err := MarshalToString(WrapInt64(100))
	should.Nil(err)
	should.Equal("100", str)
}

func Test_write_uint8(t *testing.T) {
	vals := []uint8{0, 1, 11, 111, 255}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteUint8(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 3)
	stream.WriteRaw("a")
	stream.WriteUint8(100) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a100", buf.String())
}

func Test_write_int8(t *testing.T) {
	vals := []int8{0, 1, -1, 99, 0x7f, -0x80}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteInt8(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 4)
	stream.WriteRaw("a")
	stream.WriteInt8(-100) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a-100", buf.String())
}

func Test_write_uint16(t *testing.T) {
	vals := []uint16{0, 1, 11, 111, 255, 0xfff, 0xffff}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteUint16(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 5)
	stream.WriteRaw("a")
	stream.WriteUint16(10000) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a10000", buf.String())
}

func Test_write_int16(t *testing.T) {
	vals := []int16{0, 1, 11, 111, 255, 0xfff, 0x7fff, -0x8000}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteInt16(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 6)
	stream.WriteRaw("a")
	stream.WriteInt16(-10000) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a-10000", buf.String())
}

func Test_write_uint32(t *testing.T) {
	vals := []uint32{0, 1, 11, 111, 255, 999999, 0xfff, 0xffff, 0xfffff, 0xffffff, 0xfffffff, 0xffffffff}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteUint32(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 10)
	stream.WriteRaw("a")
	stream.WriteUint32(0xffffffff) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a4294967295", buf.String())
}

func Test_write_int32(t *testing.T) {
	vals := []int32{0, 1, 11, 111, 255, 999999, 0xfff, 0xffff, 0xfffff, 0xffffff, 0xfffffff, 0x7fffffff, -0x80000000}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteInt32(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(int64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 11)
	stream.WriteRaw("a")
	stream.WriteInt32(-0x7fffffff) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a-2147483647", buf.String())
}

func Test_write_uint64(t *testing.T) {
	vals := []uint64{0, 1, 11, 111, 255, 999999, 0xfff, 0xffff, 0xfffff, 0xffffff, 0xfffffff, 0xffffffff,
		0xfffffffff, 0xffffffffff, 0xfffffffffff, 0xffffffffffff, 0xfffffffffffff, 0xffffffffffffff,
		0xfffffffffffffff, 0xffffffffffffffff}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteUint64(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatUint(uint64(val), 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 10)
	stream.WriteRaw("a")
	stream.WriteUint64(0xffffffff) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a4294967295", buf.String())
}

func Test_write_int64(t *testing.T) {
	vals := []int64{0, 1, 11, 111, 255, 999999, 0xfff, 0xffff, 0xfffff, 0xffffff, 0xfffffff, 0xffffffff,
		0xfffffffff, 0xffffffffff, 0xfffffffffff, 0xffffffffffff, 0xfffffffffffff, 0xffffffffffffff,
		0xfffffffffffffff, 0x7fffffffffffffff, -0x8000000000000000}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteInt64(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(val, 10), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatInt(val, 10), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 10)
	stream.WriteRaw("a")
	stream.WriteInt64(0xffffffff) // should clear buffer
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("a4294967295", buf.String())
}

func Test_write_val_int(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 4096)
	stream.WriteVal(1001)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("1001", buf.String())
}

func Test_write_val_int_ptr(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 4096)
	val := 1001
	stream.WriteVal(&val)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("1001", buf.String())
}

func Test_json_number(t *testing.T) {
	should := require.New(t)
	var arr []json.Number
	err := Unmarshal([]byte(`[1]`), &arr)
	should.Nil(err)
	should.Equal(json.Number("1"), arr[0])
	str, err := MarshalToString(arr)
	should.Nil(err)
	should.Equal(`[1]`, str)
}

func Test_jsoniter_number(t *testing.T) {
	should := require.New(t)
	var arr []Number
	err := Unmarshal([]byte(`[1]`), &arr)
	should.Nil(err)
	should.Equal(Number("1"), arr[0])
	str, isNumber := CastJsonNumber(arr[0])
	should.True(isNumber)
	should.Equal("1", str)
}

func Test_non_numeric_as_number(t *testing.T) {
	should := require.New(t)
	var v1 json.Number
	err := Unmarshal([]byte(`"500"`), &v1)
	should.Nil(err)
	should.Equal("500", string(v1))
	var v2 Number
	err = Unmarshal([]byte(`"500"`), &v2)
	should.Nil(err)
	should.Equal("500", string(v2))
}

func Test_null_as_number(t *testing.T) {
	should := require.New(t)
	var v1 json.Number
	err := json.Unmarshal([]byte(`null`), &v1)
	should.Nil(err)
	should.Equal("", string(v1))
	var v2 Number
	err = Unmarshal([]byte(`null`), &v2)
	should.Nil(err)
	should.Equal("", string(v2))
}

func Test_float_as_int(t *testing.T) {
	should := require.New(t)
	var i int
	should.NotNil(Unmarshal([]byte(`1.1`), &i))
}

func Benchmark_jsoniter_encode_int(b *testing.B) {
	stream := NewStream(ConfigDefault, ioutil.Discard, 64)
	for n := 0; n < b.N; n++ {
		stream.n = 0
		stream.WriteUint64(0xffffffff)
	}
}

func Benchmark_itoa(b *testing.B) {
	for n := 0; n < b.N; n++ {
		strconv.FormatInt(0xffffffff, 10)
	}
}

func Benchmark_jsoniter_int(b *testing.B) {
	iter := NewIterator(ConfigDefault)
	input := []byte(`100`)
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(input)
		iter.ReadInt64()
	}
}

func Benchmark_json_int(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := int64(0)
		json.Unmarshal([]byte(`-100`), &result)
	}
}
