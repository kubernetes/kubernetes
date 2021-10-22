package objx_test

import (
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
)

func TestStringTypeString(t *testing.T) {
	m := objx.Map{
		"string": "foo",
	}

	assert.Equal(t, "foo", m.Get("string").String())
}

func TestStringTypeBool(t *testing.T) {
	m := objx.Map{
		"bool": true,
	}

	assert.Equal(t, "true", m.Get("bool").String())
}

func TestStringTypeInt(t *testing.T) {
	m := objx.Map{
		"int":   int(1),
		"int8":  int8(8),
		"int16": int16(16),
		"int32": int32(32),
		"int64": int64(64),
	}

	assert.Equal(t, "1", m.Get("int").String())
	assert.Equal(t, "8", m.Get("int8").String())
	assert.Equal(t, "16", m.Get("int16").String())
	assert.Equal(t, "32", m.Get("int32").String())
	assert.Equal(t, "64", m.Get("int64").String())
}

func TestStringTypeUint(t *testing.T) {
	m := objx.Map{
		"uint":   uint(1),
		"uint8":  uint8(8),
		"uint16": uint16(16),
		"uint32": uint32(32),
		"uint64": uint64(64),
	}

	assert.Equal(t, "1", m.Get("uint").String())
	assert.Equal(t, "8", m.Get("uint8").String())
	assert.Equal(t, "16", m.Get("uint16").String())
	assert.Equal(t, "32", m.Get("uint32").String())
	assert.Equal(t, "64", m.Get("uint64").String())
}

func TestStringTypeFloat(t *testing.T) {
	m := objx.Map{
		"float32": float32(32.32),
		"float64": float64(64.64),
	}

	assert.Equal(t, "32.32", m.Get("float32").String())
	assert.Equal(t, "64.64", m.Get("float64").String())
}

func TestStringTypeOther(t *testing.T) {
	m := objx.Map{
		"other":    []string{"foo", "bar"},
		"nilValue": nil,
	}

	assert.Equal(t, "[]string{\"foo\", \"bar\"}", m.Get("other").String())
	assert.Equal(t, "", m.Get("nilValue").String())
}

func TestStringSliceTypeString(t *testing.T) {
	m := objx.Map{
		"string": []string{"foo", "bar"},
	}

	assert.Equal(t, []string{"foo", "bar"}, m.Get("string").StringSlice())
}

func TestStringSliceTypeBool(t *testing.T) {
	m := objx.Map{
		"bool": []bool{true, false},
	}

	assert.Equal(t, []string{"true", "false"}, m.Get("bool").StringSlice())
}

func TestStringSliceTypeInt(t *testing.T) {
	m := objx.Map{
		"int":   []int{1, 2},
		"int8":  []int8{8, 9},
		"int16": []int16{16, 17},
		"int32": []int32{32, 33},
		"int64": []int64{64, 65},
	}

	assert.Equal(t, []string{"1", "2"}, m.Get("int").StringSlice())
	assert.Equal(t, []string{"8", "9"}, m.Get("int8").StringSlice())
	assert.Equal(t, []string{"16", "17"}, m.Get("int16").StringSlice())
	assert.Equal(t, []string{"32", "33"}, m.Get("int32").StringSlice())
	assert.Equal(t, []string{"64", "65"}, m.Get("int64").StringSlice())
}

func TestStringSliceTypeUint(t *testing.T) {
	m := objx.Map{
		"uint":   []uint{1, 2},
		"uint8":  []uint8{8, 9},
		"uint16": []uint16{16, 17},
		"uint32": []uint32{32, 33},
		"uint64": []uint64{64, 65},
	}

	assert.Equal(t, []string{"1", "2"}, m.Get("uint").StringSlice())
	assert.Equal(t, []string{"8", "9"}, m.Get("uint8").StringSlice())
	assert.Equal(t, []string{"16", "17"}, m.Get("uint16").StringSlice())
	assert.Equal(t, []string{"32", "33"}, m.Get("uint32").StringSlice())
	assert.Equal(t, []string{"64", "65"}, m.Get("uint64").StringSlice())
}

func TestStringSliceTypeFloat(t *testing.T) {
	m := objx.Map{
		"float32": []float32{32.32, 33.33},
		"float64": []float64{64.64, 65.65},
	}

	assert.Equal(t, []string{"32.32", "33.33"}, m.Get("float32").StringSlice())
	assert.Equal(t, []string{"64.64", "65.65"}, m.Get("float64").StringSlice())
}

func TestStringSliceTypeOther(t *testing.T) {
	m := objx.Map{
		"other": "foo",
	}

	assert.Equal(t, []string{}, m.Get("other").StringSlice())
	assert.Equal(t, []string{"bar"}, m.Get("other").StringSlice([]string{"bar"}))
}
