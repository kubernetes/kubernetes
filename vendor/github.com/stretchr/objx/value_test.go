package objx

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestStringTypeString(t *testing.T) {
	m := New(map[string]interface{}{"string": "foo"})
	assert.Equal(t, "foo", m.Get("string").String())
}

func TestStringTypeBool(t *testing.T) {
	m := New(map[string]interface{}{"bool": true})
	assert.Equal(t, "true", m.Get("bool").String())
}

func TestStringTypeInt(t *testing.T) {
	m := New(map[string]interface{}{
		"int":   int(1),
		"int8":  int8(8),
		"int16": int16(16),
		"int32": int32(32),
		"int64": int64(64),
	})

	assert.Equal(t, "1", m.Get("int").String())
	assert.Equal(t, "8", m.Get("int8").String())
	assert.Equal(t, "16", m.Get("int16").String())
	assert.Equal(t, "32", m.Get("int32").String())
	assert.Equal(t, "64", m.Get("int64").String())
}

func TestStringTypeUint(t *testing.T) {
	m := New(map[string]interface{}{
		"uint":   uint(1),
		"uint8":  uint8(8),
		"uint16": uint16(16),
		"uint32": uint32(32),
		"uint64": uint64(64),
	})

	assert.Equal(t, "1", m.Get("uint").String())
	assert.Equal(t, "8", m.Get("uint8").String())
	assert.Equal(t, "16", m.Get("uint16").String())
	assert.Equal(t, "32", m.Get("uint32").String())
	assert.Equal(t, "64", m.Get("uint64").String())
}

func TestStringTypeFloat(t *testing.T) {
	m := New(map[string]interface{}{
		"float32": float32(32.32),
		"float64": float64(64.64),
	})

	assert.Equal(t, "32.32", m.Get("float32").String())
	assert.Equal(t, "64.64", m.Get("float64").String())
}

func TestStringTypeOther(t *testing.T) {
	m := New(map[string]interface{}{
		"other": []string{"foo", "bar"},
	})

	assert.Equal(t, "[]string{\"foo\", \"bar\"}", m.Get("other").String())
}
