package objx_test

import (
	"strings"
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExclude(t *testing.T) {
	m := objx.Map{
		"name":   "Mat",
		"age":    29,
		"secret": "ABC",
	}

	excluded := m.Exclude([]string{"secret"})

	assert.Equal(t, m["name"], excluded["name"])
	assert.Equal(t, m["age"], excluded["age"])
	assert.False(t, excluded.Has("secret"), "secret should be excluded")
}

func TestCopy(t *testing.T) {
	m1 := objx.Map{
		"name":     "Tyler",
		"location": "UT",
	}

	m2 := m1.Copy()
	require.NotNil(t, m2)
	m2["name"] = "Mat"

	assert.Equal(t, m1.Get("name").Str(), "Tyler")
	assert.Equal(t, m2.Get("name").Str(), "Mat")

}

func TestMerge(t *testing.T) {
	m1 := objx.Map{
		"name": "Mat",
	}
	m2 := objx.Map{
		"name":     "Tyler",
		"location": "UT",
	}

	merged := m1.Merge(m2)

	assert.Equal(t, merged.Get("name").Str(), m2.Get("name").Str())
	assert.Equal(t, merged.Get("location").Str(), m2.Get("location").Str())
	assert.Empty(t, m1.Get("location").Str())
}

func TestMergeHere(t *testing.T) {
	m1 := objx.Map{
		"name": "Mat",
	}
	m2 := objx.Map{
		"name":     "Tyler",
		"location": "UT",
	}

	merged := m1.MergeHere(m2)

	assert.Equal(t, m1, merged, "With MergeHere, it should return the first modified map")
	assert.Equal(t, merged.Get("name").Str(), m2.Get("name").Str())
	assert.Equal(t, merged.Get("location").Str(), m2.Get("location").Str())
	assert.Equal(t, merged.Get("location").Str(), m1.Get("location").Str())
}

func TestTransform(t *testing.T) {
	m := objx.Map{
		"name":     "Mat",
		"location": "UK",
	}
	r := m.Transform(keyToUpper)
	assert.Equal(t, objx.Map{
		"NAME":     "Mat",
		"LOCATION": "UK",
	}, r)
}

func TestTransformKeys(t *testing.T) {
	m := objx.Map{
		"a": "1",
		"b": "2",
		"c": "3",
	}
	mapping := map[string]string{
		"a": "d",
		"b": "e",
	}
	r := m.TransformKeys(mapping)
	assert.Equal(t, objx.Map{
		"c": "3",
		"d": "1",
		"e": "2",
	}, r)
}

func keyToUpper(s string, v interface{}) (string, interface{}) {
	return strings.ToUpper(s), v
}
