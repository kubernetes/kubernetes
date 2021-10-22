package objx_test

import (
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var TestMap = objx.Map{
	"name": "Tyler",
	"address": objx.Map{
		"city":  "Salt Lake City",
		"state": "UT",
	},
	"numbers": []interface{}{"one", "two", "three", "four", "five"},
}

type Convertable struct {
	name string
}

type Unconvertable struct {
	name string
}

func (c *Convertable) MSI() map[string]interface{} {
	return objx.Map{"name": c.name}
}

func TestMapCreation(t *testing.T) {
	o := objx.New(nil)
	assert.Nil(t, o)

	o = objx.New("Tyler")
	assert.Nil(t, o)

	unconvertable := &Unconvertable{name: "Tyler"}
	o = objx.New(unconvertable)
	assert.Nil(t, o)

	convertable := &Convertable{name: "Tyler"}
	o = objx.New(convertable)
	require.NotNil(t, convertable)
	assert.Equal(t, "Tyler", o["name"])

	o = objx.MSI()
	assert.NotNil(t, o)

	o = objx.MSI("name", "Tyler")
	require.NotNil(t, o)
	assert.Equal(t, o["name"], "Tyler")

	o = objx.MSI(1, "a")
	assert.Nil(t, o)

	o = objx.MSI("a")
	assert.Nil(t, o)

	o = objx.MSI("a", "b", "c")
	assert.Nil(t, o)
}

func TestMapValure(t *testing.T) {
	m := objx.Map{
		"a": 1,
	}
	v := m.Value()

	assert.Equal(t, m, v.ObjxMap())
}

func TestMapMustFromJSONWithError(t *testing.T) {
	_, err := objx.FromJSON(`"name":"Mat"}`)
	assert.Error(t, err)
}

func TestMapFromJSON(t *testing.T) {
	o := objx.MustFromJSON(`{"name":"Mat"}`)

	require.NotNil(t, o)
	assert.Equal(t, "Mat", o["name"])
}

func TestMapFromJSONWithError(t *testing.T) {
	var m objx.Map

	assert.Panics(t, func() {
		m = objx.MustFromJSON(`"name":"Mat"}`)
	})
	assert.Nil(t, m)
}

func TestConversionJSONInt(t *testing.T) {
	jsonString :=
		`{
    "a": 1,
    "b": {
      "data": 1
    },
    "c": [1],
    "d": [[1]]
  }`
	m, err := objx.FromJSON(jsonString)

	assert.Nil(t, err)
	require.NotNil(t, m)
	assert.Equal(t, 1, m.Get("a").Int())
	assert.Equal(t, 1, m.Get("b.data").Int())

	assert.True(t, m.Get("c").IsInterSlice())
	assert.Equal(t, 1, m.Get("c").InterSlice()[0])

	assert.True(t, m.Get("d").IsInterSlice())
	assert.Equal(t, []interface{}{1}, m.Get("d").InterSlice()[0])
}

func TestJSONSliceInt(t *testing.T) {
	jsonString :=
		`{
      "a": [
        {"b": 1},
        {"c": 2}
      ]
    }`
	m, err := objx.FromJSON(jsonString)

	assert.Nil(t, err)
	require.NotNil(t, m)
	assert.Equal(t, []objx.Map{{"b": 1}, {"c": 2}}, m.Get("a").ObjxMapSlice())
}

func TestJSONSliceMixed(t *testing.T) {
	jsonString :=
		`{
      "a": [
        {"b": 1},
        "a"
      ]
    }`
	m, err := objx.FromJSON(jsonString)

	assert.Nil(t, err)
	require.NotNil(t, m)

	assert.Nil(t, m.Get("a").ObjxMapSlice())
}

func TestMapFromBase64String(t *testing.T) {
	base64String := "eyJuYW1lIjoiTWF0In0="
	o, err := objx.FromBase64(base64String)

	require.NoError(t, err)
	assert.Equal(t, o.Get("name").Str(), "Mat")
	assert.Equal(t, objx.MustFromBase64(base64String).Get("name").Str(), "Mat")
}

func TestMapFromBase64StringWithError(t *testing.T) {
	base64String := "eyJuYW1lIjoiTWFasd0In0="
	_, err := objx.FromBase64(base64String)

	assert.Error(t, err)
	assert.Panics(t, func() {
		objx.MustFromBase64(base64String)
	})
}

func TestMapFromSignedBase64String(t *testing.T) {
	base64String := "eyJuYW1lIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6"

	o, err := objx.FromSignedBase64(base64String, "key")

	require.NoError(t, err)
	assert.Equal(t, o.Get("name").Str(), "Mat")
	assert.Equal(t, objx.MustFromSignedBase64(base64String, "key").Get("name").Str(), "Mat")
}

func TestMapFromSignedBase64StringWithError(t *testing.T) {
	base64String := "eyJuYW1lasdIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6"
	_, err := objx.FromSignedBase64(base64String, "key")
	assert.Error(t, err)
	assert.Panics(t, func() {
		objx.MustFromSignedBase64(base64String, "key")
	})

	base64String = "eyJuYW1lasdIjoiTWF0In0=67ee82916f90b2c0d68c903266e8998c9ef0c3d6"
	_, err = objx.FromSignedBase64(base64String, "key")
	assert.Error(t, err)
	assert.Panics(t, func() {
		objx.MustFromSignedBase64(base64String, "key")
	})

	base64String = "eyJuYW1lIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6_junk"
	_, err = objx.FromSignedBase64(base64String, "key")
	assert.Error(t, err)
	assert.Panics(t, func() {
		objx.MustFromSignedBase64(base64String, "key")
	})
}

func TestMapFromURLQuery(t *testing.T) {
	m, err := objx.FromURLQuery("name=tyler&state=UT")

	assert.NoError(t, err)
	require.NotNil(t, m)
	assert.Equal(t, "tyler", m.Get("name").Str())
	assert.Equal(t, "UT", m.Get("state").Str())
}

func TestMapMustFromURLQuery(t *testing.T) {
	m := objx.MustFromURLQuery("name=tyler&state=UT")

	require.NotNil(t, m)
	assert.Equal(t, "tyler", m.Get("name").Str())
	assert.Equal(t, "UT", m.Get("state").Str())
}

func TestMapFromURLQueryWithError(t *testing.T) {
	m, err := objx.FromURLQuery("%")

	assert.Error(t, err)
	assert.Nil(t, m)
	assert.Panics(t, func() {
		objx.MustFromURLQuery("%")
	})
}
