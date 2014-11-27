package objx

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

type Convertable struct {
	name string
}

func (c *Convertable) MSI() map[string]interface{} {
	return map[string]interface{}{"name": c.name}
}

type Unconvertable struct {
	name string
}

func TestMapCreation(t *testing.T) {

	o := New(nil)
	assert.Nil(t, o)

	o = New("Tyler")
	assert.Nil(t, o)

	unconvertable := &Unconvertable{name: "Tyler"}
	o = New(unconvertable)
	assert.Nil(t, o)

	convertable := &Convertable{name: "Tyler"}
	o = New(convertable)
	if assert.NotNil(t, convertable) {
		assert.Equal(t, "Tyler", o["name"], "Tyler")
	}

	o = MSI()
	if assert.NotNil(t, o) {
		assert.NotNil(t, o)
	}

	o = MSI("name", "Tyler")
	if assert.NotNil(t, o) {
		if assert.NotNil(t, o) {
			assert.Equal(t, o["name"], "Tyler")
		}
	}

}

func TestMapMustFromJSONWithError(t *testing.T) {

	_, err := FromJSON(`"name":"Mat"}`)
	assert.Error(t, err)

}

func TestMapFromJSON(t *testing.T) {

	o := MustFromJSON(`{"name":"Mat"}`)

	if assert.NotNil(t, o) {
		if assert.NotNil(t, o) {
			assert.Equal(t, "Mat", o["name"])
		}
	}

}

func TestMapFromJSONWithError(t *testing.T) {

	var m Map

	assert.Panics(t, func() {
		m = MustFromJSON(`"name":"Mat"}`)
	})

	assert.Nil(t, m)

}

func TestMapFromBase64String(t *testing.T) {

	base64String := "eyJuYW1lIjoiTWF0In0="

	o, err := FromBase64(base64String)

	if assert.NoError(t, err) {
		assert.Equal(t, o.Get("name").Str(), "Mat")
	}

	assert.Equal(t, MustFromBase64(base64String).Get("name").Str(), "Mat")

}

func TestMapFromBase64StringWithError(t *testing.T) {

	base64String := "eyJuYW1lIjoiTWFasd0In0="

	_, err := FromBase64(base64String)

	assert.Error(t, err)

	assert.Panics(t, func() {
		MustFromBase64(base64String)
	})

}

func TestMapFromSignedBase64String(t *testing.T) {

	base64String := "eyJuYW1lIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6"

	o, err := FromSignedBase64(base64String, "key")

	if assert.NoError(t, err) {
		assert.Equal(t, o.Get("name").Str(), "Mat")
	}

	assert.Equal(t, MustFromSignedBase64(base64String, "key").Get("name").Str(), "Mat")

}

func TestMapFromSignedBase64StringWithError(t *testing.T) {

	base64String := "eyJuYW1lasdIjoiTWF0In0=_67ee82916f90b2c0d68c903266e8998c9ef0c3d6"

	_, err := FromSignedBase64(base64String, "key")

	assert.Error(t, err)

	assert.Panics(t, func() {
		MustFromSignedBase64(base64String, "key")
	})

}

func TestMapFromURLQuery(t *testing.T) {

	m, err := FromURLQuery("name=tyler&state=UT")
	if assert.NoError(t, err) && assert.NotNil(t, m) {
		assert.Equal(t, "tyler", m.Get("name").Str())
		assert.Equal(t, "UT", m.Get("state").Str())
	}

}
