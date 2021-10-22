package objx_test

import (
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleExample(t *testing.T) {
	// build a map from a JSON object
	o := objx.MustFromJSON(`{"name":"Mat","foods":["indian","chinese"], "location":{"county":"hobbiton","city":"the shire"}}`)

	// Map can be used as a straight map[string]interface{}
	assert.Equal(t, o["name"], "Mat")

	// Get an Value object
	v := o.Get("name")
	require.NotNil(t, v)

	// Test the contained value
	assert.False(t, v.IsInt())
	assert.False(t, v.IsBool())
	assert.True(t, v.IsStr())

	// Get the contained value
	assert.Equal(t, v.Str(), "Mat")

	// Get a default value if the contained value is not of the expected type or does not exist
	assert.Equal(t, 1, v.Int(1))

	// Get a value by using array notation
	assert.Equal(t, "indian", o.Get("foods[0]").Data())

	// Set a value by using array notation
	o.Set("foods[0]", "italian")
	assert.Equal(t, "italian", o.Get("foods[0]").Str())

	// Get a value by using dot notation
	assert.Equal(t, "hobbiton", o.Get("location.county").Str())
}
