package jmespath

import (
	"testing"

	"github.com/jmespath/go-jmespath/internal/testify/assert"
)

func TestSlicePositiveStep(t *testing.T) {
	assert := assert.New(t)
	input := make([]interface{}, 5)
	input[0] = 0
	input[1] = 1
	input[2] = 2
	input[3] = 3
	input[4] = 4
	result, err := slice(input, []sliceParam{{0, true}, {3, true}, {1, true}})
	assert.Nil(err)
	assert.Equal(input[:3], result)
}

func TestIsFalseJSONTypes(t *testing.T) {
	assert := assert.New(t)
	assert.True(isFalse(false))
	assert.True(isFalse(""))
	var empty []interface{}
	assert.True(isFalse(empty))
	m := make(map[string]interface{})
	assert.True(isFalse(m))
	assert.True(isFalse(nil))

}

func TestIsFalseWithUserDefinedStructs(t *testing.T) {
	assert := assert.New(t)
	type nilStructType struct {
		SliceOfPointers []*string
	}
	nilStruct := nilStructType{SliceOfPointers: nil}
	assert.True(isFalse(nilStruct.SliceOfPointers))

	// A user defined struct will never be false though,
	// even if it's fields are the zero type.
	assert.False(isFalse(nilStruct))
}

func TestIsFalseWithNilInterface(t *testing.T) {
	assert := assert.New(t)
	var a *int
	var nilInterface interface{}
	nilInterface = a
	assert.True(isFalse(nilInterface))
}

func TestIsFalseWithMapOfUserStructs(t *testing.T) {
	assert := assert.New(t)
	type foo struct {
		Bar string
		Baz string
	}
	m := make(map[int]foo)
	assert.True(isFalse(m))
}

func TestObjsEqual(t *testing.T) {
	assert := assert.New(t)
	assert.True(objsEqual("foo", "foo"))
	assert.True(objsEqual(20, 20))
	assert.True(objsEqual([]int{1, 2, 3}, []int{1, 2, 3}))
	assert.True(objsEqual(nil, nil))
	assert.True(!objsEqual(nil, "foo"))
	assert.True(objsEqual([]int{}, []int{}))
	assert.True(!objsEqual([]int{}, nil))
}
