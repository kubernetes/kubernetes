package objx

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestExclude(t *testing.T) {

	d := make(Map)
	d["name"] = "Mat"
	d["age"] = 29
	d["secret"] = "ABC"

	excluded := d.Exclude([]string{"secret"})

	assert.Equal(t, d["name"], excluded["name"])
	assert.Equal(t, d["age"], excluded["age"])
	assert.False(t, excluded.Has("secret"), "secret should be excluded")

}

func TestCopy(t *testing.T) {

	d1 := make(map[string]interface{})
	d1["name"] = "Tyler"
	d1["location"] = "UT"

	d1Obj := New(d1)
	d2Obj := d1Obj.Copy()

	d2Obj["name"] = "Mat"

	assert.Equal(t, d1Obj.Get("name").Str(), "Tyler")
	assert.Equal(t, d2Obj.Get("name").Str(), "Mat")

}

func TestMerge(t *testing.T) {

	d := make(map[string]interface{})
	d["name"] = "Mat"

	d1 := make(map[string]interface{})
	d1["name"] = "Tyler"
	d1["location"] = "UT"

	dObj := New(d)
	d1Obj := New(d1)

	merged := dObj.Merge(d1Obj)

	assert.Equal(t, merged.Get("name").Str(), d1Obj.Get("name").Str())
	assert.Equal(t, merged.Get("location").Str(), d1Obj.Get("location").Str())
	assert.Empty(t, dObj.Get("location").Str())

}

func TestMergeHere(t *testing.T) {

	d := make(map[string]interface{})
	d["name"] = "Mat"

	d1 := make(map[string]interface{})
	d1["name"] = "Tyler"
	d1["location"] = "UT"

	dObj := New(d)
	d1Obj := New(d1)

	merged := dObj.MergeHere(d1Obj)

	assert.Equal(t, dObj, merged, "With MergeHere, it should return the first modified map")
	assert.Equal(t, merged.Get("name").Str(), d1Obj.Get("name").Str())
	assert.Equal(t, merged.Get("location").Str(), d1Obj.Get("location").Str())
	assert.Equal(t, merged.Get("location").Str(), dObj.Get("location").Str())
}
