package objx

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAccessorsAccessGetSingleField(t *testing.T) {

	current := map[string]interface{}{"name": "Tyler"}
	assert.Equal(t, "Tyler", access(current, "name", nil, false, true))

}
func TestAccessorsAccessGetDeep(t *testing.T) {

	current := map[string]interface{}{"name": map[string]interface{}{"first": "Tyler", "last": "Bunnell"}}
	assert.Equal(t, "Tyler", access(current, "name.first", nil, false, true))
	assert.Equal(t, "Bunnell", access(current, "name.last", nil, false, true))

}
func TestAccessorsAccessGetDeepDeep(t *testing.T) {

	current := map[string]interface{}{"one": map[string]interface{}{"two": map[string]interface{}{"three": map[string]interface{}{"four": 4}}}}
	assert.Equal(t, 4, access(current, "one.two.three.four", nil, false, true))

}
func TestAccessorsAccessGetInsideArray(t *testing.T) {

	current := map[string]interface{}{"names": []interface{}{map[string]interface{}{"first": "Tyler", "last": "Bunnell"}, map[string]interface{}{"first": "Capitol", "last": "Bollocks"}}}
	assert.Equal(t, "Tyler", access(current, "names[0].first", nil, false, true))
	assert.Equal(t, "Bunnell", access(current, "names[0].last", nil, false, true))
	assert.Equal(t, "Capitol", access(current, "names[1].first", nil, false, true))
	assert.Equal(t, "Bollocks", access(current, "names[1].last", nil, false, true))

	assert.Panics(t, func() {
		access(current, "names[2]", nil, false, true)
	})
	assert.Nil(t, access(current, "names[2]", nil, false, false))

}

func TestAccessorsAccessGetFromArrayWithInt(t *testing.T) {

	current := []interface{}{map[string]interface{}{"first": "Tyler", "last": "Bunnell"}, map[string]interface{}{"first": "Capitol", "last": "Bollocks"}}
	one := access(current, 0, nil, false, false)
	two := access(current, 1, nil, false, false)
	three := access(current, 2, nil, false, false)

	assert.Equal(t, "Tyler", one.(map[string]interface{})["first"])
	assert.Equal(t, "Capitol", two.(map[string]interface{})["first"])
	assert.Nil(t, three)

}

func TestAccessorsGet(t *testing.T) {

	current := New(map[string]interface{}{"name": "Tyler"})
	assert.Equal(t, "Tyler", current.Get("name").data)

}

func TestAccessorsAccessSetSingleField(t *testing.T) {

	current := map[string]interface{}{"name": "Tyler"}
	access(current, "name", "Mat", true, false)
	assert.Equal(t, current["name"], "Mat")

	access(current, "age", 29, true, true)
	assert.Equal(t, current["age"], 29)

}

func TestAccessorsAccessSetSingleFieldNotExisting(t *testing.T) {

	current := map[string]interface{}{}
	access(current, "name", "Mat", true, false)
	assert.Equal(t, current["name"], "Mat")

}

func TestAccessorsAccessSetDeep(t *testing.T) {

	current := map[string]interface{}{"name": map[string]interface{}{"first": "Tyler", "last": "Bunnell"}}

	access(current, "name.first", "Mat", true, true)
	access(current, "name.last", "Ryer", true, true)

	assert.Equal(t, "Mat", access(current, "name.first", nil, false, true))
	assert.Equal(t, "Ryer", access(current, "name.last", nil, false, true))

}
func TestAccessorsAccessSetDeepDeep(t *testing.T) {

	current := map[string]interface{}{"one": map[string]interface{}{"two": map[string]interface{}{"three": map[string]interface{}{"four": 4}}}}

	access(current, "one.two.three.four", 5, true, true)

	assert.Equal(t, 5, access(current, "one.two.three.four", nil, false, true))

}
func TestAccessorsAccessSetArray(t *testing.T) {

	current := map[string]interface{}{"names": []interface{}{"Tyler"}}

	access(current, "names[0]", "Mat", true, true)

	assert.Equal(t, "Mat", access(current, "names[0]", nil, false, true))

}
func TestAccessorsAccessSetInsideArray(t *testing.T) {

	current := map[string]interface{}{"names": []interface{}{map[string]interface{}{"first": "Tyler", "last": "Bunnell"}, map[string]interface{}{"first": "Capitol", "last": "Bollocks"}}}

	access(current, "names[0].first", "Mat", true, true)
	access(current, "names[0].last", "Ryer", true, true)
	access(current, "names[1].first", "Captain", true, true)
	access(current, "names[1].last", "Underpants", true, true)

	assert.Equal(t, "Mat", access(current, "names[0].first", nil, false, true))
	assert.Equal(t, "Ryer", access(current, "names[0].last", nil, false, true))
	assert.Equal(t, "Captain", access(current, "names[1].first", nil, false, true))
	assert.Equal(t, "Underpants", access(current, "names[1].last", nil, false, true))

}

func TestAccessorsAccessSetFromArrayWithInt(t *testing.T) {

	current := []interface{}{map[string]interface{}{"first": "Tyler", "last": "Bunnell"}, map[string]interface{}{"first": "Capitol", "last": "Bollocks"}}
	one := access(current, 0, nil, false, false)
	two := access(current, 1, nil, false, false)
	three := access(current, 2, nil, false, false)

	assert.Equal(t, "Tyler", one.(map[string]interface{})["first"])
	assert.Equal(t, "Capitol", two.(map[string]interface{})["first"])
	assert.Nil(t, three)

}

func TestAccessorsSet(t *testing.T) {

	current := New(map[string]interface{}{"name": "Tyler"})
	current.Set("name", "Mat")
	assert.Equal(t, "Mat", current.Get("name").data)

}
