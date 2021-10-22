package objx_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
)

/*
   Tests for MSI (map[string]interface{} and []map[string]interface{})
*/
func TestMSI(t *testing.T) {
	val := map[string]interface{}(map[string]interface{}{"name": "Tyler"})
	m := objx.Map{"value": val, "nothing": nil}
	mVal := map[string]interface{}{"value": val, "nothing": nil}

	assert.Equal(t, mVal, m.Value().MSI())
	assert.Equal(t, val, m.Get("value").MSI())
	assert.Equal(t, mVal, m.Value().MustMSI())
	assert.Equal(t, val, m.Get("value").MustMSI())
	assert.Equal(t, map[string]interface{}(nil), m.Get("nothing").MSI())
	assert.Equal(t, val, m.Get("nothing").MSI(map[string]interface{}{"name": "Tyler"}))
	assert.Panics(t, func() {
		m.Get("age").MustMSI()
	})
}

func TestMSISlice(t *testing.T) {
	val := map[string]interface{}(map[string]interface{}{"name": "Tyler"})
	m := objx.Map{
		"value":   []map[string]interface{}{val},
		"value2":  []objx.Map{val},
		"value3":  []interface{}{val},
		"nothing": nil,
	}

	assert.Equal(t, val, m.Get("value").MSISlice()[0])
	assert.Equal(t, val, m.Get("value2").MSISlice()[0])
	assert.Equal(t, val, m.Get("value3").MSISlice()[0])
	assert.Equal(t, val, m.Get("value").MustMSISlice()[0])
	assert.Equal(t, val, m.Get("value2").MustMSISlice()[0])
	assert.Equal(t, val, m.Get("value3").MustMSISlice()[0])
	assert.Equal(t, []map[string]interface{}(nil), m.Get("nothing").MSISlice())
	assert.Equal(t, val, m.Get("nothing").MSISlice([]map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"})})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustMSISlice()
	})

	o := objx.MustFromJSON(`{"d":[{"author":{"displayName":"DemoUser3","id":2},"classes":null,"id":9879,"v":{"code":"","created":"2013-09-19T09:38:50+02:00","published":"0001-01-01T00:00:00Z","updated":"2013-09-19T09:38:50+02:00"}}],"s":200}`)
	assert.Equal(t, 9879, o.Get("d").MustMSISlice()[0]["id"])
	assert.Equal(t, 1, len(o.Get("d").MSISlice()))

	i := objx.MustFromJSON(`{"d":[{"author":"abc"},[1]]}`)
	assert.Nil(t, i.Get("d").MSISlice())
}

func TestIsMSI(t *testing.T) {
	m := objx.Map{"data": map[string]interface{}(map[string]interface{}{"name": "Tyler"})}

	assert.True(t, m.Get("data").IsMSI())
	assert.True(t, m.Value().IsMSI())
}

func TestIsMSISlice(t *testing.T) {
	val := map[string]interface{}(map[string]interface{}{"name": "Tyler"})
	m := objx.Map{"data": []map[string]interface{}{val}, "data2": []objx.Map{val}}

	assert.True(t, m.Get("data").IsMSISlice())
	assert.True(t, m.Get("data2").IsMSISlice())

	o := objx.MustFromJSON(`{"d":[{"author":{"displayName":"DemoUser3","id":2},"classes":null,"id":9879,"v":{"code":"","created":"2013-09-19T09:38:50+02:00","published":"0001-01-01T00:00:00Z","updated":"2013-09-19T09:38:50+02:00"}}],"s":200}`)
	assert.True(t, o.Has("d"))
	assert.True(t, o.Get("d").IsMSISlice())

	o = objx.MustFromJSON(`{"d":[{"author":"abc"},[1]]}`)
	assert.True(t, o.Has("d"))
	assert.False(t, o.Get("d").IsMSISlice())
}

func TestEachMSI(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	count := 0
	replacedVals := make([]map[string]interface{}, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachMSI(func(i int, val map[string]interface{}) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	m2 := objx.Map{"data": []objx.Map{{"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}}}
	assert.Equal(t, m2.Get("data"), m2.Get("data").EachMSI(func(i int, val map[string]interface{}) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 6)
	assert.Equal(t, replacedVals[0], m.Get("data").MustMSISlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustMSISlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustMSISlice()[2])
	assert.Equal(t, replacedVals[3], m2.Get("data").MustMSISlice()[0])
	assert.Equal(t, replacedVals[4], m2.Get("data").MustMSISlice()[1])
	assert.Equal(t, replacedVals[5], m2.Get("data").MustMSISlice()[2])
}

func TestWhereMSI(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	selected := m.Get("data").WhereMSI(func(i int, val map[string]interface{}) bool {
		return i%2 == 0
	}).MustMSISlice()

	assert.Equal(t, 3, len(selected))
}

func TestWhereMSI2(t *testing.T) {
	m := objx.Map{"data": []objx.Map{{"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}}}

	selected := m.Get("data").WhereMSI(func(i int, val map[string]interface{}) bool {
		return i%2 == 0
	}).MustMSISlice()

	assert.Equal(t, 2, len(selected))
}

func TestGroupMSI(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	grouped := m.Get("data").GroupMSI(func(i int, val map[string]interface{}) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]map[string]interface{})

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestGroupMSI2(t *testing.T) {
	m := objx.Map{"data": []objx.Map{{"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}}}

	grouped := m.Get("data").GroupMSI(func(i int, val map[string]interface{}) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]map[string]interface{})

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 2, len(grouped["false"]))
}

func TestReplaceMSI(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	rawArr := m.Get("data").MustMSISlice()

	replaced := m.Get("data").ReplaceMSI(func(index int, val map[string]interface{}) map[string]interface{} {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})
	replacedArr := replaced.MustMSISlice()

	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}
}

func TestReplaceMSI2(t *testing.T) {
	m := objx.Map{"data": []objx.Map{{"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}}}
	rawArr := m.Get("data").MustMSISlice()

	replaced := m.Get("data").ReplaceMSI(func(index int, val map[string]interface{}) map[string]interface{} {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})
	replacedArr := replaced.MustMSISlice()

	if assert.Equal(t, 5, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[0])
	}
}

func TestCollectMSI(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	collected := m.Get("data").CollectMSI(func(index int, val map[string]interface{}) interface{} {
		return index
	})
	collectedArr := collected.MustInterSlice()

	if assert.Equal(t, 6, len(collectedArr)) {
		assert.Equal(t, collectedArr[0], 0)
		assert.Equal(t, collectedArr[1], 1)
		assert.Equal(t, collectedArr[2], 2)
		assert.Equal(t, collectedArr[3], 3)
		assert.Equal(t, collectedArr[4], 4)
		assert.Equal(t, collectedArr[5], 5)
	}
}

func TestCollectMSI2(t *testing.T) {
	m := objx.Map{"data": []objx.Map{{"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}, {"name": "Taylor"}}}

	collected := m.Get("data").CollectMSI(func(index int, val map[string]interface{}) interface{} {
		return index
	})
	collectedArr := collected.MustInterSlice()

	if assert.Equal(t, 5, len(collectedArr)) {
		assert.Equal(t, collectedArr[0], 0)
		assert.Equal(t, collectedArr[1], 1)
		assert.Equal(t, collectedArr[2], 2)
		assert.Equal(t, collectedArr[3], 3)
		assert.Equal(t, collectedArr[4], 4)
	}
}

/*
   Tests for ObjxMap ((objx.Map) and [](objx.Map))
*/
func TestObjxMap(t *testing.T) {
	val := (objx.Map)(objx.New(1))
	m := objx.Map{"value": val, "value2": map[string]interface{}{"name": "Taylor"}, "nothing": nil}
	valMSI := objx.Map{"name": "Taylor"}

	assert.Equal(t, val, m.Get("value").ObjxMap())
	assert.Equal(t, valMSI, m.Get("value2").ObjxMap())
	assert.Equal(t, val, m.Get("value").MustObjxMap())
	assert.Equal(t, valMSI, m.Get("value2").MustObjxMap())
	assert.Equal(t, (objx.Map)(objx.New(nil)), m.Get("nothing").ObjxMap())
	assert.Equal(t, val, m.Get("nothing").ObjxMap(objx.New(1)))
	assert.Panics(t, func() {
		m.Get("age").MustObjxMap()
	})
}

func TestObjxMapSlice(t *testing.T) {
	val := (objx.Map)(objx.New(1))
	m := objx.Map{
		"value":   [](objx.Map){val},
		"value2":  []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Taylor"})},
		"value3":  []interface{}{val},
		"value4":  []interface{}{map[string]interface{}(map[string]interface{}{"name": "Taylor"})},
		"nothing": nil,
	}
	valMSI := objx.Map{"name": "Taylor"}

	assert.Equal(t, val, m.Get("value").ObjxMapSlice()[0])
	assert.Equal(t, valMSI, m.Get("value2").ObjxMapSlice()[0])
	assert.Equal(t, val, m.Get("value3").ObjxMapSlice()[0])
	assert.Equal(t, valMSI, m.Get("value4").ObjxMapSlice()[0])
	assert.Equal(t, val, m.Get("value").MustObjxMapSlice()[0])
	assert.Equal(t, valMSI, m.Get("value2").MustObjxMapSlice()[0])
	assert.Equal(t, val, m.Get("value3").MustObjxMapSlice()[0])
	assert.Equal(t, valMSI, m.Get("value4").MustObjxMapSlice()[0])
	assert.Equal(t, [](objx.Map)(nil), m.Get("nothing").ObjxMapSlice())
	assert.Equal(t, val, m.Get("nothing").ObjxMapSlice([](objx.Map){(objx.Map)(objx.New(1))})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustObjxMapSlice()
	})

	o := objx.MustFromJSON(`{"d":[{"author":{"displayName":"DemoUser3","id":2},"classes":null,"id":9879,"v":{"code":"","created":"2013-09-19T09:38:50+02:00","published":"0001-01-01T00:00:00Z","updated":"2013-09-19T09:38:50+02:00"}}],"s":200}`)
	assert.Equal(t, 9879, o.Get("d").MustObjxMapSlice()[0].Get("id").Int())
	assert.Equal(t, 1, len(o.Get("d").ObjxMapSlice()))

	i := objx.MustFromJSON(`{"d":[{"author":"abc"},[1]]}`)
	assert.Nil(t, i.Get("d").ObjxMapSlice())
}

func TestIsObjxMap(t *testing.T) {
	m := objx.Map{"data": (objx.Map)(objx.New(1)), "data2": map[string]interface{}{"name": "Taylor"}}

	assert.True(t, m.Get("data").IsObjxMap())
	assert.True(t, m.Get("data2").IsObjxMap())
}

func TestIsObjxMapSlice(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1))}, "data2": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Taylor"})}}

	assert.True(t, m.Get("data").IsObjxMapSlice())
	assert.True(t, m.Get("data2").IsObjxMapSlice())

	o := objx.MustFromJSON(`{"d":[{"author":{"displayName":"DemoUser3","id":2},"classes":null,"id":9879,"v":{"code":"","created":"2013-09-19T09:38:50+02:00","published":"0001-01-01T00:00:00Z","updated":"2013-09-19T09:38:50+02:00"}}],"s":200}`)
	assert.True(t, o.Has("d"))
	assert.True(t, o.Get("d").IsObjxMapSlice())

	//Valid json but not MSI slice
	o = objx.MustFromJSON(`{"d":[{"author":"abc"},[1]]}`)
	assert.True(t, o.Has("d"))
	assert.False(t, o.Get("d").IsObjxMapSlice())
}

func TestEachObjxMap(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1))}}
	count := 0
	replacedVals := make([](objx.Map), 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachObjxMap(func(i int, val objx.Map) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	m2 := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	assert.Equal(t, m2.Get("data"), m2.Get("data").EachObjxMap(func(i int, val objx.Map) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 6)
	assert.Equal(t, replacedVals[0], m.Get("data").MustObjxMapSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustObjxMapSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustObjxMapSlice()[2])
	assert.Equal(t, replacedVals[3], m2.Get("data").MustObjxMapSlice()[0])
	assert.Equal(t, replacedVals[4], m2.Get("data").MustObjxMapSlice()[1])
	assert.Equal(t, replacedVals[5], m2.Get("data").MustObjxMapSlice()[2])
}

func TestWhereObjxMap(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1))}}

	selected := m.Get("data").WhereObjxMap(func(i int, val objx.Map) bool {
		return i%2 == 0
	}).MustObjxMapSlice()

	assert.Equal(t, 3, len(selected))
}

func TestWhereObjxMap2(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	selected := m.Get("data").WhereObjxMap(func(i int, val objx.Map) bool {
		return i%2 == 0
	}).MustObjxMapSlice()

	assert.Equal(t, 2, len(selected))
}

func TestGroupObjxMap(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1))}}

	grouped := m.Get("data").GroupObjxMap(func(i int, val objx.Map) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][](objx.Map))

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestGroupObjxMap2(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	grouped := m.Get("data").GroupObjxMap(func(i int, val objx.Map) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][](objx.Map))

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 2, len(grouped["false"]))
}

func TestReplaceObjxMap(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1))}}
	rawArr := m.Get("data").MustObjxMapSlice()

	replaced := m.Get("data").ReplaceObjxMap(func(index int, val objx.Map) objx.Map {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})
	replacedArr := replaced.MustObjxMapSlice()

	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}
}

func TestReplaceObjxMap2(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	rawArr := m.Get("data").MustObjxMapSlice()

	replaced := m.Get("data").ReplaceObjxMap(func(index int, val objx.Map) objx.Map {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})
	replacedArr := replaced.MustObjxMapSlice()

	if assert.Equal(t, 5, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[0])
	}
}

func TestCollectObjxMap(t *testing.T) {
	m := objx.Map{"data": [](objx.Map){(objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1)), (objx.Map)(objx.New(1))}}

	collected := m.Get("data").CollectObjxMap(func(index int, val objx.Map) interface{} {
		return index
	})
	collectedArr := collected.MustInterSlice()

	if assert.Equal(t, 6, len(collectedArr)) {
		assert.Equal(t, collectedArr[0], 0)
		assert.Equal(t, collectedArr[1], 1)
		assert.Equal(t, collectedArr[2], 2)
		assert.Equal(t, collectedArr[3], 3)
		assert.Equal(t, collectedArr[4], 4)
		assert.Equal(t, collectedArr[5], 5)
	}
}

func TestCollectObjxMap2(t *testing.T) {
	m := objx.Map{"data": []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	collected := m.Get("data").CollectObjxMap(func(index int, val objx.Map) interface{} {
		return index
	})
	collectedArr := collected.MustInterSlice()

	if assert.Equal(t, 5, len(collectedArr)) {
		assert.Equal(t, collectedArr[0], 0)
		assert.Equal(t, collectedArr[1], 1)
		assert.Equal(t, collectedArr[2], 2)
		assert.Equal(t, collectedArr[3], 3)
		assert.Equal(t, collectedArr[4], 4)
	}
}
