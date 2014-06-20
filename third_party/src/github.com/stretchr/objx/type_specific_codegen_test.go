package objx

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

// ************************************************************
// TESTS
// ************************************************************

func TestInter(t *testing.T) {

	val := interface{}("something")
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Inter())
	assert.Equal(t, val, New(m).Get("value").MustInter())
	assert.Equal(t, interface{}(nil), New(m).Get("nothing").Inter())
	assert.Equal(t, val, New(m).Get("nothing").Inter("something"))

	assert.Panics(t, func() {
		New(m).Get("age").MustInter()
	})

}

func TestInterSlice(t *testing.T) {

	val := interface{}("something")
	m := map[string]interface{}{"value": []interface{}{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").InterSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustInterSlice()[0])
	assert.Equal(t, []interface{}(nil), New(m).Get("nothing").InterSlice())
	assert.Equal(t, val, New(m).Get("nothing").InterSlice([]interface{}{interface{}("something")})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustInterSlice()
	})

}

func TestIsInter(t *testing.T) {

	var v *Value

	v = &Value{data: interface{}("something")}
	assert.True(t, v.IsInter())

	v = &Value{data: []interface{}{interface{}("something")}}
	assert.True(t, v.IsInterSlice())

}

func TestEachInter(t *testing.T) {

	v := &Value{data: []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}
	count := 0
	replacedVals := make([]interface{}, 0)
	assert.Equal(t, v, v.EachInter(func(i int, val interface{}) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustInterSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustInterSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustInterSlice()[2])

}

func TestWhereInter(t *testing.T) {

	v := &Value{data: []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	selected := v.WhereInter(func(i int, val interface{}) bool {
		return i%2 == 0
	}).MustInterSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInter(t *testing.T) {

	v := &Value{data: []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	grouped := v.GroupInter(func(i int, val interface{}) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]interface{})

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInter(t *testing.T) {

	v := &Value{data: []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	rawArr := v.MustInterSlice()

	replaced := v.ReplaceInter(func(index int, val interface{}) interface{} {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustInterSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInter(t *testing.T) {

	v := &Value{data: []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	collected := v.CollectInter(func(index int, val interface{}) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestMSI(t *testing.T) {

	val := map[string]interface{}(map[string]interface{}{"name": "Tyler"})
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").MSI())
	assert.Equal(t, val, New(m).Get("value").MustMSI())
	assert.Equal(t, map[string]interface{}(nil), New(m).Get("nothing").MSI())
	assert.Equal(t, val, New(m).Get("nothing").MSI(map[string]interface{}{"name": "Tyler"}))

	assert.Panics(t, func() {
		New(m).Get("age").MustMSI()
	})

}

func TestMSISlice(t *testing.T) {

	val := map[string]interface{}(map[string]interface{}{"name": "Tyler"})
	m := map[string]interface{}{"value": []map[string]interface{}{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").MSISlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustMSISlice()[0])
	assert.Equal(t, []map[string]interface{}(nil), New(m).Get("nothing").MSISlice())
	assert.Equal(t, val, New(m).Get("nothing").MSISlice([]map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"})})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustMSISlice()
	})

}

func TestIsMSI(t *testing.T) {

	var v *Value

	v = &Value{data: map[string]interface{}(map[string]interface{}{"name": "Tyler"})}
	assert.True(t, v.IsMSI())

	v = &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	assert.True(t, v.IsMSISlice())

}

func TestEachMSI(t *testing.T) {

	v := &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}
	count := 0
	replacedVals := make([]map[string]interface{}, 0)
	assert.Equal(t, v, v.EachMSI(func(i int, val map[string]interface{}) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustMSISlice()[0])
	assert.Equal(t, replacedVals[1], v.MustMSISlice()[1])
	assert.Equal(t, replacedVals[2], v.MustMSISlice()[2])

}

func TestWhereMSI(t *testing.T) {

	v := &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	selected := v.WhereMSI(func(i int, val map[string]interface{}) bool {
		return i%2 == 0
	}).MustMSISlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupMSI(t *testing.T) {

	v := &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	grouped := v.GroupMSI(func(i int, val map[string]interface{}) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]map[string]interface{})

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceMSI(t *testing.T) {

	v := &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	rawArr := v.MustMSISlice()

	replaced := v.ReplaceMSI(func(index int, val map[string]interface{}) map[string]interface{} {
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

func TestCollectMSI(t *testing.T) {

	v := &Value{data: []map[string]interface{}{map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"}), map[string]interface{}(map[string]interface{}{"name": "Tyler"})}}

	collected := v.CollectMSI(func(index int, val map[string]interface{}) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestObjxMap(t *testing.T) {

	val := (Map)(New(1))
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").ObjxMap())
	assert.Equal(t, val, New(m).Get("value").MustObjxMap())
	assert.Equal(t, (Map)(New(nil)), New(m).Get("nothing").ObjxMap())
	assert.Equal(t, val, New(m).Get("nothing").ObjxMap(New(1)))

	assert.Panics(t, func() {
		New(m).Get("age").MustObjxMap()
	})

}

func TestObjxMapSlice(t *testing.T) {

	val := (Map)(New(1))
	m := map[string]interface{}{"value": [](Map){val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").ObjxMapSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustObjxMapSlice()[0])
	assert.Equal(t, [](Map)(nil), New(m).Get("nothing").ObjxMapSlice())
	assert.Equal(t, val, New(m).Get("nothing").ObjxMapSlice([](Map){(Map)(New(1))})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustObjxMapSlice()
	})

}

func TestIsObjxMap(t *testing.T) {

	var v *Value

	v = &Value{data: (Map)(New(1))}
	assert.True(t, v.IsObjxMap())

	v = &Value{data: [](Map){(Map)(New(1))}}
	assert.True(t, v.IsObjxMapSlice())

}

func TestEachObjxMap(t *testing.T) {

	v := &Value{data: [](Map){(Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1))}}
	count := 0
	replacedVals := make([](Map), 0)
	assert.Equal(t, v, v.EachObjxMap(func(i int, val Map) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustObjxMapSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustObjxMapSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustObjxMapSlice()[2])

}

func TestWhereObjxMap(t *testing.T) {

	v := &Value{data: [](Map){(Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1))}}

	selected := v.WhereObjxMap(func(i int, val Map) bool {
		return i%2 == 0
	}).MustObjxMapSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupObjxMap(t *testing.T) {

	v := &Value{data: [](Map){(Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1))}}

	grouped := v.GroupObjxMap(func(i int, val Map) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][](Map))

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceObjxMap(t *testing.T) {

	v := &Value{data: [](Map){(Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1))}}

	rawArr := v.MustObjxMapSlice()

	replaced := v.ReplaceObjxMap(func(index int, val Map) Map {
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

func TestCollectObjxMap(t *testing.T) {

	v := &Value{data: [](Map){(Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1)), (Map)(New(1))}}

	collected := v.CollectObjxMap(func(index int, val Map) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestBool(t *testing.T) {

	val := bool(true)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Bool())
	assert.Equal(t, val, New(m).Get("value").MustBool())
	assert.Equal(t, bool(false), New(m).Get("nothing").Bool())
	assert.Equal(t, val, New(m).Get("nothing").Bool(true))

	assert.Panics(t, func() {
		New(m).Get("age").MustBool()
	})

}

func TestBoolSlice(t *testing.T) {

	val := bool(true)
	m := map[string]interface{}{"value": []bool{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").BoolSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustBoolSlice()[0])
	assert.Equal(t, []bool(nil), New(m).Get("nothing").BoolSlice())
	assert.Equal(t, val, New(m).Get("nothing").BoolSlice([]bool{bool(true)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustBoolSlice()
	})

}

func TestIsBool(t *testing.T) {

	var v *Value

	v = &Value{data: bool(true)}
	assert.True(t, v.IsBool())

	v = &Value{data: []bool{bool(true)}}
	assert.True(t, v.IsBoolSlice())

}

func TestEachBool(t *testing.T) {

	v := &Value{data: []bool{bool(true), bool(true), bool(true), bool(true), bool(true)}}
	count := 0
	replacedVals := make([]bool, 0)
	assert.Equal(t, v, v.EachBool(func(i int, val bool) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustBoolSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustBoolSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustBoolSlice()[2])

}

func TestWhereBool(t *testing.T) {

	v := &Value{data: []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	selected := v.WhereBool(func(i int, val bool) bool {
		return i%2 == 0
	}).MustBoolSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupBool(t *testing.T) {

	v := &Value{data: []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	grouped := v.GroupBool(func(i int, val bool) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]bool)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceBool(t *testing.T) {

	v := &Value{data: []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	rawArr := v.MustBoolSlice()

	replaced := v.ReplaceBool(func(index int, val bool) bool {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustBoolSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectBool(t *testing.T) {

	v := &Value{data: []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	collected := v.CollectBool(func(index int, val bool) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestStr(t *testing.T) {

	val := string("hello")
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Str())
	assert.Equal(t, val, New(m).Get("value").MustStr())
	assert.Equal(t, string(""), New(m).Get("nothing").Str())
	assert.Equal(t, val, New(m).Get("nothing").Str("hello"))

	assert.Panics(t, func() {
		New(m).Get("age").MustStr()
	})

}

func TestStrSlice(t *testing.T) {

	val := string("hello")
	m := map[string]interface{}{"value": []string{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").StrSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustStrSlice()[0])
	assert.Equal(t, []string(nil), New(m).Get("nothing").StrSlice())
	assert.Equal(t, val, New(m).Get("nothing").StrSlice([]string{string("hello")})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustStrSlice()
	})

}

func TestIsStr(t *testing.T) {

	var v *Value

	v = &Value{data: string("hello")}
	assert.True(t, v.IsStr())

	v = &Value{data: []string{string("hello")}}
	assert.True(t, v.IsStrSlice())

}

func TestEachStr(t *testing.T) {

	v := &Value{data: []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}
	count := 0
	replacedVals := make([]string, 0)
	assert.Equal(t, v, v.EachStr(func(i int, val string) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustStrSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustStrSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustStrSlice()[2])

}

func TestWhereStr(t *testing.T) {

	v := &Value{data: []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	selected := v.WhereStr(func(i int, val string) bool {
		return i%2 == 0
	}).MustStrSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupStr(t *testing.T) {

	v := &Value{data: []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	grouped := v.GroupStr(func(i int, val string) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]string)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceStr(t *testing.T) {

	v := &Value{data: []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	rawArr := v.MustStrSlice()

	replaced := v.ReplaceStr(func(index int, val string) string {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustStrSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectStr(t *testing.T) {

	v := &Value{data: []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	collected := v.CollectStr(func(index int, val string) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestInt(t *testing.T) {

	val := int(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int())
	assert.Equal(t, val, New(m).Get("value").MustInt())
	assert.Equal(t, int(0), New(m).Get("nothing").Int())
	assert.Equal(t, val, New(m).Get("nothing").Int(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustInt()
	})

}

func TestIntSlice(t *testing.T) {

	val := int(1)
	m := map[string]interface{}{"value": []int{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").IntSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustIntSlice()[0])
	assert.Equal(t, []int(nil), New(m).Get("nothing").IntSlice())
	assert.Equal(t, val, New(m).Get("nothing").IntSlice([]int{int(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustIntSlice()
	})

}

func TestIsInt(t *testing.T) {

	var v *Value

	v = &Value{data: int(1)}
	assert.True(t, v.IsInt())

	v = &Value{data: []int{int(1)}}
	assert.True(t, v.IsIntSlice())

}

func TestEachInt(t *testing.T) {

	v := &Value{data: []int{int(1), int(1), int(1), int(1), int(1)}}
	count := 0
	replacedVals := make([]int, 0)
	assert.Equal(t, v, v.EachInt(func(i int, val int) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustIntSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustIntSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustIntSlice()[2])

}

func TestWhereInt(t *testing.T) {

	v := &Value{data: []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	selected := v.WhereInt(func(i int, val int) bool {
		return i%2 == 0
	}).MustIntSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInt(t *testing.T) {

	v := &Value{data: []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	grouped := v.GroupInt(func(i int, val int) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]int)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInt(t *testing.T) {

	v := &Value{data: []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	rawArr := v.MustIntSlice()

	replaced := v.ReplaceInt(func(index int, val int) int {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustIntSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInt(t *testing.T) {

	v := &Value{data: []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	collected := v.CollectInt(func(index int, val int) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestInt8(t *testing.T) {

	val := int8(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int8())
	assert.Equal(t, val, New(m).Get("value").MustInt8())
	assert.Equal(t, int8(0), New(m).Get("nothing").Int8())
	assert.Equal(t, val, New(m).Get("nothing").Int8(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustInt8()
	})

}

func TestInt8Slice(t *testing.T) {

	val := int8(1)
	m := map[string]interface{}{"value": []int8{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int8Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustInt8Slice()[0])
	assert.Equal(t, []int8(nil), New(m).Get("nothing").Int8Slice())
	assert.Equal(t, val, New(m).Get("nothing").Int8Slice([]int8{int8(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustInt8Slice()
	})

}

func TestIsInt8(t *testing.T) {

	var v *Value

	v = &Value{data: int8(1)}
	assert.True(t, v.IsInt8())

	v = &Value{data: []int8{int8(1)}}
	assert.True(t, v.IsInt8Slice())

}

func TestEachInt8(t *testing.T) {

	v := &Value{data: []int8{int8(1), int8(1), int8(1), int8(1), int8(1)}}
	count := 0
	replacedVals := make([]int8, 0)
	assert.Equal(t, v, v.EachInt8(func(i int, val int8) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustInt8Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustInt8Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustInt8Slice()[2])

}

func TestWhereInt8(t *testing.T) {

	v := &Value{data: []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	selected := v.WhereInt8(func(i int, val int8) bool {
		return i%2 == 0
	}).MustInt8Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInt8(t *testing.T) {

	v := &Value{data: []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	grouped := v.GroupInt8(func(i int, val int8) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]int8)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInt8(t *testing.T) {

	v := &Value{data: []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	rawArr := v.MustInt8Slice()

	replaced := v.ReplaceInt8(func(index int, val int8) int8 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustInt8Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInt8(t *testing.T) {

	v := &Value{data: []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	collected := v.CollectInt8(func(index int, val int8) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestInt16(t *testing.T) {

	val := int16(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int16())
	assert.Equal(t, val, New(m).Get("value").MustInt16())
	assert.Equal(t, int16(0), New(m).Get("nothing").Int16())
	assert.Equal(t, val, New(m).Get("nothing").Int16(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustInt16()
	})

}

func TestInt16Slice(t *testing.T) {

	val := int16(1)
	m := map[string]interface{}{"value": []int16{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int16Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustInt16Slice()[0])
	assert.Equal(t, []int16(nil), New(m).Get("nothing").Int16Slice())
	assert.Equal(t, val, New(m).Get("nothing").Int16Slice([]int16{int16(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustInt16Slice()
	})

}

func TestIsInt16(t *testing.T) {

	var v *Value

	v = &Value{data: int16(1)}
	assert.True(t, v.IsInt16())

	v = &Value{data: []int16{int16(1)}}
	assert.True(t, v.IsInt16Slice())

}

func TestEachInt16(t *testing.T) {

	v := &Value{data: []int16{int16(1), int16(1), int16(1), int16(1), int16(1)}}
	count := 0
	replacedVals := make([]int16, 0)
	assert.Equal(t, v, v.EachInt16(func(i int, val int16) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustInt16Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustInt16Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustInt16Slice()[2])

}

func TestWhereInt16(t *testing.T) {

	v := &Value{data: []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	selected := v.WhereInt16(func(i int, val int16) bool {
		return i%2 == 0
	}).MustInt16Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInt16(t *testing.T) {

	v := &Value{data: []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	grouped := v.GroupInt16(func(i int, val int16) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]int16)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInt16(t *testing.T) {

	v := &Value{data: []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	rawArr := v.MustInt16Slice()

	replaced := v.ReplaceInt16(func(index int, val int16) int16 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustInt16Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInt16(t *testing.T) {

	v := &Value{data: []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	collected := v.CollectInt16(func(index int, val int16) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestInt32(t *testing.T) {

	val := int32(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int32())
	assert.Equal(t, val, New(m).Get("value").MustInt32())
	assert.Equal(t, int32(0), New(m).Get("nothing").Int32())
	assert.Equal(t, val, New(m).Get("nothing").Int32(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustInt32()
	})

}

func TestInt32Slice(t *testing.T) {

	val := int32(1)
	m := map[string]interface{}{"value": []int32{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int32Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustInt32Slice()[0])
	assert.Equal(t, []int32(nil), New(m).Get("nothing").Int32Slice())
	assert.Equal(t, val, New(m).Get("nothing").Int32Slice([]int32{int32(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustInt32Slice()
	})

}

func TestIsInt32(t *testing.T) {

	var v *Value

	v = &Value{data: int32(1)}
	assert.True(t, v.IsInt32())

	v = &Value{data: []int32{int32(1)}}
	assert.True(t, v.IsInt32Slice())

}

func TestEachInt32(t *testing.T) {

	v := &Value{data: []int32{int32(1), int32(1), int32(1), int32(1), int32(1)}}
	count := 0
	replacedVals := make([]int32, 0)
	assert.Equal(t, v, v.EachInt32(func(i int, val int32) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustInt32Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustInt32Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustInt32Slice()[2])

}

func TestWhereInt32(t *testing.T) {

	v := &Value{data: []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	selected := v.WhereInt32(func(i int, val int32) bool {
		return i%2 == 0
	}).MustInt32Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInt32(t *testing.T) {

	v := &Value{data: []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	grouped := v.GroupInt32(func(i int, val int32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]int32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInt32(t *testing.T) {

	v := &Value{data: []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	rawArr := v.MustInt32Slice()

	replaced := v.ReplaceInt32(func(index int, val int32) int32 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustInt32Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInt32(t *testing.T) {

	v := &Value{data: []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	collected := v.CollectInt32(func(index int, val int32) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestInt64(t *testing.T) {

	val := int64(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int64())
	assert.Equal(t, val, New(m).Get("value").MustInt64())
	assert.Equal(t, int64(0), New(m).Get("nothing").Int64())
	assert.Equal(t, val, New(m).Get("nothing").Int64(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustInt64()
	})

}

func TestInt64Slice(t *testing.T) {

	val := int64(1)
	m := map[string]interface{}{"value": []int64{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Int64Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustInt64Slice()[0])
	assert.Equal(t, []int64(nil), New(m).Get("nothing").Int64Slice())
	assert.Equal(t, val, New(m).Get("nothing").Int64Slice([]int64{int64(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustInt64Slice()
	})

}

func TestIsInt64(t *testing.T) {

	var v *Value

	v = &Value{data: int64(1)}
	assert.True(t, v.IsInt64())

	v = &Value{data: []int64{int64(1)}}
	assert.True(t, v.IsInt64Slice())

}

func TestEachInt64(t *testing.T) {

	v := &Value{data: []int64{int64(1), int64(1), int64(1), int64(1), int64(1)}}
	count := 0
	replacedVals := make([]int64, 0)
	assert.Equal(t, v, v.EachInt64(func(i int, val int64) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustInt64Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustInt64Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustInt64Slice()[2])

}

func TestWhereInt64(t *testing.T) {

	v := &Value{data: []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	selected := v.WhereInt64(func(i int, val int64) bool {
		return i%2 == 0
	}).MustInt64Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupInt64(t *testing.T) {

	v := &Value{data: []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	grouped := v.GroupInt64(func(i int, val int64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]int64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceInt64(t *testing.T) {

	v := &Value{data: []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	rawArr := v.MustInt64Slice()

	replaced := v.ReplaceInt64(func(index int, val int64) int64 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustInt64Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectInt64(t *testing.T) {

	v := &Value{data: []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	collected := v.CollectInt64(func(index int, val int64) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUint(t *testing.T) {

	val := uint(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint())
	assert.Equal(t, val, New(m).Get("value").MustUint())
	assert.Equal(t, uint(0), New(m).Get("nothing").Uint())
	assert.Equal(t, val, New(m).Get("nothing").Uint(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUint()
	})

}

func TestUintSlice(t *testing.T) {

	val := uint(1)
	m := map[string]interface{}{"value": []uint{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").UintSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUintSlice()[0])
	assert.Equal(t, []uint(nil), New(m).Get("nothing").UintSlice())
	assert.Equal(t, val, New(m).Get("nothing").UintSlice([]uint{uint(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUintSlice()
	})

}

func TestIsUint(t *testing.T) {

	var v *Value

	v = &Value{data: uint(1)}
	assert.True(t, v.IsUint())

	v = &Value{data: []uint{uint(1)}}
	assert.True(t, v.IsUintSlice())

}

func TestEachUint(t *testing.T) {

	v := &Value{data: []uint{uint(1), uint(1), uint(1), uint(1), uint(1)}}
	count := 0
	replacedVals := make([]uint, 0)
	assert.Equal(t, v, v.EachUint(func(i int, val uint) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUintSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustUintSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustUintSlice()[2])

}

func TestWhereUint(t *testing.T) {

	v := &Value{data: []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	selected := v.WhereUint(func(i int, val uint) bool {
		return i%2 == 0
	}).MustUintSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUint(t *testing.T) {

	v := &Value{data: []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	grouped := v.GroupUint(func(i int, val uint) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uint)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUint(t *testing.T) {

	v := &Value{data: []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	rawArr := v.MustUintSlice()

	replaced := v.ReplaceUint(func(index int, val uint) uint {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUintSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUint(t *testing.T) {

	v := &Value{data: []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	collected := v.CollectUint(func(index int, val uint) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUint8(t *testing.T) {

	val := uint8(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint8())
	assert.Equal(t, val, New(m).Get("value").MustUint8())
	assert.Equal(t, uint8(0), New(m).Get("nothing").Uint8())
	assert.Equal(t, val, New(m).Get("nothing").Uint8(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUint8()
	})

}

func TestUint8Slice(t *testing.T) {

	val := uint8(1)
	m := map[string]interface{}{"value": []uint8{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint8Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUint8Slice()[0])
	assert.Equal(t, []uint8(nil), New(m).Get("nothing").Uint8Slice())
	assert.Equal(t, val, New(m).Get("nothing").Uint8Slice([]uint8{uint8(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUint8Slice()
	})

}

func TestIsUint8(t *testing.T) {

	var v *Value

	v = &Value{data: uint8(1)}
	assert.True(t, v.IsUint8())

	v = &Value{data: []uint8{uint8(1)}}
	assert.True(t, v.IsUint8Slice())

}

func TestEachUint8(t *testing.T) {

	v := &Value{data: []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}
	count := 0
	replacedVals := make([]uint8, 0)
	assert.Equal(t, v, v.EachUint8(func(i int, val uint8) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUint8Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustUint8Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustUint8Slice()[2])

}

func TestWhereUint8(t *testing.T) {

	v := &Value{data: []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	selected := v.WhereUint8(func(i int, val uint8) bool {
		return i%2 == 0
	}).MustUint8Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUint8(t *testing.T) {

	v := &Value{data: []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	grouped := v.GroupUint8(func(i int, val uint8) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uint8)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUint8(t *testing.T) {

	v := &Value{data: []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	rawArr := v.MustUint8Slice()

	replaced := v.ReplaceUint8(func(index int, val uint8) uint8 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUint8Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUint8(t *testing.T) {

	v := &Value{data: []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	collected := v.CollectUint8(func(index int, val uint8) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUint16(t *testing.T) {

	val := uint16(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint16())
	assert.Equal(t, val, New(m).Get("value").MustUint16())
	assert.Equal(t, uint16(0), New(m).Get("nothing").Uint16())
	assert.Equal(t, val, New(m).Get("nothing").Uint16(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUint16()
	})

}

func TestUint16Slice(t *testing.T) {

	val := uint16(1)
	m := map[string]interface{}{"value": []uint16{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint16Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUint16Slice()[0])
	assert.Equal(t, []uint16(nil), New(m).Get("nothing").Uint16Slice())
	assert.Equal(t, val, New(m).Get("nothing").Uint16Slice([]uint16{uint16(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUint16Slice()
	})

}

func TestIsUint16(t *testing.T) {

	var v *Value

	v = &Value{data: uint16(1)}
	assert.True(t, v.IsUint16())

	v = &Value{data: []uint16{uint16(1)}}
	assert.True(t, v.IsUint16Slice())

}

func TestEachUint16(t *testing.T) {

	v := &Value{data: []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}
	count := 0
	replacedVals := make([]uint16, 0)
	assert.Equal(t, v, v.EachUint16(func(i int, val uint16) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUint16Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustUint16Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustUint16Slice()[2])

}

func TestWhereUint16(t *testing.T) {

	v := &Value{data: []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	selected := v.WhereUint16(func(i int, val uint16) bool {
		return i%2 == 0
	}).MustUint16Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUint16(t *testing.T) {

	v := &Value{data: []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	grouped := v.GroupUint16(func(i int, val uint16) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uint16)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUint16(t *testing.T) {

	v := &Value{data: []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	rawArr := v.MustUint16Slice()

	replaced := v.ReplaceUint16(func(index int, val uint16) uint16 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUint16Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUint16(t *testing.T) {

	v := &Value{data: []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	collected := v.CollectUint16(func(index int, val uint16) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUint32(t *testing.T) {

	val := uint32(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint32())
	assert.Equal(t, val, New(m).Get("value").MustUint32())
	assert.Equal(t, uint32(0), New(m).Get("nothing").Uint32())
	assert.Equal(t, val, New(m).Get("nothing").Uint32(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUint32()
	})

}

func TestUint32Slice(t *testing.T) {

	val := uint32(1)
	m := map[string]interface{}{"value": []uint32{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint32Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUint32Slice()[0])
	assert.Equal(t, []uint32(nil), New(m).Get("nothing").Uint32Slice())
	assert.Equal(t, val, New(m).Get("nothing").Uint32Slice([]uint32{uint32(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUint32Slice()
	})

}

func TestIsUint32(t *testing.T) {

	var v *Value

	v = &Value{data: uint32(1)}
	assert.True(t, v.IsUint32())

	v = &Value{data: []uint32{uint32(1)}}
	assert.True(t, v.IsUint32Slice())

}

func TestEachUint32(t *testing.T) {

	v := &Value{data: []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}
	count := 0
	replacedVals := make([]uint32, 0)
	assert.Equal(t, v, v.EachUint32(func(i int, val uint32) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUint32Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustUint32Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustUint32Slice()[2])

}

func TestWhereUint32(t *testing.T) {

	v := &Value{data: []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	selected := v.WhereUint32(func(i int, val uint32) bool {
		return i%2 == 0
	}).MustUint32Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUint32(t *testing.T) {

	v := &Value{data: []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	grouped := v.GroupUint32(func(i int, val uint32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uint32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUint32(t *testing.T) {

	v := &Value{data: []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	rawArr := v.MustUint32Slice()

	replaced := v.ReplaceUint32(func(index int, val uint32) uint32 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUint32Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUint32(t *testing.T) {

	v := &Value{data: []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	collected := v.CollectUint32(func(index int, val uint32) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUint64(t *testing.T) {

	val := uint64(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint64())
	assert.Equal(t, val, New(m).Get("value").MustUint64())
	assert.Equal(t, uint64(0), New(m).Get("nothing").Uint64())
	assert.Equal(t, val, New(m).Get("nothing").Uint64(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUint64()
	})

}

func TestUint64Slice(t *testing.T) {

	val := uint64(1)
	m := map[string]interface{}{"value": []uint64{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uint64Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUint64Slice()[0])
	assert.Equal(t, []uint64(nil), New(m).Get("nothing").Uint64Slice())
	assert.Equal(t, val, New(m).Get("nothing").Uint64Slice([]uint64{uint64(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUint64Slice()
	})

}

func TestIsUint64(t *testing.T) {

	var v *Value

	v = &Value{data: uint64(1)}
	assert.True(t, v.IsUint64())

	v = &Value{data: []uint64{uint64(1)}}
	assert.True(t, v.IsUint64Slice())

}

func TestEachUint64(t *testing.T) {

	v := &Value{data: []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}
	count := 0
	replacedVals := make([]uint64, 0)
	assert.Equal(t, v, v.EachUint64(func(i int, val uint64) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUint64Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustUint64Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustUint64Slice()[2])

}

func TestWhereUint64(t *testing.T) {

	v := &Value{data: []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	selected := v.WhereUint64(func(i int, val uint64) bool {
		return i%2 == 0
	}).MustUint64Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUint64(t *testing.T) {

	v := &Value{data: []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	grouped := v.GroupUint64(func(i int, val uint64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uint64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUint64(t *testing.T) {

	v := &Value{data: []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	rawArr := v.MustUint64Slice()

	replaced := v.ReplaceUint64(func(index int, val uint64) uint64 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUint64Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUint64(t *testing.T) {

	v := &Value{data: []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	collected := v.CollectUint64(func(index int, val uint64) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestUintptr(t *testing.T) {

	val := uintptr(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Uintptr())
	assert.Equal(t, val, New(m).Get("value").MustUintptr())
	assert.Equal(t, uintptr(0), New(m).Get("nothing").Uintptr())
	assert.Equal(t, val, New(m).Get("nothing").Uintptr(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustUintptr()
	})

}

func TestUintptrSlice(t *testing.T) {

	val := uintptr(1)
	m := map[string]interface{}{"value": []uintptr{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").UintptrSlice()[0])
	assert.Equal(t, val, New(m).Get("value").MustUintptrSlice()[0])
	assert.Equal(t, []uintptr(nil), New(m).Get("nothing").UintptrSlice())
	assert.Equal(t, val, New(m).Get("nothing").UintptrSlice([]uintptr{uintptr(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustUintptrSlice()
	})

}

func TestIsUintptr(t *testing.T) {

	var v *Value

	v = &Value{data: uintptr(1)}
	assert.True(t, v.IsUintptr())

	v = &Value{data: []uintptr{uintptr(1)}}
	assert.True(t, v.IsUintptrSlice())

}

func TestEachUintptr(t *testing.T) {

	v := &Value{data: []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}
	count := 0
	replacedVals := make([]uintptr, 0)
	assert.Equal(t, v, v.EachUintptr(func(i int, val uintptr) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustUintptrSlice()[0])
	assert.Equal(t, replacedVals[1], v.MustUintptrSlice()[1])
	assert.Equal(t, replacedVals[2], v.MustUintptrSlice()[2])

}

func TestWhereUintptr(t *testing.T) {

	v := &Value{data: []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	selected := v.WhereUintptr(func(i int, val uintptr) bool {
		return i%2 == 0
	}).MustUintptrSlice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupUintptr(t *testing.T) {

	v := &Value{data: []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	grouped := v.GroupUintptr(func(i int, val uintptr) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]uintptr)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceUintptr(t *testing.T) {

	v := &Value{data: []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	rawArr := v.MustUintptrSlice()

	replaced := v.ReplaceUintptr(func(index int, val uintptr) uintptr {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustUintptrSlice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectUintptr(t *testing.T) {

	v := &Value{data: []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	collected := v.CollectUintptr(func(index int, val uintptr) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestFloat32(t *testing.T) {

	val := float32(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Float32())
	assert.Equal(t, val, New(m).Get("value").MustFloat32())
	assert.Equal(t, float32(0), New(m).Get("nothing").Float32())
	assert.Equal(t, val, New(m).Get("nothing").Float32(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustFloat32()
	})

}

func TestFloat32Slice(t *testing.T) {

	val := float32(1)
	m := map[string]interface{}{"value": []float32{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Float32Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustFloat32Slice()[0])
	assert.Equal(t, []float32(nil), New(m).Get("nothing").Float32Slice())
	assert.Equal(t, val, New(m).Get("nothing").Float32Slice([]float32{float32(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustFloat32Slice()
	})

}

func TestIsFloat32(t *testing.T) {

	var v *Value

	v = &Value{data: float32(1)}
	assert.True(t, v.IsFloat32())

	v = &Value{data: []float32{float32(1)}}
	assert.True(t, v.IsFloat32Slice())

}

func TestEachFloat32(t *testing.T) {

	v := &Value{data: []float32{float32(1), float32(1), float32(1), float32(1), float32(1)}}
	count := 0
	replacedVals := make([]float32, 0)
	assert.Equal(t, v, v.EachFloat32(func(i int, val float32) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustFloat32Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustFloat32Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustFloat32Slice()[2])

}

func TestWhereFloat32(t *testing.T) {

	v := &Value{data: []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	selected := v.WhereFloat32(func(i int, val float32) bool {
		return i%2 == 0
	}).MustFloat32Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupFloat32(t *testing.T) {

	v := &Value{data: []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	grouped := v.GroupFloat32(func(i int, val float32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]float32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceFloat32(t *testing.T) {

	v := &Value{data: []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	rawArr := v.MustFloat32Slice()

	replaced := v.ReplaceFloat32(func(index int, val float32) float32 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustFloat32Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectFloat32(t *testing.T) {

	v := &Value{data: []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	collected := v.CollectFloat32(func(index int, val float32) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestFloat64(t *testing.T) {

	val := float64(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Float64())
	assert.Equal(t, val, New(m).Get("value").MustFloat64())
	assert.Equal(t, float64(0), New(m).Get("nothing").Float64())
	assert.Equal(t, val, New(m).Get("nothing").Float64(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustFloat64()
	})

}

func TestFloat64Slice(t *testing.T) {

	val := float64(1)
	m := map[string]interface{}{"value": []float64{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Float64Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustFloat64Slice()[0])
	assert.Equal(t, []float64(nil), New(m).Get("nothing").Float64Slice())
	assert.Equal(t, val, New(m).Get("nothing").Float64Slice([]float64{float64(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustFloat64Slice()
	})

}

func TestIsFloat64(t *testing.T) {

	var v *Value

	v = &Value{data: float64(1)}
	assert.True(t, v.IsFloat64())

	v = &Value{data: []float64{float64(1)}}
	assert.True(t, v.IsFloat64Slice())

}

func TestEachFloat64(t *testing.T) {

	v := &Value{data: []float64{float64(1), float64(1), float64(1), float64(1), float64(1)}}
	count := 0
	replacedVals := make([]float64, 0)
	assert.Equal(t, v, v.EachFloat64(func(i int, val float64) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustFloat64Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustFloat64Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustFloat64Slice()[2])

}

func TestWhereFloat64(t *testing.T) {

	v := &Value{data: []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	selected := v.WhereFloat64(func(i int, val float64) bool {
		return i%2 == 0
	}).MustFloat64Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupFloat64(t *testing.T) {

	v := &Value{data: []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	grouped := v.GroupFloat64(func(i int, val float64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]float64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceFloat64(t *testing.T) {

	v := &Value{data: []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	rawArr := v.MustFloat64Slice()

	replaced := v.ReplaceFloat64(func(index int, val float64) float64 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustFloat64Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectFloat64(t *testing.T) {

	v := &Value{data: []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	collected := v.CollectFloat64(func(index int, val float64) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestComplex64(t *testing.T) {

	val := complex64(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Complex64())
	assert.Equal(t, val, New(m).Get("value").MustComplex64())
	assert.Equal(t, complex64(0), New(m).Get("nothing").Complex64())
	assert.Equal(t, val, New(m).Get("nothing").Complex64(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustComplex64()
	})

}

func TestComplex64Slice(t *testing.T) {

	val := complex64(1)
	m := map[string]interface{}{"value": []complex64{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Complex64Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustComplex64Slice()[0])
	assert.Equal(t, []complex64(nil), New(m).Get("nothing").Complex64Slice())
	assert.Equal(t, val, New(m).Get("nothing").Complex64Slice([]complex64{complex64(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustComplex64Slice()
	})

}

func TestIsComplex64(t *testing.T) {

	var v *Value

	v = &Value{data: complex64(1)}
	assert.True(t, v.IsComplex64())

	v = &Value{data: []complex64{complex64(1)}}
	assert.True(t, v.IsComplex64Slice())

}

func TestEachComplex64(t *testing.T) {

	v := &Value{data: []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}
	count := 0
	replacedVals := make([]complex64, 0)
	assert.Equal(t, v, v.EachComplex64(func(i int, val complex64) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustComplex64Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustComplex64Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustComplex64Slice()[2])

}

func TestWhereComplex64(t *testing.T) {

	v := &Value{data: []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	selected := v.WhereComplex64(func(i int, val complex64) bool {
		return i%2 == 0
	}).MustComplex64Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupComplex64(t *testing.T) {

	v := &Value{data: []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	grouped := v.GroupComplex64(func(i int, val complex64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]complex64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceComplex64(t *testing.T) {

	v := &Value{data: []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	rawArr := v.MustComplex64Slice()

	replaced := v.ReplaceComplex64(func(index int, val complex64) complex64 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustComplex64Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectComplex64(t *testing.T) {

	v := &Value{data: []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	collected := v.CollectComplex64(func(index int, val complex64) interface{} {
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

// ************************************************************
// TESTS
// ************************************************************

func TestComplex128(t *testing.T) {

	val := complex128(1)
	m := map[string]interface{}{"value": val, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Complex128())
	assert.Equal(t, val, New(m).Get("value").MustComplex128())
	assert.Equal(t, complex128(0), New(m).Get("nothing").Complex128())
	assert.Equal(t, val, New(m).Get("nothing").Complex128(1))

	assert.Panics(t, func() {
		New(m).Get("age").MustComplex128()
	})

}

func TestComplex128Slice(t *testing.T) {

	val := complex128(1)
	m := map[string]interface{}{"value": []complex128{val}, "nothing": nil}
	assert.Equal(t, val, New(m).Get("value").Complex128Slice()[0])
	assert.Equal(t, val, New(m).Get("value").MustComplex128Slice()[0])
	assert.Equal(t, []complex128(nil), New(m).Get("nothing").Complex128Slice())
	assert.Equal(t, val, New(m).Get("nothing").Complex128Slice([]complex128{complex128(1)})[0])

	assert.Panics(t, func() {
		New(m).Get("nothing").MustComplex128Slice()
	})

}

func TestIsComplex128(t *testing.T) {

	var v *Value

	v = &Value{data: complex128(1)}
	assert.True(t, v.IsComplex128())

	v = &Value{data: []complex128{complex128(1)}}
	assert.True(t, v.IsComplex128Slice())

}

func TestEachComplex128(t *testing.T) {

	v := &Value{data: []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}
	count := 0
	replacedVals := make([]complex128, 0)
	assert.Equal(t, v, v.EachComplex128(func(i int, val complex128) bool {

		count++
		replacedVals = append(replacedVals, val)

		// abort early
		if i == 2 {
			return false
		}

		return true

	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], v.MustComplex128Slice()[0])
	assert.Equal(t, replacedVals[1], v.MustComplex128Slice()[1])
	assert.Equal(t, replacedVals[2], v.MustComplex128Slice()[2])

}

func TestWhereComplex128(t *testing.T) {

	v := &Value{data: []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	selected := v.WhereComplex128(func(i int, val complex128) bool {
		return i%2 == 0
	}).MustComplex128Slice()

	assert.Equal(t, 3, len(selected))

}

func TestGroupComplex128(t *testing.T) {

	v := &Value{data: []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	grouped := v.GroupComplex128(func(i int, val complex128) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).data.(map[string][]complex128)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))

}

func TestReplaceComplex128(t *testing.T) {

	v := &Value{data: []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	rawArr := v.MustComplex128Slice()

	replaced := v.ReplaceComplex128(func(index int, val complex128) complex128 {
		if index < len(rawArr)-1 {
			return rawArr[index+1]
		}
		return rawArr[0]
	})

	replacedArr := replaced.MustComplex128Slice()
	if assert.Equal(t, 6, len(replacedArr)) {
		assert.Equal(t, replacedArr[0], rawArr[1])
		assert.Equal(t, replacedArr[1], rawArr[2])
		assert.Equal(t, replacedArr[2], rawArr[3])
		assert.Equal(t, replacedArr[3], rawArr[4])
		assert.Equal(t, replacedArr[4], rawArr[5])
		assert.Equal(t, replacedArr[5], rawArr[0])
	}

}

func TestCollectComplex128(t *testing.T) {

	v := &Value{data: []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	collected := v.CollectComplex128(func(index int, val complex128) interface{} {
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
