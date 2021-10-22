package objx_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/objx"
	"github.com/stretchr/testify/assert"
)

/*
   Tests for Inter (interface{} and []interface{})
*/
func TestInter(t *testing.T) {
	val := interface{}("something")
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Inter())
	assert.Equal(t, val, m.Get("value").MustInter())
	assert.Equal(t, interface{}(nil), m.Get("nothing").Inter())
	assert.Equal(t, val, m.Get("nothing").Inter("something"))
	assert.Panics(t, func() {
		m.Get("age").MustInter()
	})
}

func TestInterSlice(t *testing.T) {
	val := interface{}("something")
	m := objx.Map{"value": []interface{}{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").InterSlice()[0])
	assert.Equal(t, val, m.Get("value").MustInterSlice()[0])
	assert.Equal(t, []interface{}(nil), m.Get("nothing").InterSlice())
	assert.Equal(t, val, m.Get("nothing").InterSlice([]interface{}{interface{}("something")})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustInterSlice()
	})
}

func TestIsInter(t *testing.T) {
	m := objx.Map{"data": interface{}("something")}

	assert.True(t, m.Get("data").IsInter())
}

func TestIsInterSlice(t *testing.T) {
	m := objx.Map{"data": []interface{}{interface{}("something")}}

	assert.True(t, m.Get("data").IsInterSlice())
}

func TestEachInter(t *testing.T) {
	m := objx.Map{"data": []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}
	count := 0
	replacedVals := make([]interface{}, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInter(func(i int, val interface{}) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustInterSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustInterSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustInterSlice()[2])
}

func TestWhereInter(t *testing.T) {
	m := objx.Map{"data": []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	selected := m.Get("data").WhereInter(func(i int, val interface{}) bool {
		return i%2 == 0
	}).MustInterSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInter(t *testing.T) {
	m := objx.Map{"data": []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	grouped := m.Get("data").GroupInter(func(i int, val interface{}) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]interface{})

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInter(t *testing.T) {
	m := objx.Map{"data": []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}
	rawArr := m.Get("data").MustInterSlice()

	replaced := m.Get("data").ReplaceInter(func(index int, val interface{}) interface{} {
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
	m := objx.Map{"data": []interface{}{interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something"), interface{}("something")}}

	collected := m.Get("data").CollectInter(func(index int, val interface{}) interface{} {
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

/*
   Tests for Bool (bool and []bool)
*/
func TestBool(t *testing.T) {
	val := bool(true)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Bool())
	assert.Equal(t, val, m.Get("value").MustBool())
	assert.Equal(t, bool(false), m.Get("nothing").Bool())
	assert.Equal(t, val, m.Get("nothing").Bool(true))
	assert.Panics(t, func() {
		m.Get("age").MustBool()
	})
}

func TestBoolSlice(t *testing.T) {
	val := bool(true)
	m := objx.Map{"value": []bool{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").BoolSlice()[0])
	assert.Equal(t, val, m.Get("value").MustBoolSlice()[0])
	assert.Equal(t, []bool(nil), m.Get("nothing").BoolSlice())
	assert.Equal(t, val, m.Get("nothing").BoolSlice([]bool{bool(true)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustBoolSlice()
	})
}

func TestIsBool(t *testing.T) {
	m := objx.Map{"data": bool(true)}

	assert.True(t, m.Get("data").IsBool())
}

func TestIsBoolSlice(t *testing.T) {
	m := objx.Map{"data": []bool{bool(true)}}

	assert.True(t, m.Get("data").IsBoolSlice())
}

func TestEachBool(t *testing.T) {
	m := objx.Map{"data": []bool{bool(true), bool(true), bool(true), bool(true), bool(true)}}
	count := 0
	replacedVals := make([]bool, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachBool(func(i int, val bool) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustBoolSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustBoolSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustBoolSlice()[2])
}

func TestWhereBool(t *testing.T) {
	m := objx.Map{"data": []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	selected := m.Get("data").WhereBool(func(i int, val bool) bool {
		return i%2 == 0
	}).MustBoolSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupBool(t *testing.T) {
	m := objx.Map{"data": []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	grouped := m.Get("data").GroupBool(func(i int, val bool) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]bool)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceBool(t *testing.T) {
	m := objx.Map{"data": []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}
	rawArr := m.Get("data").MustBoolSlice()

	replaced := m.Get("data").ReplaceBool(func(index int, val bool) bool {
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
	m := objx.Map{"data": []bool{bool(true), bool(true), bool(true), bool(true), bool(true), bool(true)}}

	collected := m.Get("data").CollectBool(func(index int, val bool) interface{} {
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

/*
   Tests for Str (string and []string)
*/
func TestStr(t *testing.T) {
	val := string("hello")
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Str())
	assert.Equal(t, val, m.Get("value").MustStr())
	assert.Equal(t, string(""), m.Get("nothing").Str())
	assert.Equal(t, val, m.Get("nothing").Str("hello"))
	assert.Panics(t, func() {
		m.Get("age").MustStr()
	})
}

func TestStrSlice(t *testing.T) {
	val := string("hello")
	m := objx.Map{"value": []string{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").StrSlice()[0])
	assert.Equal(t, val, m.Get("value").MustStrSlice()[0])
	assert.Equal(t, []string(nil), m.Get("nothing").StrSlice())
	assert.Equal(t, val, m.Get("nothing").StrSlice([]string{string("hello")})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustStrSlice()
	})
}

func TestIsStr(t *testing.T) {
	m := objx.Map{"data": string("hello")}

	assert.True(t, m.Get("data").IsStr())
}

func TestIsStrSlice(t *testing.T) {
	m := objx.Map{"data": []string{string("hello")}}

	assert.True(t, m.Get("data").IsStrSlice())
}

func TestEachStr(t *testing.T) {
	m := objx.Map{"data": []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}
	count := 0
	replacedVals := make([]string, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachStr(func(i int, val string) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustStrSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustStrSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustStrSlice()[2])
}

func TestWhereStr(t *testing.T) {
	m := objx.Map{"data": []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	selected := m.Get("data").WhereStr(func(i int, val string) bool {
		return i%2 == 0
	}).MustStrSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupStr(t *testing.T) {
	m := objx.Map{"data": []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	grouped := m.Get("data").GroupStr(func(i int, val string) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]string)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceStr(t *testing.T) {
	m := objx.Map{"data": []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}
	rawArr := m.Get("data").MustStrSlice()

	replaced := m.Get("data").ReplaceStr(func(index int, val string) string {
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
	m := objx.Map{"data": []string{string("hello"), string("hello"), string("hello"), string("hello"), string("hello"), string("hello")}}

	collected := m.Get("data").CollectStr(func(index int, val string) interface{} {
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

/*
   Tests for Int (int and []int)
*/
func TestInt(t *testing.T) {
	val := int(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int())
	assert.Equal(t, val, m.Get("value").MustInt())
	assert.Equal(t, int(0), m.Get("nothing").Int())
	assert.Equal(t, val, m.Get("nothing").Int(1))
	assert.Panics(t, func() {
		m.Get("age").MustInt()
	})
}

func TestIntSlice(t *testing.T) {
	val := int(1)
	m := objx.Map{"value": []int{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").IntSlice()[0])
	assert.Equal(t, val, m.Get("value").MustIntSlice()[0])
	assert.Equal(t, []int(nil), m.Get("nothing").IntSlice())
	assert.Equal(t, val, m.Get("nothing").IntSlice([]int{int(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustIntSlice()
	})
}

func TestIsInt(t *testing.T) {
	m := objx.Map{"data": int(1)}

	assert.True(t, m.Get("data").IsInt())
}

func TestIsIntSlice(t *testing.T) {
	m := objx.Map{"data": []int{int(1)}}

	assert.True(t, m.Get("data").IsIntSlice())
}

func TestEachInt(t *testing.T) {
	m := objx.Map{"data": []int{int(1), int(1), int(1), int(1), int(1)}}
	count := 0
	replacedVals := make([]int, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInt(func(i int, val int) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustIntSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustIntSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustIntSlice()[2])
}

func TestWhereInt(t *testing.T) {
	m := objx.Map{"data": []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	selected := m.Get("data").WhereInt(func(i int, val int) bool {
		return i%2 == 0
	}).MustIntSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInt(t *testing.T) {
	m := objx.Map{"data": []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	grouped := m.Get("data").GroupInt(func(i int, val int) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]int)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInt(t *testing.T) {
	m := objx.Map{"data": []int{int(1), int(1), int(1), int(1), int(1), int(1)}}
	rawArr := m.Get("data").MustIntSlice()

	replaced := m.Get("data").ReplaceInt(func(index int, val int) int {
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
	m := objx.Map{"data": []int{int(1), int(1), int(1), int(1), int(1), int(1)}}

	collected := m.Get("data").CollectInt(func(index int, val int) interface{} {
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

/*
   Tests for Int8 (int8 and []int8)
*/
func TestInt8(t *testing.T) {
	val := int8(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int8())
	assert.Equal(t, val, m.Get("value").MustInt8())
	assert.Equal(t, int8(0), m.Get("nothing").Int8())
	assert.Equal(t, val, m.Get("nothing").Int8(1))
	assert.Panics(t, func() {
		m.Get("age").MustInt8()
	})
}

func TestInt8Slice(t *testing.T) {
	val := int8(1)
	m := objx.Map{"value": []int8{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int8Slice()[0])
	assert.Equal(t, val, m.Get("value").MustInt8Slice()[0])
	assert.Equal(t, []int8(nil), m.Get("nothing").Int8Slice())
	assert.Equal(t, val, m.Get("nothing").Int8Slice([]int8{int8(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustInt8Slice()
	})
}

func TestIsInt8(t *testing.T) {
	m := objx.Map{"data": int8(1)}

	assert.True(t, m.Get("data").IsInt8())
}

func TestIsInt8Slice(t *testing.T) {
	m := objx.Map{"data": []int8{int8(1)}}

	assert.True(t, m.Get("data").IsInt8Slice())
}

func TestEachInt8(t *testing.T) {
	m := objx.Map{"data": []int8{int8(1), int8(1), int8(1), int8(1), int8(1)}}
	count := 0
	replacedVals := make([]int8, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInt8(func(i int, val int8) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustInt8Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustInt8Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustInt8Slice()[2])
}

func TestWhereInt8(t *testing.T) {
	m := objx.Map{"data": []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	selected := m.Get("data").WhereInt8(func(i int, val int8) bool {
		return i%2 == 0
	}).MustInt8Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInt8(t *testing.T) {
	m := objx.Map{"data": []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	grouped := m.Get("data").GroupInt8(func(i int, val int8) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]int8)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInt8(t *testing.T) {
	m := objx.Map{"data": []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}
	rawArr := m.Get("data").MustInt8Slice()

	replaced := m.Get("data").ReplaceInt8(func(index int, val int8) int8 {
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
	m := objx.Map{"data": []int8{int8(1), int8(1), int8(1), int8(1), int8(1), int8(1)}}

	collected := m.Get("data").CollectInt8(func(index int, val int8) interface{} {
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

/*
   Tests for Int16 (int16 and []int16)
*/
func TestInt16(t *testing.T) {
	val := int16(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int16())
	assert.Equal(t, val, m.Get("value").MustInt16())
	assert.Equal(t, int16(0), m.Get("nothing").Int16())
	assert.Equal(t, val, m.Get("nothing").Int16(1))
	assert.Panics(t, func() {
		m.Get("age").MustInt16()
	})
}

func TestInt16Slice(t *testing.T) {
	val := int16(1)
	m := objx.Map{"value": []int16{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int16Slice()[0])
	assert.Equal(t, val, m.Get("value").MustInt16Slice()[0])
	assert.Equal(t, []int16(nil), m.Get("nothing").Int16Slice())
	assert.Equal(t, val, m.Get("nothing").Int16Slice([]int16{int16(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustInt16Slice()
	})
}

func TestIsInt16(t *testing.T) {
	m := objx.Map{"data": int16(1)}

	assert.True(t, m.Get("data").IsInt16())
}

func TestIsInt16Slice(t *testing.T) {
	m := objx.Map{"data": []int16{int16(1)}}

	assert.True(t, m.Get("data").IsInt16Slice())
}

func TestEachInt16(t *testing.T) {
	m := objx.Map{"data": []int16{int16(1), int16(1), int16(1), int16(1), int16(1)}}
	count := 0
	replacedVals := make([]int16, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInt16(func(i int, val int16) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustInt16Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustInt16Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustInt16Slice()[2])
}

func TestWhereInt16(t *testing.T) {
	m := objx.Map{"data": []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	selected := m.Get("data").WhereInt16(func(i int, val int16) bool {
		return i%2 == 0
	}).MustInt16Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInt16(t *testing.T) {
	m := objx.Map{"data": []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	grouped := m.Get("data").GroupInt16(func(i int, val int16) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]int16)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInt16(t *testing.T) {
	m := objx.Map{"data": []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}
	rawArr := m.Get("data").MustInt16Slice()

	replaced := m.Get("data").ReplaceInt16(func(index int, val int16) int16 {
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
	m := objx.Map{"data": []int16{int16(1), int16(1), int16(1), int16(1), int16(1), int16(1)}}

	collected := m.Get("data").CollectInt16(func(index int, val int16) interface{} {
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

/*
   Tests for Int32 (int32 and []int32)
*/
func TestInt32(t *testing.T) {
	val := int32(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int32())
	assert.Equal(t, val, m.Get("value").MustInt32())
	assert.Equal(t, int32(0), m.Get("nothing").Int32())
	assert.Equal(t, val, m.Get("nothing").Int32(1))
	assert.Panics(t, func() {
		m.Get("age").MustInt32()
	})
}

func TestInt32Slice(t *testing.T) {
	val := int32(1)
	m := objx.Map{"value": []int32{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int32Slice()[0])
	assert.Equal(t, val, m.Get("value").MustInt32Slice()[0])
	assert.Equal(t, []int32(nil), m.Get("nothing").Int32Slice())
	assert.Equal(t, val, m.Get("nothing").Int32Slice([]int32{int32(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustInt32Slice()
	})
}

func TestIsInt32(t *testing.T) {
	m := objx.Map{"data": int32(1)}

	assert.True(t, m.Get("data").IsInt32())
}

func TestIsInt32Slice(t *testing.T) {
	m := objx.Map{"data": []int32{int32(1)}}

	assert.True(t, m.Get("data").IsInt32Slice())
}

func TestEachInt32(t *testing.T) {
	m := objx.Map{"data": []int32{int32(1), int32(1), int32(1), int32(1), int32(1)}}
	count := 0
	replacedVals := make([]int32, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInt32(func(i int, val int32) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustInt32Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustInt32Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustInt32Slice()[2])
}

func TestWhereInt32(t *testing.T) {
	m := objx.Map{"data": []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	selected := m.Get("data").WhereInt32(func(i int, val int32) bool {
		return i%2 == 0
	}).MustInt32Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInt32(t *testing.T) {
	m := objx.Map{"data": []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	grouped := m.Get("data").GroupInt32(func(i int, val int32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]int32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInt32(t *testing.T) {
	m := objx.Map{"data": []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}
	rawArr := m.Get("data").MustInt32Slice()

	replaced := m.Get("data").ReplaceInt32(func(index int, val int32) int32 {
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
	m := objx.Map{"data": []int32{int32(1), int32(1), int32(1), int32(1), int32(1), int32(1)}}

	collected := m.Get("data").CollectInt32(func(index int, val int32) interface{} {
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

/*
   Tests for Int64 (int64 and []int64)
*/
func TestInt64(t *testing.T) {
	val := int64(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int64())
	assert.Equal(t, val, m.Get("value").MustInt64())
	assert.Equal(t, int64(0), m.Get("nothing").Int64())
	assert.Equal(t, val, m.Get("nothing").Int64(1))
	assert.Panics(t, func() {
		m.Get("age").MustInt64()
	})
}

func TestInt64Slice(t *testing.T) {
	val := int64(1)
	m := objx.Map{"value": []int64{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Int64Slice()[0])
	assert.Equal(t, val, m.Get("value").MustInt64Slice()[0])
	assert.Equal(t, []int64(nil), m.Get("nothing").Int64Slice())
	assert.Equal(t, val, m.Get("nothing").Int64Slice([]int64{int64(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustInt64Slice()
	})
}

func TestIsInt64(t *testing.T) {
	m := objx.Map{"data": int64(1)}

	assert.True(t, m.Get("data").IsInt64())
}

func TestIsInt64Slice(t *testing.T) {
	m := objx.Map{"data": []int64{int64(1)}}

	assert.True(t, m.Get("data").IsInt64Slice())
}

func TestEachInt64(t *testing.T) {
	m := objx.Map{"data": []int64{int64(1), int64(1), int64(1), int64(1), int64(1)}}
	count := 0
	replacedVals := make([]int64, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachInt64(func(i int, val int64) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustInt64Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustInt64Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustInt64Slice()[2])
}

func TestWhereInt64(t *testing.T) {
	m := objx.Map{"data": []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	selected := m.Get("data").WhereInt64(func(i int, val int64) bool {
		return i%2 == 0
	}).MustInt64Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupInt64(t *testing.T) {
	m := objx.Map{"data": []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	grouped := m.Get("data").GroupInt64(func(i int, val int64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]int64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceInt64(t *testing.T) {
	m := objx.Map{"data": []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}
	rawArr := m.Get("data").MustInt64Slice()

	replaced := m.Get("data").ReplaceInt64(func(index int, val int64) int64 {
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
	m := objx.Map{"data": []int64{int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)}}

	collected := m.Get("data").CollectInt64(func(index int, val int64) interface{} {
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

/*
   Tests for Uint (uint and []uint)
*/
func TestUint(t *testing.T) {
	val := uint(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint())
	assert.Equal(t, val, m.Get("value").MustUint())
	assert.Equal(t, uint(0), m.Get("nothing").Uint())
	assert.Equal(t, val, m.Get("nothing").Uint(1))
	assert.Panics(t, func() {
		m.Get("age").MustUint()
	})
}

func TestUintSlice(t *testing.T) {
	val := uint(1)
	m := objx.Map{"value": []uint{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").UintSlice()[0])
	assert.Equal(t, val, m.Get("value").MustUintSlice()[0])
	assert.Equal(t, []uint(nil), m.Get("nothing").UintSlice())
	assert.Equal(t, val, m.Get("nothing").UintSlice([]uint{uint(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUintSlice()
	})
}

func TestIsUint(t *testing.T) {
	m := objx.Map{"data": uint(1)}

	assert.True(t, m.Get("data").IsUint())
}

func TestIsUintSlice(t *testing.T) {
	m := objx.Map{"data": []uint{uint(1)}}

	assert.True(t, m.Get("data").IsUintSlice())
}

func TestEachUint(t *testing.T) {
	m := objx.Map{"data": []uint{uint(1), uint(1), uint(1), uint(1), uint(1)}}
	count := 0
	replacedVals := make([]uint, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUint(func(i int, val uint) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUintSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUintSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUintSlice()[2])
}

func TestWhereUint(t *testing.T) {
	m := objx.Map{"data": []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	selected := m.Get("data").WhereUint(func(i int, val uint) bool {
		return i%2 == 0
	}).MustUintSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUint(t *testing.T) {
	m := objx.Map{"data": []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	grouped := m.Get("data").GroupUint(func(i int, val uint) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uint)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUint(t *testing.T) {
	m := objx.Map{"data": []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}
	rawArr := m.Get("data").MustUintSlice()

	replaced := m.Get("data").ReplaceUint(func(index int, val uint) uint {
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
	m := objx.Map{"data": []uint{uint(1), uint(1), uint(1), uint(1), uint(1), uint(1)}}

	collected := m.Get("data").CollectUint(func(index int, val uint) interface{} {
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

/*
   Tests for Uint8 (uint8 and []uint8)
*/
func TestUint8(t *testing.T) {
	val := uint8(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint8())
	assert.Equal(t, val, m.Get("value").MustUint8())
	assert.Equal(t, uint8(0), m.Get("nothing").Uint8())
	assert.Equal(t, val, m.Get("nothing").Uint8(1))
	assert.Panics(t, func() {
		m.Get("age").MustUint8()
	})
}

func TestUint8Slice(t *testing.T) {
	val := uint8(1)
	m := objx.Map{"value": []uint8{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint8Slice()[0])
	assert.Equal(t, val, m.Get("value").MustUint8Slice()[0])
	assert.Equal(t, []uint8(nil), m.Get("nothing").Uint8Slice())
	assert.Equal(t, val, m.Get("nothing").Uint8Slice([]uint8{uint8(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUint8Slice()
	})
}

func TestIsUint8(t *testing.T) {
	m := objx.Map{"data": uint8(1)}

	assert.True(t, m.Get("data").IsUint8())
}

func TestIsUint8Slice(t *testing.T) {
	m := objx.Map{"data": []uint8{uint8(1)}}

	assert.True(t, m.Get("data").IsUint8Slice())
}

func TestEachUint8(t *testing.T) {
	m := objx.Map{"data": []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}
	count := 0
	replacedVals := make([]uint8, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUint8(func(i int, val uint8) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUint8Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUint8Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUint8Slice()[2])
}

func TestWhereUint8(t *testing.T) {
	m := objx.Map{"data": []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	selected := m.Get("data").WhereUint8(func(i int, val uint8) bool {
		return i%2 == 0
	}).MustUint8Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUint8(t *testing.T) {
	m := objx.Map{"data": []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	grouped := m.Get("data").GroupUint8(func(i int, val uint8) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uint8)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUint8(t *testing.T) {
	m := objx.Map{"data": []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}
	rawArr := m.Get("data").MustUint8Slice()

	replaced := m.Get("data").ReplaceUint8(func(index int, val uint8) uint8 {
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
	m := objx.Map{"data": []uint8{uint8(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(1)}}

	collected := m.Get("data").CollectUint8(func(index int, val uint8) interface{} {
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

/*
   Tests for Uint16 (uint16 and []uint16)
*/
func TestUint16(t *testing.T) {
	val := uint16(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint16())
	assert.Equal(t, val, m.Get("value").MustUint16())
	assert.Equal(t, uint16(0), m.Get("nothing").Uint16())
	assert.Equal(t, val, m.Get("nothing").Uint16(1))
	assert.Panics(t, func() {
		m.Get("age").MustUint16()
	})
}

func TestUint16Slice(t *testing.T) {
	val := uint16(1)
	m := objx.Map{"value": []uint16{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint16Slice()[0])
	assert.Equal(t, val, m.Get("value").MustUint16Slice()[0])
	assert.Equal(t, []uint16(nil), m.Get("nothing").Uint16Slice())
	assert.Equal(t, val, m.Get("nothing").Uint16Slice([]uint16{uint16(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUint16Slice()
	})
}

func TestIsUint16(t *testing.T) {
	m := objx.Map{"data": uint16(1)}

	assert.True(t, m.Get("data").IsUint16())
}

func TestIsUint16Slice(t *testing.T) {
	m := objx.Map{"data": []uint16{uint16(1)}}

	assert.True(t, m.Get("data").IsUint16Slice())
}

func TestEachUint16(t *testing.T) {
	m := objx.Map{"data": []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}
	count := 0
	replacedVals := make([]uint16, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUint16(func(i int, val uint16) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUint16Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUint16Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUint16Slice()[2])
}

func TestWhereUint16(t *testing.T) {
	m := objx.Map{"data": []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	selected := m.Get("data").WhereUint16(func(i int, val uint16) bool {
		return i%2 == 0
	}).MustUint16Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUint16(t *testing.T) {
	m := objx.Map{"data": []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	grouped := m.Get("data").GroupUint16(func(i int, val uint16) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uint16)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUint16(t *testing.T) {
	m := objx.Map{"data": []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}
	rawArr := m.Get("data").MustUint16Slice()

	replaced := m.Get("data").ReplaceUint16(func(index int, val uint16) uint16 {
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
	m := objx.Map{"data": []uint16{uint16(1), uint16(1), uint16(1), uint16(1), uint16(1), uint16(1)}}

	collected := m.Get("data").CollectUint16(func(index int, val uint16) interface{} {
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

/*
   Tests for Uint32 (uint32 and []uint32)
*/
func TestUint32(t *testing.T) {
	val := uint32(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint32())
	assert.Equal(t, val, m.Get("value").MustUint32())
	assert.Equal(t, uint32(0), m.Get("nothing").Uint32())
	assert.Equal(t, val, m.Get("nothing").Uint32(1))
	assert.Panics(t, func() {
		m.Get("age").MustUint32()
	})
}

func TestUint32Slice(t *testing.T) {
	val := uint32(1)
	m := objx.Map{"value": []uint32{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint32Slice()[0])
	assert.Equal(t, val, m.Get("value").MustUint32Slice()[0])
	assert.Equal(t, []uint32(nil), m.Get("nothing").Uint32Slice())
	assert.Equal(t, val, m.Get("nothing").Uint32Slice([]uint32{uint32(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUint32Slice()
	})
}

func TestIsUint32(t *testing.T) {
	m := objx.Map{"data": uint32(1)}

	assert.True(t, m.Get("data").IsUint32())
}

func TestIsUint32Slice(t *testing.T) {
	m := objx.Map{"data": []uint32{uint32(1)}}

	assert.True(t, m.Get("data").IsUint32Slice())
}

func TestEachUint32(t *testing.T) {
	m := objx.Map{"data": []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}
	count := 0
	replacedVals := make([]uint32, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUint32(func(i int, val uint32) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUint32Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUint32Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUint32Slice()[2])
}

func TestWhereUint32(t *testing.T) {
	m := objx.Map{"data": []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	selected := m.Get("data").WhereUint32(func(i int, val uint32) bool {
		return i%2 == 0
	}).MustUint32Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUint32(t *testing.T) {
	m := objx.Map{"data": []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	grouped := m.Get("data").GroupUint32(func(i int, val uint32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uint32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUint32(t *testing.T) {
	m := objx.Map{"data": []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}
	rawArr := m.Get("data").MustUint32Slice()

	replaced := m.Get("data").ReplaceUint32(func(index int, val uint32) uint32 {
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
	m := objx.Map{"data": []uint32{uint32(1), uint32(1), uint32(1), uint32(1), uint32(1), uint32(1)}}

	collected := m.Get("data").CollectUint32(func(index int, val uint32) interface{} {
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

/*
   Tests for Uint64 (uint64 and []uint64)
*/
func TestUint64(t *testing.T) {
	val := uint64(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint64())
	assert.Equal(t, val, m.Get("value").MustUint64())
	assert.Equal(t, uint64(0), m.Get("nothing").Uint64())
	assert.Equal(t, val, m.Get("nothing").Uint64(1))
	assert.Panics(t, func() {
		m.Get("age").MustUint64()
	})
}

func TestUint64Slice(t *testing.T) {
	val := uint64(1)
	m := objx.Map{"value": []uint64{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uint64Slice()[0])
	assert.Equal(t, val, m.Get("value").MustUint64Slice()[0])
	assert.Equal(t, []uint64(nil), m.Get("nothing").Uint64Slice())
	assert.Equal(t, val, m.Get("nothing").Uint64Slice([]uint64{uint64(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUint64Slice()
	})
}

func TestIsUint64(t *testing.T) {
	m := objx.Map{"data": uint64(1)}

	assert.True(t, m.Get("data").IsUint64())
}

func TestIsUint64Slice(t *testing.T) {
	m := objx.Map{"data": []uint64{uint64(1)}}

	assert.True(t, m.Get("data").IsUint64Slice())
}

func TestEachUint64(t *testing.T) {
	m := objx.Map{"data": []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}
	count := 0
	replacedVals := make([]uint64, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUint64(func(i int, val uint64) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUint64Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUint64Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUint64Slice()[2])
}

func TestWhereUint64(t *testing.T) {
	m := objx.Map{"data": []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	selected := m.Get("data").WhereUint64(func(i int, val uint64) bool {
		return i%2 == 0
	}).MustUint64Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUint64(t *testing.T) {
	m := objx.Map{"data": []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	grouped := m.Get("data").GroupUint64(func(i int, val uint64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uint64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUint64(t *testing.T) {
	m := objx.Map{"data": []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}
	rawArr := m.Get("data").MustUint64Slice()

	replaced := m.Get("data").ReplaceUint64(func(index int, val uint64) uint64 {
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
	m := objx.Map{"data": []uint64{uint64(1), uint64(1), uint64(1), uint64(1), uint64(1), uint64(1)}}

	collected := m.Get("data").CollectUint64(func(index int, val uint64) interface{} {
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

/*
   Tests for Uintptr (uintptr and []uintptr)
*/
func TestUintptr(t *testing.T) {
	val := uintptr(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Uintptr())
	assert.Equal(t, val, m.Get("value").MustUintptr())
	assert.Equal(t, uintptr(0), m.Get("nothing").Uintptr())
	assert.Equal(t, val, m.Get("nothing").Uintptr(1))
	assert.Panics(t, func() {
		m.Get("age").MustUintptr()
	})
}

func TestUintptrSlice(t *testing.T) {
	val := uintptr(1)
	m := objx.Map{"value": []uintptr{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").UintptrSlice()[0])
	assert.Equal(t, val, m.Get("value").MustUintptrSlice()[0])
	assert.Equal(t, []uintptr(nil), m.Get("nothing").UintptrSlice())
	assert.Equal(t, val, m.Get("nothing").UintptrSlice([]uintptr{uintptr(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustUintptrSlice()
	})
}

func TestIsUintptr(t *testing.T) {
	m := objx.Map{"data": uintptr(1)}

	assert.True(t, m.Get("data").IsUintptr())
}

func TestIsUintptrSlice(t *testing.T) {
	m := objx.Map{"data": []uintptr{uintptr(1)}}

	assert.True(t, m.Get("data").IsUintptrSlice())
}

func TestEachUintptr(t *testing.T) {
	m := objx.Map{"data": []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}
	count := 0
	replacedVals := make([]uintptr, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachUintptr(func(i int, val uintptr) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustUintptrSlice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustUintptrSlice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustUintptrSlice()[2])
}

func TestWhereUintptr(t *testing.T) {
	m := objx.Map{"data": []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	selected := m.Get("data").WhereUintptr(func(i int, val uintptr) bool {
		return i%2 == 0
	}).MustUintptrSlice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupUintptr(t *testing.T) {
	m := objx.Map{"data": []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	grouped := m.Get("data").GroupUintptr(func(i int, val uintptr) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]uintptr)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceUintptr(t *testing.T) {
	m := objx.Map{"data": []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}
	rawArr := m.Get("data").MustUintptrSlice()

	replaced := m.Get("data").ReplaceUintptr(func(index int, val uintptr) uintptr {
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
	m := objx.Map{"data": []uintptr{uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1), uintptr(1)}}

	collected := m.Get("data").CollectUintptr(func(index int, val uintptr) interface{} {
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

/*
   Tests for Float32 (float32 and []float32)
*/
func TestFloat32(t *testing.T) {
	val := float32(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Float32())
	assert.Equal(t, val, m.Get("value").MustFloat32())
	assert.Equal(t, float32(0), m.Get("nothing").Float32())
	assert.Equal(t, val, m.Get("nothing").Float32(1))
	assert.Panics(t, func() {
		m.Get("age").MustFloat32()
	})
}

func TestFloat32Slice(t *testing.T) {
	val := float32(1)
	m := objx.Map{"value": []float32{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Float32Slice()[0])
	assert.Equal(t, val, m.Get("value").MustFloat32Slice()[0])
	assert.Equal(t, []float32(nil), m.Get("nothing").Float32Slice())
	assert.Equal(t, val, m.Get("nothing").Float32Slice([]float32{float32(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustFloat32Slice()
	})
}

func TestIsFloat32(t *testing.T) {
	m := objx.Map{"data": float32(1)}

	assert.True(t, m.Get("data").IsFloat32())
}

func TestIsFloat32Slice(t *testing.T) {
	m := objx.Map{"data": []float32{float32(1)}}

	assert.True(t, m.Get("data").IsFloat32Slice())
}

func TestEachFloat32(t *testing.T) {
	m := objx.Map{"data": []float32{float32(1), float32(1), float32(1), float32(1), float32(1)}}
	count := 0
	replacedVals := make([]float32, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachFloat32(func(i int, val float32) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustFloat32Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustFloat32Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustFloat32Slice()[2])
}

func TestWhereFloat32(t *testing.T) {
	m := objx.Map{"data": []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	selected := m.Get("data").WhereFloat32(func(i int, val float32) bool {
		return i%2 == 0
	}).MustFloat32Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupFloat32(t *testing.T) {
	m := objx.Map{"data": []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	grouped := m.Get("data").GroupFloat32(func(i int, val float32) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]float32)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceFloat32(t *testing.T) {
	m := objx.Map{"data": []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}
	rawArr := m.Get("data").MustFloat32Slice()

	replaced := m.Get("data").ReplaceFloat32(func(index int, val float32) float32 {
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
	m := objx.Map{"data": []float32{float32(1), float32(1), float32(1), float32(1), float32(1), float32(1)}}

	collected := m.Get("data").CollectFloat32(func(index int, val float32) interface{} {
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

/*
   Tests for Float64 (float64 and []float64)
*/
func TestFloat64(t *testing.T) {
	val := float64(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Float64())
	assert.Equal(t, val, m.Get("value").MustFloat64())
	assert.Equal(t, float64(0), m.Get("nothing").Float64())
	assert.Equal(t, val, m.Get("nothing").Float64(1))
	assert.Panics(t, func() {
		m.Get("age").MustFloat64()
	})
}

func TestFloat64Slice(t *testing.T) {
	val := float64(1)
	m := objx.Map{"value": []float64{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Float64Slice()[0])
	assert.Equal(t, val, m.Get("value").MustFloat64Slice()[0])
	assert.Equal(t, []float64(nil), m.Get("nothing").Float64Slice())
	assert.Equal(t, val, m.Get("nothing").Float64Slice([]float64{float64(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustFloat64Slice()
	})
}

func TestIsFloat64(t *testing.T) {
	m := objx.Map{"data": float64(1)}

	assert.True(t, m.Get("data").IsFloat64())
}

func TestIsFloat64Slice(t *testing.T) {
	m := objx.Map{"data": []float64{float64(1)}}

	assert.True(t, m.Get("data").IsFloat64Slice())
}

func TestEachFloat64(t *testing.T) {
	m := objx.Map{"data": []float64{float64(1), float64(1), float64(1), float64(1), float64(1)}}
	count := 0
	replacedVals := make([]float64, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachFloat64(func(i int, val float64) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustFloat64Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustFloat64Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustFloat64Slice()[2])
}

func TestWhereFloat64(t *testing.T) {
	m := objx.Map{"data": []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	selected := m.Get("data").WhereFloat64(func(i int, val float64) bool {
		return i%2 == 0
	}).MustFloat64Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupFloat64(t *testing.T) {
	m := objx.Map{"data": []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	grouped := m.Get("data").GroupFloat64(func(i int, val float64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]float64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceFloat64(t *testing.T) {
	m := objx.Map{"data": []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}
	rawArr := m.Get("data").MustFloat64Slice()

	replaced := m.Get("data").ReplaceFloat64(func(index int, val float64) float64 {
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
	m := objx.Map{"data": []float64{float64(1), float64(1), float64(1), float64(1), float64(1), float64(1)}}

	collected := m.Get("data").CollectFloat64(func(index int, val float64) interface{} {
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

/*
   Tests for Complex64 (complex64 and []complex64)
*/
func TestComplex64(t *testing.T) {
	val := complex64(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Complex64())
	assert.Equal(t, val, m.Get("value").MustComplex64())
	assert.Equal(t, complex64(0), m.Get("nothing").Complex64())
	assert.Equal(t, val, m.Get("nothing").Complex64(1))
	assert.Panics(t, func() {
		m.Get("age").MustComplex64()
	})
}

func TestComplex64Slice(t *testing.T) {
	val := complex64(1)
	m := objx.Map{"value": []complex64{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Complex64Slice()[0])
	assert.Equal(t, val, m.Get("value").MustComplex64Slice()[0])
	assert.Equal(t, []complex64(nil), m.Get("nothing").Complex64Slice())
	assert.Equal(t, val, m.Get("nothing").Complex64Slice([]complex64{complex64(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustComplex64Slice()
	})
}

func TestIsComplex64(t *testing.T) {
	m := objx.Map{"data": complex64(1)}

	assert.True(t, m.Get("data").IsComplex64())
}

func TestIsComplex64Slice(t *testing.T) {
	m := objx.Map{"data": []complex64{complex64(1)}}

	assert.True(t, m.Get("data").IsComplex64Slice())
}

func TestEachComplex64(t *testing.T) {
	m := objx.Map{"data": []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}
	count := 0
	replacedVals := make([]complex64, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachComplex64(func(i int, val complex64) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustComplex64Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustComplex64Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustComplex64Slice()[2])
}

func TestWhereComplex64(t *testing.T) {
	m := objx.Map{"data": []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	selected := m.Get("data").WhereComplex64(func(i int, val complex64) bool {
		return i%2 == 0
	}).MustComplex64Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupComplex64(t *testing.T) {
	m := objx.Map{"data": []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	grouped := m.Get("data").GroupComplex64(func(i int, val complex64) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]complex64)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceComplex64(t *testing.T) {
	m := objx.Map{"data": []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}
	rawArr := m.Get("data").MustComplex64Slice()

	replaced := m.Get("data").ReplaceComplex64(func(index int, val complex64) complex64 {
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
	m := objx.Map{"data": []complex64{complex64(1), complex64(1), complex64(1), complex64(1), complex64(1), complex64(1)}}

	collected := m.Get("data").CollectComplex64(func(index int, val complex64) interface{} {
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

/*
   Tests for Complex128 (complex128 and []complex128)
*/
func TestComplex128(t *testing.T) {
	val := complex128(1)
	m := objx.Map{"value": val, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Complex128())
	assert.Equal(t, val, m.Get("value").MustComplex128())
	assert.Equal(t, complex128(0), m.Get("nothing").Complex128())
	assert.Equal(t, val, m.Get("nothing").Complex128(1))
	assert.Panics(t, func() {
		m.Get("age").MustComplex128()
	})
}

func TestComplex128Slice(t *testing.T) {
	val := complex128(1)
	m := objx.Map{"value": []complex128{val}, "nothing": nil}

	assert.Equal(t, val, m.Get("value").Complex128Slice()[0])
	assert.Equal(t, val, m.Get("value").MustComplex128Slice()[0])
	assert.Equal(t, []complex128(nil), m.Get("nothing").Complex128Slice())
	assert.Equal(t, val, m.Get("nothing").Complex128Slice([]complex128{complex128(1)})[0])
	assert.Panics(t, func() {
		m.Get("nothing").MustComplex128Slice()
	})
}

func TestIsComplex128(t *testing.T) {
	m := objx.Map{"data": complex128(1)}

	assert.True(t, m.Get("data").IsComplex128())
}

func TestIsComplex128Slice(t *testing.T) {
	m := objx.Map{"data": []complex128{complex128(1)}}

	assert.True(t, m.Get("data").IsComplex128Slice())
}

func TestEachComplex128(t *testing.T) {
	m := objx.Map{"data": []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}
	count := 0
	replacedVals := make([]complex128, 0)
	assert.Equal(t, m.Get("data"), m.Get("data").EachComplex128(func(i int, val complex128) bool {
		count++
		replacedVals = append(replacedVals, val)

		// abort early
		return i != 2
	}))

	assert.Equal(t, count, 3)
	assert.Equal(t, replacedVals[0], m.Get("data").MustComplex128Slice()[0])
	assert.Equal(t, replacedVals[1], m.Get("data").MustComplex128Slice()[1])
	assert.Equal(t, replacedVals[2], m.Get("data").MustComplex128Slice()[2])
}

func TestWhereComplex128(t *testing.T) {
	m := objx.Map{"data": []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	selected := m.Get("data").WhereComplex128(func(i int, val complex128) bool {
		return i%2 == 0
	}).MustComplex128Slice()

	assert.Equal(t, 3, len(selected))
}

func TestGroupComplex128(t *testing.T) {
	m := objx.Map{"data": []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	grouped := m.Get("data").GroupComplex128(func(i int, val complex128) string {
		return fmt.Sprintf("%v", i%2 == 0)
	}).Data().(map[string][]complex128)

	assert.Equal(t, 2, len(grouped))
	assert.Equal(t, 3, len(grouped["true"]))
	assert.Equal(t, 3, len(grouped["false"]))
}

func TestReplaceComplex128(t *testing.T) {
	m := objx.Map{"data": []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}
	rawArr := m.Get("data").MustComplex128Slice()

	replaced := m.Get("data").ReplaceComplex128(func(index int, val complex128) complex128 {
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
	m := objx.Map{"data": []complex128{complex128(1), complex128(1), complex128(1), complex128(1), complex128(1), complex128(1)}}

	collected := m.Get("data").CollectComplex128(func(index int, val complex128) interface{} {
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
