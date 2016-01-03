package objx

/*
	Inter (interface{} and []interface{})
	--------------------------------------------------
*/

// Inter gets the value as a interface{}, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Inter(optionalDefault ...interface{}) interface{} {
	if s, ok := v.data.(interface{}); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInter gets the value as a interface{}.
//
// Panics if the object is not a interface{}.
func (v *Value) MustInter() interface{} {
	return v.data.(interface{})
}

// InterSlice gets the value as a []interface{}, returns the optionalDefault
// value or nil if the value is not a []interface{}.
func (v *Value) InterSlice(optionalDefault ...[]interface{}) []interface{} {
	if s, ok := v.data.([]interface{}); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInterSlice gets the value as a []interface{}.
//
// Panics if the object is not a []interface{}.
func (v *Value) MustInterSlice() []interface{} {
	return v.data.([]interface{})
}

// IsInter gets whether the object contained is a interface{} or not.
func (v *Value) IsInter() bool {
	_, ok := v.data.(interface{})
	return ok
}

// IsInterSlice gets whether the object contained is a []interface{} or not.
func (v *Value) IsInterSlice() bool {
	_, ok := v.data.([]interface{})
	return ok
}

// EachInter calls the specified callback for each object
// in the []interface{}.
//
// Panics if the object is the wrong type.
func (v *Value) EachInter(callback func(int, interface{}) bool) *Value {

	for index, val := range v.MustInterSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInter uses the specified decider function to select items
// from the []interface{}.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInter(decider func(int, interface{}) bool) *Value {

	var selected []interface{}

	v.EachInter(func(index int, val interface{}) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInter uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]interface{}.
func (v *Value) GroupInter(grouper func(int, interface{}) string) *Value {

	groups := make(map[string][]interface{})

	v.EachInter(func(index int, val interface{}) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]interface{}, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInter uses the specified function to replace each interface{}s
// by iterating each item.  The data in the returned result will be a
// []interface{} containing the replaced items.
func (v *Value) ReplaceInter(replacer func(int, interface{}) interface{}) *Value {

	arr := v.MustInterSlice()
	replaced := make([]interface{}, len(arr))

	v.EachInter(func(index int, val interface{}) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInter uses the specified collector function to collect a value
// for each of the interface{}s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInter(collector func(int, interface{}) interface{}) *Value {

	arr := v.MustInterSlice()
	collected := make([]interface{}, len(arr))

	v.EachInter(func(index int, val interface{}) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	MSI (map[string]interface{} and []map[string]interface{})
	--------------------------------------------------
*/

// MSI gets the value as a map[string]interface{}, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) MSI(optionalDefault ...map[string]interface{}) map[string]interface{} {
	if s, ok := v.data.(map[string]interface{}); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustMSI gets the value as a map[string]interface{}.
//
// Panics if the object is not a map[string]interface{}.
func (v *Value) MustMSI() map[string]interface{} {
	return v.data.(map[string]interface{})
}

// MSISlice gets the value as a []map[string]interface{}, returns the optionalDefault
// value or nil if the value is not a []map[string]interface{}.
func (v *Value) MSISlice(optionalDefault ...[]map[string]interface{}) []map[string]interface{} {
	if s, ok := v.data.([]map[string]interface{}); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustMSISlice gets the value as a []map[string]interface{}.
//
// Panics if the object is not a []map[string]interface{}.
func (v *Value) MustMSISlice() []map[string]interface{} {
	return v.data.([]map[string]interface{})
}

// IsMSI gets whether the object contained is a map[string]interface{} or not.
func (v *Value) IsMSI() bool {
	_, ok := v.data.(map[string]interface{})
	return ok
}

// IsMSISlice gets whether the object contained is a []map[string]interface{} or not.
func (v *Value) IsMSISlice() bool {
	_, ok := v.data.([]map[string]interface{})
	return ok
}

// EachMSI calls the specified callback for each object
// in the []map[string]interface{}.
//
// Panics if the object is the wrong type.
func (v *Value) EachMSI(callback func(int, map[string]interface{}) bool) *Value {

	for index, val := range v.MustMSISlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereMSI uses the specified decider function to select items
// from the []map[string]interface{}.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereMSI(decider func(int, map[string]interface{}) bool) *Value {

	var selected []map[string]interface{}

	v.EachMSI(func(index int, val map[string]interface{}) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupMSI uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]map[string]interface{}.
func (v *Value) GroupMSI(grouper func(int, map[string]interface{}) string) *Value {

	groups := make(map[string][]map[string]interface{})

	v.EachMSI(func(index int, val map[string]interface{}) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]map[string]interface{}, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceMSI uses the specified function to replace each map[string]interface{}s
// by iterating each item.  The data in the returned result will be a
// []map[string]interface{} containing the replaced items.
func (v *Value) ReplaceMSI(replacer func(int, map[string]interface{}) map[string]interface{}) *Value {

	arr := v.MustMSISlice()
	replaced := make([]map[string]interface{}, len(arr))

	v.EachMSI(func(index int, val map[string]interface{}) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectMSI uses the specified collector function to collect a value
// for each of the map[string]interface{}s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectMSI(collector func(int, map[string]interface{}) interface{}) *Value {

	arr := v.MustMSISlice()
	collected := make([]interface{}, len(arr))

	v.EachMSI(func(index int, val map[string]interface{}) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	ObjxMap ((Map) and [](Map))
	--------------------------------------------------
*/

// ObjxMap gets the value as a (Map), returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) ObjxMap(optionalDefault ...(Map)) Map {
	if s, ok := v.data.((Map)); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return New(nil)
}

// MustObjxMap gets the value as a (Map).
//
// Panics if the object is not a (Map).
func (v *Value) MustObjxMap() Map {
	return v.data.((Map))
}

// ObjxMapSlice gets the value as a [](Map), returns the optionalDefault
// value or nil if the value is not a [](Map).
func (v *Value) ObjxMapSlice(optionalDefault ...[](Map)) [](Map) {
	if s, ok := v.data.([](Map)); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustObjxMapSlice gets the value as a [](Map).
//
// Panics if the object is not a [](Map).
func (v *Value) MustObjxMapSlice() [](Map) {
	return v.data.([](Map))
}

// IsObjxMap gets whether the object contained is a (Map) or not.
func (v *Value) IsObjxMap() bool {
	_, ok := v.data.((Map))
	return ok
}

// IsObjxMapSlice gets whether the object contained is a [](Map) or not.
func (v *Value) IsObjxMapSlice() bool {
	_, ok := v.data.([](Map))
	return ok
}

// EachObjxMap calls the specified callback for each object
// in the [](Map).
//
// Panics if the object is the wrong type.
func (v *Value) EachObjxMap(callback func(int, Map) bool) *Value {

	for index, val := range v.MustObjxMapSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereObjxMap uses the specified decider function to select items
// from the [](Map).  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereObjxMap(decider func(int, Map) bool) *Value {

	var selected [](Map)

	v.EachObjxMap(func(index int, val Map) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupObjxMap uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][](Map).
func (v *Value) GroupObjxMap(grouper func(int, Map) string) *Value {

	groups := make(map[string][](Map))

	v.EachObjxMap(func(index int, val Map) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([](Map), 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceObjxMap uses the specified function to replace each (Map)s
// by iterating each item.  The data in the returned result will be a
// [](Map) containing the replaced items.
func (v *Value) ReplaceObjxMap(replacer func(int, Map) Map) *Value {

	arr := v.MustObjxMapSlice()
	replaced := make([](Map), len(arr))

	v.EachObjxMap(func(index int, val Map) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectObjxMap uses the specified collector function to collect a value
// for each of the (Map)s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectObjxMap(collector func(int, Map) interface{}) *Value {

	arr := v.MustObjxMapSlice()
	collected := make([]interface{}, len(arr))

	v.EachObjxMap(func(index int, val Map) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Bool (bool and []bool)
	--------------------------------------------------
*/

// Bool gets the value as a bool, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Bool(optionalDefault ...bool) bool {
	if s, ok := v.data.(bool); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return false
}

// MustBool gets the value as a bool.
//
// Panics if the object is not a bool.
func (v *Value) MustBool() bool {
	return v.data.(bool)
}

// BoolSlice gets the value as a []bool, returns the optionalDefault
// value or nil if the value is not a []bool.
func (v *Value) BoolSlice(optionalDefault ...[]bool) []bool {
	if s, ok := v.data.([]bool); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustBoolSlice gets the value as a []bool.
//
// Panics if the object is not a []bool.
func (v *Value) MustBoolSlice() []bool {
	return v.data.([]bool)
}

// IsBool gets whether the object contained is a bool or not.
func (v *Value) IsBool() bool {
	_, ok := v.data.(bool)
	return ok
}

// IsBoolSlice gets whether the object contained is a []bool or not.
func (v *Value) IsBoolSlice() bool {
	_, ok := v.data.([]bool)
	return ok
}

// EachBool calls the specified callback for each object
// in the []bool.
//
// Panics if the object is the wrong type.
func (v *Value) EachBool(callback func(int, bool) bool) *Value {

	for index, val := range v.MustBoolSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereBool uses the specified decider function to select items
// from the []bool.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereBool(decider func(int, bool) bool) *Value {

	var selected []bool

	v.EachBool(func(index int, val bool) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupBool uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]bool.
func (v *Value) GroupBool(grouper func(int, bool) string) *Value {

	groups := make(map[string][]bool)

	v.EachBool(func(index int, val bool) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]bool, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceBool uses the specified function to replace each bools
// by iterating each item.  The data in the returned result will be a
// []bool containing the replaced items.
func (v *Value) ReplaceBool(replacer func(int, bool) bool) *Value {

	arr := v.MustBoolSlice()
	replaced := make([]bool, len(arr))

	v.EachBool(func(index int, val bool) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectBool uses the specified collector function to collect a value
// for each of the bools in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectBool(collector func(int, bool) interface{}) *Value {

	arr := v.MustBoolSlice()
	collected := make([]interface{}, len(arr))

	v.EachBool(func(index int, val bool) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Str (string and []string)
	--------------------------------------------------
*/

// Str gets the value as a string, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Str(optionalDefault ...string) string {
	if s, ok := v.data.(string); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return ""
}

// MustStr gets the value as a string.
//
// Panics if the object is not a string.
func (v *Value) MustStr() string {
	return v.data.(string)
}

// StrSlice gets the value as a []string, returns the optionalDefault
// value or nil if the value is not a []string.
func (v *Value) StrSlice(optionalDefault ...[]string) []string {
	if s, ok := v.data.([]string); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustStrSlice gets the value as a []string.
//
// Panics if the object is not a []string.
func (v *Value) MustStrSlice() []string {
	return v.data.([]string)
}

// IsStr gets whether the object contained is a string or not.
func (v *Value) IsStr() bool {
	_, ok := v.data.(string)
	return ok
}

// IsStrSlice gets whether the object contained is a []string or not.
func (v *Value) IsStrSlice() bool {
	_, ok := v.data.([]string)
	return ok
}

// EachStr calls the specified callback for each object
// in the []string.
//
// Panics if the object is the wrong type.
func (v *Value) EachStr(callback func(int, string) bool) *Value {

	for index, val := range v.MustStrSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereStr uses the specified decider function to select items
// from the []string.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereStr(decider func(int, string) bool) *Value {

	var selected []string

	v.EachStr(func(index int, val string) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupStr uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]string.
func (v *Value) GroupStr(grouper func(int, string) string) *Value {

	groups := make(map[string][]string)

	v.EachStr(func(index int, val string) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]string, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceStr uses the specified function to replace each strings
// by iterating each item.  The data in the returned result will be a
// []string containing the replaced items.
func (v *Value) ReplaceStr(replacer func(int, string) string) *Value {

	arr := v.MustStrSlice()
	replaced := make([]string, len(arr))

	v.EachStr(func(index int, val string) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectStr uses the specified collector function to collect a value
// for each of the strings in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectStr(collector func(int, string) interface{}) *Value {

	arr := v.MustStrSlice()
	collected := make([]interface{}, len(arr))

	v.EachStr(func(index int, val string) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Int (int and []int)
	--------------------------------------------------
*/

// Int gets the value as a int, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Int(optionalDefault ...int) int {
	if s, ok := v.data.(int); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustInt gets the value as a int.
//
// Panics if the object is not a int.
func (v *Value) MustInt() int {
	return v.data.(int)
}

// IntSlice gets the value as a []int, returns the optionalDefault
// value or nil if the value is not a []int.
func (v *Value) IntSlice(optionalDefault ...[]int) []int {
	if s, ok := v.data.([]int); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustIntSlice gets the value as a []int.
//
// Panics if the object is not a []int.
func (v *Value) MustIntSlice() []int {
	return v.data.([]int)
}

// IsInt gets whether the object contained is a int or not.
func (v *Value) IsInt() bool {
	_, ok := v.data.(int)
	return ok
}

// IsIntSlice gets whether the object contained is a []int or not.
func (v *Value) IsIntSlice() bool {
	_, ok := v.data.([]int)
	return ok
}

// EachInt calls the specified callback for each object
// in the []int.
//
// Panics if the object is the wrong type.
func (v *Value) EachInt(callback func(int, int) bool) *Value {

	for index, val := range v.MustIntSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInt uses the specified decider function to select items
// from the []int.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInt(decider func(int, int) bool) *Value {

	var selected []int

	v.EachInt(func(index int, val int) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInt uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]int.
func (v *Value) GroupInt(grouper func(int, int) string) *Value {

	groups := make(map[string][]int)

	v.EachInt(func(index int, val int) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]int, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInt uses the specified function to replace each ints
// by iterating each item.  The data in the returned result will be a
// []int containing the replaced items.
func (v *Value) ReplaceInt(replacer func(int, int) int) *Value {

	arr := v.MustIntSlice()
	replaced := make([]int, len(arr))

	v.EachInt(func(index int, val int) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInt uses the specified collector function to collect a value
// for each of the ints in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInt(collector func(int, int) interface{}) *Value {

	arr := v.MustIntSlice()
	collected := make([]interface{}, len(arr))

	v.EachInt(func(index int, val int) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Int8 (int8 and []int8)
	--------------------------------------------------
*/

// Int8 gets the value as a int8, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Int8(optionalDefault ...int8) int8 {
	if s, ok := v.data.(int8); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustInt8 gets the value as a int8.
//
// Panics if the object is not a int8.
func (v *Value) MustInt8() int8 {
	return v.data.(int8)
}

// Int8Slice gets the value as a []int8, returns the optionalDefault
// value or nil if the value is not a []int8.
func (v *Value) Int8Slice(optionalDefault ...[]int8) []int8 {
	if s, ok := v.data.([]int8); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInt8Slice gets the value as a []int8.
//
// Panics if the object is not a []int8.
func (v *Value) MustInt8Slice() []int8 {
	return v.data.([]int8)
}

// IsInt8 gets whether the object contained is a int8 or not.
func (v *Value) IsInt8() bool {
	_, ok := v.data.(int8)
	return ok
}

// IsInt8Slice gets whether the object contained is a []int8 or not.
func (v *Value) IsInt8Slice() bool {
	_, ok := v.data.([]int8)
	return ok
}

// EachInt8 calls the specified callback for each object
// in the []int8.
//
// Panics if the object is the wrong type.
func (v *Value) EachInt8(callback func(int, int8) bool) *Value {

	for index, val := range v.MustInt8Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInt8 uses the specified decider function to select items
// from the []int8.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInt8(decider func(int, int8) bool) *Value {

	var selected []int8

	v.EachInt8(func(index int, val int8) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInt8 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]int8.
func (v *Value) GroupInt8(grouper func(int, int8) string) *Value {

	groups := make(map[string][]int8)

	v.EachInt8(func(index int, val int8) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]int8, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInt8 uses the specified function to replace each int8s
// by iterating each item.  The data in the returned result will be a
// []int8 containing the replaced items.
func (v *Value) ReplaceInt8(replacer func(int, int8) int8) *Value {

	arr := v.MustInt8Slice()
	replaced := make([]int8, len(arr))

	v.EachInt8(func(index int, val int8) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInt8 uses the specified collector function to collect a value
// for each of the int8s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInt8(collector func(int, int8) interface{}) *Value {

	arr := v.MustInt8Slice()
	collected := make([]interface{}, len(arr))

	v.EachInt8(func(index int, val int8) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Int16 (int16 and []int16)
	--------------------------------------------------
*/

// Int16 gets the value as a int16, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Int16(optionalDefault ...int16) int16 {
	if s, ok := v.data.(int16); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustInt16 gets the value as a int16.
//
// Panics if the object is not a int16.
func (v *Value) MustInt16() int16 {
	return v.data.(int16)
}

// Int16Slice gets the value as a []int16, returns the optionalDefault
// value or nil if the value is not a []int16.
func (v *Value) Int16Slice(optionalDefault ...[]int16) []int16 {
	if s, ok := v.data.([]int16); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInt16Slice gets the value as a []int16.
//
// Panics if the object is not a []int16.
func (v *Value) MustInt16Slice() []int16 {
	return v.data.([]int16)
}

// IsInt16 gets whether the object contained is a int16 or not.
func (v *Value) IsInt16() bool {
	_, ok := v.data.(int16)
	return ok
}

// IsInt16Slice gets whether the object contained is a []int16 or not.
func (v *Value) IsInt16Slice() bool {
	_, ok := v.data.([]int16)
	return ok
}

// EachInt16 calls the specified callback for each object
// in the []int16.
//
// Panics if the object is the wrong type.
func (v *Value) EachInt16(callback func(int, int16) bool) *Value {

	for index, val := range v.MustInt16Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInt16 uses the specified decider function to select items
// from the []int16.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInt16(decider func(int, int16) bool) *Value {

	var selected []int16

	v.EachInt16(func(index int, val int16) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInt16 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]int16.
func (v *Value) GroupInt16(grouper func(int, int16) string) *Value {

	groups := make(map[string][]int16)

	v.EachInt16(func(index int, val int16) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]int16, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInt16 uses the specified function to replace each int16s
// by iterating each item.  The data in the returned result will be a
// []int16 containing the replaced items.
func (v *Value) ReplaceInt16(replacer func(int, int16) int16) *Value {

	arr := v.MustInt16Slice()
	replaced := make([]int16, len(arr))

	v.EachInt16(func(index int, val int16) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInt16 uses the specified collector function to collect a value
// for each of the int16s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInt16(collector func(int, int16) interface{}) *Value {

	arr := v.MustInt16Slice()
	collected := make([]interface{}, len(arr))

	v.EachInt16(func(index int, val int16) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Int32 (int32 and []int32)
	--------------------------------------------------
*/

// Int32 gets the value as a int32, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Int32(optionalDefault ...int32) int32 {
	if s, ok := v.data.(int32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustInt32 gets the value as a int32.
//
// Panics if the object is not a int32.
func (v *Value) MustInt32() int32 {
	return v.data.(int32)
}

// Int32Slice gets the value as a []int32, returns the optionalDefault
// value or nil if the value is not a []int32.
func (v *Value) Int32Slice(optionalDefault ...[]int32) []int32 {
	if s, ok := v.data.([]int32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInt32Slice gets the value as a []int32.
//
// Panics if the object is not a []int32.
func (v *Value) MustInt32Slice() []int32 {
	return v.data.([]int32)
}

// IsInt32 gets whether the object contained is a int32 or not.
func (v *Value) IsInt32() bool {
	_, ok := v.data.(int32)
	return ok
}

// IsInt32Slice gets whether the object contained is a []int32 or not.
func (v *Value) IsInt32Slice() bool {
	_, ok := v.data.([]int32)
	return ok
}

// EachInt32 calls the specified callback for each object
// in the []int32.
//
// Panics if the object is the wrong type.
func (v *Value) EachInt32(callback func(int, int32) bool) *Value {

	for index, val := range v.MustInt32Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInt32 uses the specified decider function to select items
// from the []int32.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInt32(decider func(int, int32) bool) *Value {

	var selected []int32

	v.EachInt32(func(index int, val int32) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInt32 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]int32.
func (v *Value) GroupInt32(grouper func(int, int32) string) *Value {

	groups := make(map[string][]int32)

	v.EachInt32(func(index int, val int32) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]int32, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInt32 uses the specified function to replace each int32s
// by iterating each item.  The data in the returned result will be a
// []int32 containing the replaced items.
func (v *Value) ReplaceInt32(replacer func(int, int32) int32) *Value {

	arr := v.MustInt32Slice()
	replaced := make([]int32, len(arr))

	v.EachInt32(func(index int, val int32) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInt32 uses the specified collector function to collect a value
// for each of the int32s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInt32(collector func(int, int32) interface{}) *Value {

	arr := v.MustInt32Slice()
	collected := make([]interface{}, len(arr))

	v.EachInt32(func(index int, val int32) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Int64 (int64 and []int64)
	--------------------------------------------------
*/

// Int64 gets the value as a int64, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Int64(optionalDefault ...int64) int64 {
	if s, ok := v.data.(int64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustInt64 gets the value as a int64.
//
// Panics if the object is not a int64.
func (v *Value) MustInt64() int64 {
	return v.data.(int64)
}

// Int64Slice gets the value as a []int64, returns the optionalDefault
// value or nil if the value is not a []int64.
func (v *Value) Int64Slice(optionalDefault ...[]int64) []int64 {
	if s, ok := v.data.([]int64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustInt64Slice gets the value as a []int64.
//
// Panics if the object is not a []int64.
func (v *Value) MustInt64Slice() []int64 {
	return v.data.([]int64)
}

// IsInt64 gets whether the object contained is a int64 or not.
func (v *Value) IsInt64() bool {
	_, ok := v.data.(int64)
	return ok
}

// IsInt64Slice gets whether the object contained is a []int64 or not.
func (v *Value) IsInt64Slice() bool {
	_, ok := v.data.([]int64)
	return ok
}

// EachInt64 calls the specified callback for each object
// in the []int64.
//
// Panics if the object is the wrong type.
func (v *Value) EachInt64(callback func(int, int64) bool) *Value {

	for index, val := range v.MustInt64Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereInt64 uses the specified decider function to select items
// from the []int64.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereInt64(decider func(int, int64) bool) *Value {

	var selected []int64

	v.EachInt64(func(index int, val int64) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupInt64 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]int64.
func (v *Value) GroupInt64(grouper func(int, int64) string) *Value {

	groups := make(map[string][]int64)

	v.EachInt64(func(index int, val int64) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]int64, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceInt64 uses the specified function to replace each int64s
// by iterating each item.  The data in the returned result will be a
// []int64 containing the replaced items.
func (v *Value) ReplaceInt64(replacer func(int, int64) int64) *Value {

	arr := v.MustInt64Slice()
	replaced := make([]int64, len(arr))

	v.EachInt64(func(index int, val int64) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectInt64 uses the specified collector function to collect a value
// for each of the int64s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectInt64(collector func(int, int64) interface{}) *Value {

	arr := v.MustInt64Slice()
	collected := make([]interface{}, len(arr))

	v.EachInt64(func(index int, val int64) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uint (uint and []uint)
	--------------------------------------------------
*/

// Uint gets the value as a uint, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uint(optionalDefault ...uint) uint {
	if s, ok := v.data.(uint); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUint gets the value as a uint.
//
// Panics if the object is not a uint.
func (v *Value) MustUint() uint {
	return v.data.(uint)
}

// UintSlice gets the value as a []uint, returns the optionalDefault
// value or nil if the value is not a []uint.
func (v *Value) UintSlice(optionalDefault ...[]uint) []uint {
	if s, ok := v.data.([]uint); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUintSlice gets the value as a []uint.
//
// Panics if the object is not a []uint.
func (v *Value) MustUintSlice() []uint {
	return v.data.([]uint)
}

// IsUint gets whether the object contained is a uint or not.
func (v *Value) IsUint() bool {
	_, ok := v.data.(uint)
	return ok
}

// IsUintSlice gets whether the object contained is a []uint or not.
func (v *Value) IsUintSlice() bool {
	_, ok := v.data.([]uint)
	return ok
}

// EachUint calls the specified callback for each object
// in the []uint.
//
// Panics if the object is the wrong type.
func (v *Value) EachUint(callback func(int, uint) bool) *Value {

	for index, val := range v.MustUintSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUint uses the specified decider function to select items
// from the []uint.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUint(decider func(int, uint) bool) *Value {

	var selected []uint

	v.EachUint(func(index int, val uint) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUint uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uint.
func (v *Value) GroupUint(grouper func(int, uint) string) *Value {

	groups := make(map[string][]uint)

	v.EachUint(func(index int, val uint) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uint, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUint uses the specified function to replace each uints
// by iterating each item.  The data in the returned result will be a
// []uint containing the replaced items.
func (v *Value) ReplaceUint(replacer func(int, uint) uint) *Value {

	arr := v.MustUintSlice()
	replaced := make([]uint, len(arr))

	v.EachUint(func(index int, val uint) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUint uses the specified collector function to collect a value
// for each of the uints in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUint(collector func(int, uint) interface{}) *Value {

	arr := v.MustUintSlice()
	collected := make([]interface{}, len(arr))

	v.EachUint(func(index int, val uint) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uint8 (uint8 and []uint8)
	--------------------------------------------------
*/

// Uint8 gets the value as a uint8, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uint8(optionalDefault ...uint8) uint8 {
	if s, ok := v.data.(uint8); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUint8 gets the value as a uint8.
//
// Panics if the object is not a uint8.
func (v *Value) MustUint8() uint8 {
	return v.data.(uint8)
}

// Uint8Slice gets the value as a []uint8, returns the optionalDefault
// value or nil if the value is not a []uint8.
func (v *Value) Uint8Slice(optionalDefault ...[]uint8) []uint8 {
	if s, ok := v.data.([]uint8); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUint8Slice gets the value as a []uint8.
//
// Panics if the object is not a []uint8.
func (v *Value) MustUint8Slice() []uint8 {
	return v.data.([]uint8)
}

// IsUint8 gets whether the object contained is a uint8 or not.
func (v *Value) IsUint8() bool {
	_, ok := v.data.(uint8)
	return ok
}

// IsUint8Slice gets whether the object contained is a []uint8 or not.
func (v *Value) IsUint8Slice() bool {
	_, ok := v.data.([]uint8)
	return ok
}

// EachUint8 calls the specified callback for each object
// in the []uint8.
//
// Panics if the object is the wrong type.
func (v *Value) EachUint8(callback func(int, uint8) bool) *Value {

	for index, val := range v.MustUint8Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUint8 uses the specified decider function to select items
// from the []uint8.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUint8(decider func(int, uint8) bool) *Value {

	var selected []uint8

	v.EachUint8(func(index int, val uint8) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUint8 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uint8.
func (v *Value) GroupUint8(grouper func(int, uint8) string) *Value {

	groups := make(map[string][]uint8)

	v.EachUint8(func(index int, val uint8) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uint8, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUint8 uses the specified function to replace each uint8s
// by iterating each item.  The data in the returned result will be a
// []uint8 containing the replaced items.
func (v *Value) ReplaceUint8(replacer func(int, uint8) uint8) *Value {

	arr := v.MustUint8Slice()
	replaced := make([]uint8, len(arr))

	v.EachUint8(func(index int, val uint8) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUint8 uses the specified collector function to collect a value
// for each of the uint8s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUint8(collector func(int, uint8) interface{}) *Value {

	arr := v.MustUint8Slice()
	collected := make([]interface{}, len(arr))

	v.EachUint8(func(index int, val uint8) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uint16 (uint16 and []uint16)
	--------------------------------------------------
*/

// Uint16 gets the value as a uint16, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uint16(optionalDefault ...uint16) uint16 {
	if s, ok := v.data.(uint16); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUint16 gets the value as a uint16.
//
// Panics if the object is not a uint16.
func (v *Value) MustUint16() uint16 {
	return v.data.(uint16)
}

// Uint16Slice gets the value as a []uint16, returns the optionalDefault
// value or nil if the value is not a []uint16.
func (v *Value) Uint16Slice(optionalDefault ...[]uint16) []uint16 {
	if s, ok := v.data.([]uint16); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUint16Slice gets the value as a []uint16.
//
// Panics if the object is not a []uint16.
func (v *Value) MustUint16Slice() []uint16 {
	return v.data.([]uint16)
}

// IsUint16 gets whether the object contained is a uint16 or not.
func (v *Value) IsUint16() bool {
	_, ok := v.data.(uint16)
	return ok
}

// IsUint16Slice gets whether the object contained is a []uint16 or not.
func (v *Value) IsUint16Slice() bool {
	_, ok := v.data.([]uint16)
	return ok
}

// EachUint16 calls the specified callback for each object
// in the []uint16.
//
// Panics if the object is the wrong type.
func (v *Value) EachUint16(callback func(int, uint16) bool) *Value {

	for index, val := range v.MustUint16Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUint16 uses the specified decider function to select items
// from the []uint16.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUint16(decider func(int, uint16) bool) *Value {

	var selected []uint16

	v.EachUint16(func(index int, val uint16) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUint16 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uint16.
func (v *Value) GroupUint16(grouper func(int, uint16) string) *Value {

	groups := make(map[string][]uint16)

	v.EachUint16(func(index int, val uint16) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uint16, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUint16 uses the specified function to replace each uint16s
// by iterating each item.  The data in the returned result will be a
// []uint16 containing the replaced items.
func (v *Value) ReplaceUint16(replacer func(int, uint16) uint16) *Value {

	arr := v.MustUint16Slice()
	replaced := make([]uint16, len(arr))

	v.EachUint16(func(index int, val uint16) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUint16 uses the specified collector function to collect a value
// for each of the uint16s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUint16(collector func(int, uint16) interface{}) *Value {

	arr := v.MustUint16Slice()
	collected := make([]interface{}, len(arr))

	v.EachUint16(func(index int, val uint16) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uint32 (uint32 and []uint32)
	--------------------------------------------------
*/

// Uint32 gets the value as a uint32, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uint32(optionalDefault ...uint32) uint32 {
	if s, ok := v.data.(uint32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUint32 gets the value as a uint32.
//
// Panics if the object is not a uint32.
func (v *Value) MustUint32() uint32 {
	return v.data.(uint32)
}

// Uint32Slice gets the value as a []uint32, returns the optionalDefault
// value or nil if the value is not a []uint32.
func (v *Value) Uint32Slice(optionalDefault ...[]uint32) []uint32 {
	if s, ok := v.data.([]uint32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUint32Slice gets the value as a []uint32.
//
// Panics if the object is not a []uint32.
func (v *Value) MustUint32Slice() []uint32 {
	return v.data.([]uint32)
}

// IsUint32 gets whether the object contained is a uint32 or not.
func (v *Value) IsUint32() bool {
	_, ok := v.data.(uint32)
	return ok
}

// IsUint32Slice gets whether the object contained is a []uint32 or not.
func (v *Value) IsUint32Slice() bool {
	_, ok := v.data.([]uint32)
	return ok
}

// EachUint32 calls the specified callback for each object
// in the []uint32.
//
// Panics if the object is the wrong type.
func (v *Value) EachUint32(callback func(int, uint32) bool) *Value {

	for index, val := range v.MustUint32Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUint32 uses the specified decider function to select items
// from the []uint32.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUint32(decider func(int, uint32) bool) *Value {

	var selected []uint32

	v.EachUint32(func(index int, val uint32) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUint32 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uint32.
func (v *Value) GroupUint32(grouper func(int, uint32) string) *Value {

	groups := make(map[string][]uint32)

	v.EachUint32(func(index int, val uint32) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uint32, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUint32 uses the specified function to replace each uint32s
// by iterating each item.  The data in the returned result will be a
// []uint32 containing the replaced items.
func (v *Value) ReplaceUint32(replacer func(int, uint32) uint32) *Value {

	arr := v.MustUint32Slice()
	replaced := make([]uint32, len(arr))

	v.EachUint32(func(index int, val uint32) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUint32 uses the specified collector function to collect a value
// for each of the uint32s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUint32(collector func(int, uint32) interface{}) *Value {

	arr := v.MustUint32Slice()
	collected := make([]interface{}, len(arr))

	v.EachUint32(func(index int, val uint32) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uint64 (uint64 and []uint64)
	--------------------------------------------------
*/

// Uint64 gets the value as a uint64, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uint64(optionalDefault ...uint64) uint64 {
	if s, ok := v.data.(uint64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUint64 gets the value as a uint64.
//
// Panics if the object is not a uint64.
func (v *Value) MustUint64() uint64 {
	return v.data.(uint64)
}

// Uint64Slice gets the value as a []uint64, returns the optionalDefault
// value or nil if the value is not a []uint64.
func (v *Value) Uint64Slice(optionalDefault ...[]uint64) []uint64 {
	if s, ok := v.data.([]uint64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUint64Slice gets the value as a []uint64.
//
// Panics if the object is not a []uint64.
func (v *Value) MustUint64Slice() []uint64 {
	return v.data.([]uint64)
}

// IsUint64 gets whether the object contained is a uint64 or not.
func (v *Value) IsUint64() bool {
	_, ok := v.data.(uint64)
	return ok
}

// IsUint64Slice gets whether the object contained is a []uint64 or not.
func (v *Value) IsUint64Slice() bool {
	_, ok := v.data.([]uint64)
	return ok
}

// EachUint64 calls the specified callback for each object
// in the []uint64.
//
// Panics if the object is the wrong type.
func (v *Value) EachUint64(callback func(int, uint64) bool) *Value {

	for index, val := range v.MustUint64Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUint64 uses the specified decider function to select items
// from the []uint64.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUint64(decider func(int, uint64) bool) *Value {

	var selected []uint64

	v.EachUint64(func(index int, val uint64) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUint64 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uint64.
func (v *Value) GroupUint64(grouper func(int, uint64) string) *Value {

	groups := make(map[string][]uint64)

	v.EachUint64(func(index int, val uint64) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uint64, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUint64 uses the specified function to replace each uint64s
// by iterating each item.  The data in the returned result will be a
// []uint64 containing the replaced items.
func (v *Value) ReplaceUint64(replacer func(int, uint64) uint64) *Value {

	arr := v.MustUint64Slice()
	replaced := make([]uint64, len(arr))

	v.EachUint64(func(index int, val uint64) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUint64 uses the specified collector function to collect a value
// for each of the uint64s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUint64(collector func(int, uint64) interface{}) *Value {

	arr := v.MustUint64Slice()
	collected := make([]interface{}, len(arr))

	v.EachUint64(func(index int, val uint64) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Uintptr (uintptr and []uintptr)
	--------------------------------------------------
*/

// Uintptr gets the value as a uintptr, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Uintptr(optionalDefault ...uintptr) uintptr {
	if s, ok := v.data.(uintptr); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustUintptr gets the value as a uintptr.
//
// Panics if the object is not a uintptr.
func (v *Value) MustUintptr() uintptr {
	return v.data.(uintptr)
}

// UintptrSlice gets the value as a []uintptr, returns the optionalDefault
// value or nil if the value is not a []uintptr.
func (v *Value) UintptrSlice(optionalDefault ...[]uintptr) []uintptr {
	if s, ok := v.data.([]uintptr); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustUintptrSlice gets the value as a []uintptr.
//
// Panics if the object is not a []uintptr.
func (v *Value) MustUintptrSlice() []uintptr {
	return v.data.([]uintptr)
}

// IsUintptr gets whether the object contained is a uintptr or not.
func (v *Value) IsUintptr() bool {
	_, ok := v.data.(uintptr)
	return ok
}

// IsUintptrSlice gets whether the object contained is a []uintptr or not.
func (v *Value) IsUintptrSlice() bool {
	_, ok := v.data.([]uintptr)
	return ok
}

// EachUintptr calls the specified callback for each object
// in the []uintptr.
//
// Panics if the object is the wrong type.
func (v *Value) EachUintptr(callback func(int, uintptr) bool) *Value {

	for index, val := range v.MustUintptrSlice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereUintptr uses the specified decider function to select items
// from the []uintptr.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereUintptr(decider func(int, uintptr) bool) *Value {

	var selected []uintptr

	v.EachUintptr(func(index int, val uintptr) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupUintptr uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]uintptr.
func (v *Value) GroupUintptr(grouper func(int, uintptr) string) *Value {

	groups := make(map[string][]uintptr)

	v.EachUintptr(func(index int, val uintptr) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]uintptr, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceUintptr uses the specified function to replace each uintptrs
// by iterating each item.  The data in the returned result will be a
// []uintptr containing the replaced items.
func (v *Value) ReplaceUintptr(replacer func(int, uintptr) uintptr) *Value {

	arr := v.MustUintptrSlice()
	replaced := make([]uintptr, len(arr))

	v.EachUintptr(func(index int, val uintptr) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectUintptr uses the specified collector function to collect a value
// for each of the uintptrs in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectUintptr(collector func(int, uintptr) interface{}) *Value {

	arr := v.MustUintptrSlice()
	collected := make([]interface{}, len(arr))

	v.EachUintptr(func(index int, val uintptr) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Float32 (float32 and []float32)
	--------------------------------------------------
*/

// Float32 gets the value as a float32, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Float32(optionalDefault ...float32) float32 {
	if s, ok := v.data.(float32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustFloat32 gets the value as a float32.
//
// Panics if the object is not a float32.
func (v *Value) MustFloat32() float32 {
	return v.data.(float32)
}

// Float32Slice gets the value as a []float32, returns the optionalDefault
// value or nil if the value is not a []float32.
func (v *Value) Float32Slice(optionalDefault ...[]float32) []float32 {
	if s, ok := v.data.([]float32); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustFloat32Slice gets the value as a []float32.
//
// Panics if the object is not a []float32.
func (v *Value) MustFloat32Slice() []float32 {
	return v.data.([]float32)
}

// IsFloat32 gets whether the object contained is a float32 or not.
func (v *Value) IsFloat32() bool {
	_, ok := v.data.(float32)
	return ok
}

// IsFloat32Slice gets whether the object contained is a []float32 or not.
func (v *Value) IsFloat32Slice() bool {
	_, ok := v.data.([]float32)
	return ok
}

// EachFloat32 calls the specified callback for each object
// in the []float32.
//
// Panics if the object is the wrong type.
func (v *Value) EachFloat32(callback func(int, float32) bool) *Value {

	for index, val := range v.MustFloat32Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereFloat32 uses the specified decider function to select items
// from the []float32.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereFloat32(decider func(int, float32) bool) *Value {

	var selected []float32

	v.EachFloat32(func(index int, val float32) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupFloat32 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]float32.
func (v *Value) GroupFloat32(grouper func(int, float32) string) *Value {

	groups := make(map[string][]float32)

	v.EachFloat32(func(index int, val float32) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]float32, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceFloat32 uses the specified function to replace each float32s
// by iterating each item.  The data in the returned result will be a
// []float32 containing the replaced items.
func (v *Value) ReplaceFloat32(replacer func(int, float32) float32) *Value {

	arr := v.MustFloat32Slice()
	replaced := make([]float32, len(arr))

	v.EachFloat32(func(index int, val float32) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectFloat32 uses the specified collector function to collect a value
// for each of the float32s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectFloat32(collector func(int, float32) interface{}) *Value {

	arr := v.MustFloat32Slice()
	collected := make([]interface{}, len(arr))

	v.EachFloat32(func(index int, val float32) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Float64 (float64 and []float64)
	--------------------------------------------------
*/

// Float64 gets the value as a float64, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Float64(optionalDefault ...float64) float64 {
	if s, ok := v.data.(float64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustFloat64 gets the value as a float64.
//
// Panics if the object is not a float64.
func (v *Value) MustFloat64() float64 {
	return v.data.(float64)
}

// Float64Slice gets the value as a []float64, returns the optionalDefault
// value or nil if the value is not a []float64.
func (v *Value) Float64Slice(optionalDefault ...[]float64) []float64 {
	if s, ok := v.data.([]float64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustFloat64Slice gets the value as a []float64.
//
// Panics if the object is not a []float64.
func (v *Value) MustFloat64Slice() []float64 {
	return v.data.([]float64)
}

// IsFloat64 gets whether the object contained is a float64 or not.
func (v *Value) IsFloat64() bool {
	_, ok := v.data.(float64)
	return ok
}

// IsFloat64Slice gets whether the object contained is a []float64 or not.
func (v *Value) IsFloat64Slice() bool {
	_, ok := v.data.([]float64)
	return ok
}

// EachFloat64 calls the specified callback for each object
// in the []float64.
//
// Panics if the object is the wrong type.
func (v *Value) EachFloat64(callback func(int, float64) bool) *Value {

	for index, val := range v.MustFloat64Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereFloat64 uses the specified decider function to select items
// from the []float64.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereFloat64(decider func(int, float64) bool) *Value {

	var selected []float64

	v.EachFloat64(func(index int, val float64) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupFloat64 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]float64.
func (v *Value) GroupFloat64(grouper func(int, float64) string) *Value {

	groups := make(map[string][]float64)

	v.EachFloat64(func(index int, val float64) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]float64, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceFloat64 uses the specified function to replace each float64s
// by iterating each item.  The data in the returned result will be a
// []float64 containing the replaced items.
func (v *Value) ReplaceFloat64(replacer func(int, float64) float64) *Value {

	arr := v.MustFloat64Slice()
	replaced := make([]float64, len(arr))

	v.EachFloat64(func(index int, val float64) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectFloat64 uses the specified collector function to collect a value
// for each of the float64s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectFloat64(collector func(int, float64) interface{}) *Value {

	arr := v.MustFloat64Slice()
	collected := make([]interface{}, len(arr))

	v.EachFloat64(func(index int, val float64) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Complex64 (complex64 and []complex64)
	--------------------------------------------------
*/

// Complex64 gets the value as a complex64, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Complex64(optionalDefault ...complex64) complex64 {
	if s, ok := v.data.(complex64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustComplex64 gets the value as a complex64.
//
// Panics if the object is not a complex64.
func (v *Value) MustComplex64() complex64 {
	return v.data.(complex64)
}

// Complex64Slice gets the value as a []complex64, returns the optionalDefault
// value or nil if the value is not a []complex64.
func (v *Value) Complex64Slice(optionalDefault ...[]complex64) []complex64 {
	if s, ok := v.data.([]complex64); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustComplex64Slice gets the value as a []complex64.
//
// Panics if the object is not a []complex64.
func (v *Value) MustComplex64Slice() []complex64 {
	return v.data.([]complex64)
}

// IsComplex64 gets whether the object contained is a complex64 or not.
func (v *Value) IsComplex64() bool {
	_, ok := v.data.(complex64)
	return ok
}

// IsComplex64Slice gets whether the object contained is a []complex64 or not.
func (v *Value) IsComplex64Slice() bool {
	_, ok := v.data.([]complex64)
	return ok
}

// EachComplex64 calls the specified callback for each object
// in the []complex64.
//
// Panics if the object is the wrong type.
func (v *Value) EachComplex64(callback func(int, complex64) bool) *Value {

	for index, val := range v.MustComplex64Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereComplex64 uses the specified decider function to select items
// from the []complex64.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereComplex64(decider func(int, complex64) bool) *Value {

	var selected []complex64

	v.EachComplex64(func(index int, val complex64) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupComplex64 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]complex64.
func (v *Value) GroupComplex64(grouper func(int, complex64) string) *Value {

	groups := make(map[string][]complex64)

	v.EachComplex64(func(index int, val complex64) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]complex64, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceComplex64 uses the specified function to replace each complex64s
// by iterating each item.  The data in the returned result will be a
// []complex64 containing the replaced items.
func (v *Value) ReplaceComplex64(replacer func(int, complex64) complex64) *Value {

	arr := v.MustComplex64Slice()
	replaced := make([]complex64, len(arr))

	v.EachComplex64(func(index int, val complex64) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectComplex64 uses the specified collector function to collect a value
// for each of the complex64s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectComplex64(collector func(int, complex64) interface{}) *Value {

	arr := v.MustComplex64Slice()
	collected := make([]interface{}, len(arr))

	v.EachComplex64(func(index int, val complex64) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}

/*
	Complex128 (complex128 and []complex128)
	--------------------------------------------------
*/

// Complex128 gets the value as a complex128, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) Complex128(optionalDefault ...complex128) complex128 {
	if s, ok := v.data.(complex128); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return 0
}

// MustComplex128 gets the value as a complex128.
//
// Panics if the object is not a complex128.
func (v *Value) MustComplex128() complex128 {
	return v.data.(complex128)
}

// Complex128Slice gets the value as a []complex128, returns the optionalDefault
// value or nil if the value is not a []complex128.
func (v *Value) Complex128Slice(optionalDefault ...[]complex128) []complex128 {
	if s, ok := v.data.([]complex128); ok {
		return s
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}
	return nil
}

// MustComplex128Slice gets the value as a []complex128.
//
// Panics if the object is not a []complex128.
func (v *Value) MustComplex128Slice() []complex128 {
	return v.data.([]complex128)
}

// IsComplex128 gets whether the object contained is a complex128 or not.
func (v *Value) IsComplex128() bool {
	_, ok := v.data.(complex128)
	return ok
}

// IsComplex128Slice gets whether the object contained is a []complex128 or not.
func (v *Value) IsComplex128Slice() bool {
	_, ok := v.data.([]complex128)
	return ok
}

// EachComplex128 calls the specified callback for each object
// in the []complex128.
//
// Panics if the object is the wrong type.
func (v *Value) EachComplex128(callback func(int, complex128) bool) *Value {

	for index, val := range v.MustComplex128Slice() {
		carryon := callback(index, val)
		if carryon == false {
			break
		}
	}

	return v

}

// WhereComplex128 uses the specified decider function to select items
// from the []complex128.  The object contained in the result will contain
// only the selected items.
func (v *Value) WhereComplex128(decider func(int, complex128) bool) *Value {

	var selected []complex128

	v.EachComplex128(func(index int, val complex128) bool {
		shouldSelect := decider(index, val)
		if shouldSelect == false {
			selected = append(selected, val)
		}
		return true
	})

	return &Value{data: selected}

}

// GroupComplex128 uses the specified grouper function to group the items
// keyed by the return of the grouper.  The object contained in the
// result will contain a map[string][]complex128.
func (v *Value) GroupComplex128(grouper func(int, complex128) string) *Value {

	groups := make(map[string][]complex128)

	v.EachComplex128(func(index int, val complex128) bool {
		group := grouper(index, val)
		if _, ok := groups[group]; !ok {
			groups[group] = make([]complex128, 0)
		}
		groups[group] = append(groups[group], val)
		return true
	})

	return &Value{data: groups}

}

// ReplaceComplex128 uses the specified function to replace each complex128s
// by iterating each item.  The data in the returned result will be a
// []complex128 containing the replaced items.
func (v *Value) ReplaceComplex128(replacer func(int, complex128) complex128) *Value {

	arr := v.MustComplex128Slice()
	replaced := make([]complex128, len(arr))

	v.EachComplex128(func(index int, val complex128) bool {
		replaced[index] = replacer(index, val)
		return true
	})

	return &Value{data: replaced}

}

// CollectComplex128 uses the specified collector function to collect a value
// for each of the complex128s in the slice.  The data returned will be a
// []interface{}.
func (v *Value) CollectComplex128(collector func(int, complex128) interface{}) *Value {

	arr := v.MustComplex128Slice()
	collected := make([]interface{}, len(arr))

	v.EachComplex128(func(index int, val complex128) bool {
		collected[index] = collector(index, val)
		return true
	})

	return &Value{data: collected}
}
