package objx

/*
   MSI (map[string]interface{} and []map[string]interface{})
*/

// MSI gets the value as a map[string]interface{}, returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) MSI(optionalDefault ...map[string]interface{}) map[string]interface{} {
	if s, ok := v.data.(map[string]interface{}); ok {
		return s
	}
	if s, ok := v.data.(Map); ok {
		return map[string]interface{}(s)
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
	if s, ok := v.data.(Map); ok {
		return map[string]interface{}(s)
	}
	return v.data.(map[string]interface{})
}

// MSISlice gets the value as a []map[string]interface{}, returns the optionalDefault
// value or nil if the value is not a []map[string]interface{}.
func (v *Value) MSISlice(optionalDefault ...[]map[string]interface{}) []map[string]interface{} {
	if s, ok := v.data.([]map[string]interface{}); ok {
		return s
	}

	s := v.ObjxMapSlice()
	if s == nil {
		if len(optionalDefault) == 1 {
			return optionalDefault[0]
		}
		return nil
	}

	result := make([]map[string]interface{}, len(s))
	for i := range s {
		result[i] = s[i].Value().MSI()
	}
	return result
}

// MustMSISlice gets the value as a []map[string]interface{}.
//
// Panics if the object is not a []map[string]interface{}.
func (v *Value) MustMSISlice() []map[string]interface{} {
	if s := v.MSISlice(); s != nil {
		return s
	}

	return v.data.([]map[string]interface{})
}

// IsMSI gets whether the object contained is a map[string]interface{} or not.
func (v *Value) IsMSI() bool {
	_, ok := v.data.(map[string]interface{})
	if !ok {
		_, ok = v.data.(Map)
	}
	return ok
}

// IsMSISlice gets whether the object contained is a []map[string]interface{} or not.
func (v *Value) IsMSISlice() bool {
	_, ok := v.data.([]map[string]interface{})
	if !ok {
		_, ok = v.data.([]Map)
		if !ok {
			s, ok := v.data.([]interface{})
			if ok {
				for i := range s {
					switch s[i].(type) {
					case Map:
					case map[string]interface{}:
					default:
						return false
					}
				}
				return true
			}
		}
	}
	return ok
}

// EachMSI calls the specified callback for each object
// in the []map[string]interface{}.
//
// Panics if the object is the wrong type.
func (v *Value) EachMSI(callback func(int, map[string]interface{}) bool) *Value {
	for index, val := range v.MustMSISlice() {
		carryon := callback(index, val)
		if !carryon {
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
		if !shouldSelect {
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
*/

// ObjxMap gets the value as a (Map), returns the optionalDefault
// value or a system default object if the value is the wrong type.
func (v *Value) ObjxMap(optionalDefault ...(Map)) Map {
	if s, ok := v.data.((Map)); ok {
		return s
	}
	if s, ok := v.data.(map[string]interface{}); ok {
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
	if s, ok := v.data.(map[string]interface{}); ok {
		return s
	}
	return v.data.((Map))
}

// ObjxMapSlice gets the value as a [](Map), returns the optionalDefault
// value or nil if the value is not a [](Map).
func (v *Value) ObjxMapSlice(optionalDefault ...[](Map)) [](Map) {
	if s, ok := v.data.([]Map); ok {
		return s
	}

	if s, ok := v.data.([]map[string]interface{}); ok {
		result := make([]Map, len(s))
		for i := range s {
			result[i] = s[i]
		}
		return result
	}

	s, ok := v.data.([]interface{})
	if !ok {
		if len(optionalDefault) == 1 {
			return optionalDefault[0]
		}
		return nil
	}

	result := make([]Map, len(s))
	for i := range s {
		switch s[i].(type) {
		case Map:
			result[i] = s[i].(Map)
		case map[string]interface{}:
			result[i] = New(s[i])
		default:
			return nil
		}
	}
	return result
}

// MustObjxMapSlice gets the value as a [](Map).
//
// Panics if the object is not a [](Map).
func (v *Value) MustObjxMapSlice() [](Map) {
	if s := v.ObjxMapSlice(); s != nil {
		return s
	}
	return v.data.([](Map))
}

// IsObjxMap gets whether the object contained is a (Map) or not.
func (v *Value) IsObjxMap() bool {
	_, ok := v.data.((Map))
	if !ok {
		_, ok = v.data.(map[string]interface{})
	}
	return ok
}

// IsObjxMapSlice gets whether the object contained is a [](Map) or not.
func (v *Value) IsObjxMapSlice() bool {
	_, ok := v.data.([](Map))
	if !ok {
		_, ok = v.data.([]map[string]interface{})
		if !ok {
			s, ok := v.data.([]interface{})
			if ok {
				for i := range s {
					switch s[i].(type) {
					case Map:
					case map[string]interface{}:
					default:
						return false
					}
				}
				return true
			}
		}
	}

	return ok
}

// EachObjxMap calls the specified callback for each object
// in the [](Map).
//
// Panics if the object is the wrong type.
func (v *Value) EachObjxMap(callback func(int, Map) bool) *Value {
	for index, val := range v.MustObjxMapSlice() {
		carryon := callback(index, val)
		if !carryon {
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
		if !shouldSelect {
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
