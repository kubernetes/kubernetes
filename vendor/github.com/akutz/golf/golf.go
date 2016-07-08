// Golf (Go List Fields) enables retrieving a map of an object's field
// names and values through either interface implementation or
// reflection.
package golf

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strconv"
	"strings"
)

var isDebug bool

func init() {
	isDebug, _ = strconv.ParseBool(os.Getenv("GOLF_DEBUG"))
}

func debug(format string, args ...interface{}) {
	if isDebug {
		log.Printf(format, args...)
	}
}

const (
	// A field's JSON and Golf tags are both inspected with preference given to
	// the Golf tags. This is the default.
	PreferGolfTags = iota

	// A field's JSON and Golf tags are both inspected with preference given to
	// the JSON tags.
	PreferJsonTags

	// A field's JSON tags are ignored and only the Golf tags are used.
	IgnoreJsonTags

	// A field's Golf tags are ignored and only the JSON tags are used.
	IgnoreGolfTags
)

const (
	// One of the following types:
	//
	//   - Array
	//   - Chan
	//   - Func
	//   - Interface
	//   - Map
	//   - Array
	//   - Ptr
	//   - Slice
	//   - UnsafePointer
	NilKind = iota

	// One of the following types:
	//
	//   - Int
	//   - Int8
	//   - Int16
	//   - Int32
	//   - Int64
	//   - Uint
	//   - Uint8
	//   - Uint16
	//   - Uint32
	//   - Uint64
	//   - Uintptr
	//   - Float32
	//   - Float64
	//   - Complex64
	//   - Complex128
	ValueKind

	// A string type
	StringKind

	// A struct type
	StructKind
)

// Golfs is an interface implemented by types in order to indicate to external
// packages that the type is aware that Golf may be used, thus giving the
// external package a way to test if a type is golf-aware.
type Golfs interface {
	// PlayGolf is a dummy function that has no impact on anything other than
	// supplying an empty interface with a way to be implemented.
	PlayGolf() bool
}

// GolfsWithExportedFields is an interface implemented by types in order to
// return a map of explicit field names and values.
//
// If this function returns nil, and the inspected type is a struct then 
// reflection is used to infer the object's field names and values.
type GolfsWithExportedFields interface {
	// GolfExportedFields returns a map of the exported field names and their
	// values.
	GolfExportedFields() map[string]interface{}
}

// GolfsWithJsonTagBehavior is an interface impelemented by types in order to
// influence the logic of how a struct's field's tags are handled when 
// reflection is used to infer an object's field names and values. This
// interface is only valid for struct types and will be ignored for others.
type GolfsWithJsonTagBehavior interface {
	// GolfJsonTagBehavior returns an integer that dictates how a field's
	// possible JSON tags should be treated.
	//
	//   0 - A field's JSON and Golf tags are both inspected with preference
	//       given to the Golf tags.
	//
	//   1 - A field's JSON and Golf tags are both inspected with preference
	//       given to the JSON tags.
	//
	//   2 - A field's JSON tags are ignored and only the Golf tags are used.
	//
	//   3 - A field's Golf tags are ignored and only the JSON tags are used.
	GolfJsonTagBehavior() int
}

// Fore produces a map or an object's field names and values using the provided
// key as the prefix for any field name. 

// This function will inspect an object to see if it implements the 
// GolfsWithExportedFields interface first and foremost in order to enable
// non-struct types to return a map of field name/value data from the 
// GolfExportedFields function.
//
// However, if the GolfExportedFields function returns nil and the provided
// object is a struct, reflection will be used to gather information about 
// field names and values.
//
// Regardless of whether the GolfExportedFields map or data gathered through
// reflection is used, this function will recurse into the top-level data in 
// order to map out the object graph, producing key/value pairs and adding them
// to the returned map for the entire graph.
func Fore(key string, val interface{}) map[string]interface{} {

	if isNil(val) {
		return nil
	}

	bk9 := map[string]interface{}{}
	fore(key, val, bk9)
	return bk9
}

func fore(key string, val interface{}, bk9 map[string]interface{}) {

	// get val's golf tips
	ef, jtb := caddy(val)

	// if the value is not a struct or a pointer to a struct and there are
	// no explicit exported fields then place the value in the map and return
	isaStruct := isStruct(val)
	if !isaStruct && ef == nil {
		bk9[key] = val
		return
	}

	// if the value is a struct or a pointer to a struct and there are no
	// exported fields then use reflection to inspect the struct's fields
	if isaStruct && ef == nil {

		ef = map[string]interface{}{}

		// get the value's type
		vt := reflect.ValueOf(val).Elem()

		// get the value's type's type
		tt := vt.Type()

		// populate a map with the value's exported field names and values
		for x := 0; x < vt.NumField(); x++ {

			ttf := tt.Field(x)
			omit, fieldName, omitEmpty := parseField(&ttf, jtb)

			if omit {
				continue
			}

			vtf := vt.Field(x)

			// only work with exported fields
			if !vtf.CanInterface() {
				continue
			}

			ie, vv, vk := parseValue(&vtf)
			if omitEmpty && (vk == StringKind || vk == NilKind) && ie {
				continue
			}

			ef[fieldName] = vv
		}
	}

	for k, v := range ef {

		kk := fmt.Sprintf("%s.%s", key, k)
		rv := reflect.ValueOf(v)
		ie, vv, vk := parseValue(&rv)

		if vk == NilKind && ie {
			bk9[kk] = vv
			continue
		}

		if vk == StringKind || vk == ValueKind {
			bk9[kk] = vv
		}
		fore(kk, vv, bk9)
	}
}

func parseField(
	field *reflect.StructField,
	tagBehavior int) (omit bool, name string, omitEmpty bool) {

	name = field.Name
	tagPtr := &field.Tag

	switch tagBehavior {

	case IgnoreJsonTags:
		parseTag(tagPtr, "golf", &omit, &name, &omitEmpty)

	case IgnoreGolfTags:
		parseTag(tagPtr, "json", &omit, &name, &omitEmpty)

	case PreferJsonTags:
		ios, ins, ies := parseTag(
			tagPtr, "json", &omit, &name, &omitEmpty)

		var omitPtr *bool
		if !ios {
			omitPtr = &omit
		}
		var namePtr *string
		if !ins {
			namePtr = &name
		}
		var omitEmptyPtr *bool
		if !ies {
			omitEmptyPtr = &omitEmpty
		}

		parseTag(
			tagPtr, "golf", omitPtr, namePtr, omitEmptyPtr)

	case PreferGolfTags:
		ios, ins, ies := parseTag(
			tagPtr, "golf", &omit, &name, &omitEmpty)

		var omitPtr *bool
		if !ios {
			omitPtr = &omit
		}
		var namePtr *string
		if !ins {
			namePtr = &name
		}
		var omitEmptyPtr *bool
		if !ies {
			omitEmptyPtr = &omitEmpty
		}

		parseTag(
			tagPtr, "json", omitPtr, namePtr, omitEmptyPtr)
	}

	debug("field %s omit=%v omitEmpty=%v", name, omit, omitEmpty)

	return
}

func caddy(val interface{}) (map[string]interface{}, int) {

	var ef map[string]interface{}
	var jtb int

	switch vt := val.(type) {
	case GolfsWithExportedFields:
		ef = vt.GolfExportedFields()
	default:
		ef = nil
	}

	switch vt := val.(type) {
	case GolfsWithJsonTagBehavior:
		jtb = vt.GolfJsonTagBehavior()
	default:
		jtb = PreferGolfTags
	}

	return ef, jtb
}

func parseTag(
	tag *reflect.StructTag,
	tagName string,
	omit *bool,
	name *string,
	omitEmpty *bool) (isOmitSet, isNameSet, isOmitEmptySet bool) {

	isOmitSet = false
	isNameSet = false
	isOmitEmptySet = false

	if omit == nil && name == nil && omitEmpty == nil {
		return
	}

	t := tag.Get(tagName)

	if t == "" {
		return
	}

	debug("parsing %s tag %s", tagName, t)

	tagParts := strings.Split(t, ",")
	for x := 0; x < len(tagParts); x++ {
		tp := tagParts[x]
		switch x {
		case 0:
			if tp == "-" && omit != nil {
				*omit = true
				isOmitSet = true
			} else if tp != "" && name != nil {
				*name = tp
				isNameSet = true
			}
		case 1:
			if tp == "omitempty" && omitEmpty != nil {
				*omitEmpty = true
				isOmitEmptySet = true
			} else if tp != "" && name != nil {
				*name = tp
				isNameSet = true
			}
		}
	}

	return
}

func parseValue(v *reflect.Value) (isEmpty bool, value interface{}, kind int) {

	if isStringKind(v, &isEmpty, &value) {
		kind = StringKind
	} else if isStructKind(v, &isEmpty, &value) {
		kind = StructKind
	} else if isValueKind(v, &isEmpty, &value) {
		kind = ValueKind
	} else if isNilKind(v, &isEmpty, &value) {
		kind = NilKind
	}

	return
}

func isStringKind(v *reflect.Value, isEmpty *bool, val *interface{}) bool {
	switch v.Kind() {
	case reflect.String:
		if val != nil && v.CanInterface() {
			*val = v.String()
		}
		if isEmpty != nil {
			*isEmpty = *val == ""
		}
		return true
	}
	return false
}

func isStructKind(v *reflect.Value, isEmpty *bool, val *interface{}) bool {
	switch v.Kind() {
	case reflect.Struct:
		if val != nil && v.CanInterface() {
			*val = v.Interface()
		}
		if isEmpty != nil {
			*isEmpty = *val == reflect.New(v.Type()).Elem().Interface()
		}
		return true
	}
	return false
}

func isStruct(v interface{}) bool {

	// to determine if val is a struct we need to test its reflected kind and
	// if that fails, also test to see if val is a pointer or interface that
	// points to a struct
	rv := reflect.ValueOf(v)
	rk := rv.Kind()
	isaStruct := rk == reflect.Struct
	if !isaStruct && (rk == reflect.Interface || rk == reflect.Ptr) {
		isaStruct = rv.Elem().Kind() == reflect.Struct
	}
	return isaStruct
}

func isValueKind(v *reflect.Value, isEmpty *bool, val *interface{}) bool {
	switch v.Kind() {
	case
		reflect.Bool,
		reflect.Int,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Int64,
		reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32,
		reflect.Uint64,
		reflect.Uintptr,
		reflect.Float32,
		reflect.Float64,
		reflect.Complex64,
		reflect.Complex128:
		if val != nil && v.CanInterface() {
			*val = v.Interface()
		}
		if isEmpty != nil {
			*isEmpty = false
		}
		return true
	}
	return false
}

func isNilKind(v *reflect.Value, isEmpty *bool, val *interface{}) bool {
	switch v.Kind() {
	case
		reflect.Array,
		reflect.Chan,
		reflect.Func,
		reflect.Interface,
		reflect.Map,
		reflect.Ptr,
		reflect.Slice,
		reflect.UnsafePointer:
		if val != nil && v.CanInterface() {
			*val = v.Interface()
		}

		if isEmpty != nil {
			*isEmpty = v.IsNil()
		}
		return true
	}
	return false
}

func isNil(v interface{}) bool {
	if v == nil {
		return true
	}
	var ie bool
	rv := reflect.ValueOf(v)
	return isNilKind(&rv, &ie, nil) && ie
}
