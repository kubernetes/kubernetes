// +build codegen

package api

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/aws/aws-sdk-go/private/util"
)

// A paramFiller provides string formatting for a shape and its types.
type paramFiller struct {
	prefixPackageName bool
}

// typeName returns the type name of a shape.
func (f paramFiller) typeName(shape *Shape) string {
	if f.prefixPackageName && shape.Type == "structure" {
		return "*" + shape.API.PackageName() + "." + shape.GoTypeElem()
	}
	return shape.GoType()
}

// ParamsStructFromJSON returns a JSON string representation of a structure.
func ParamsStructFromJSON(value interface{}, shape *Shape, prefixPackageName bool) string {
	f := paramFiller{prefixPackageName: prefixPackageName}
	return util.GoFmt(f.paramsStructAny(value, shape))
}

// paramsStructAny returns the string representation of any value.
func (f paramFiller) paramsStructAny(value interface{}, shape *Shape) string {
	if value == nil {
		return ""
	}

	switch shape.Type {
	case "structure":
		if value != nil {
			vmap := value.(map[string]interface{})
			return f.paramsStructStruct(vmap, shape)
		}
	case "list":
		vlist := value.([]interface{})
		return f.paramsStructList(vlist, shape)
	case "map":
		vmap := value.(map[string]interface{})
		return f.paramsStructMap(vmap, shape)
	case "string", "character":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() {
			return fmt.Sprintf("aws.String(%#v)", v.Interface())
		}
	case "blob":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() && shape.Streaming {
			return fmt.Sprintf("bytes.NewReader([]byte(%#v))", v.Interface())
		} else if v.IsValid() {
			return fmt.Sprintf("[]byte(%#v)", v.Interface())
		}
	case "boolean":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() {
			return fmt.Sprintf("aws.Bool(%#v)", v.Interface())
		}
	case "integer", "long":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() {
			return fmt.Sprintf("aws.Int64(%v)", v.Interface())
		}
	case "float", "double":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() {
			return fmt.Sprintf("aws.Float64(%v)", v.Interface())
		}
	case "timestamp":
		v := reflect.Indirect(reflect.ValueOf(value))
		if v.IsValid() {
			return fmt.Sprintf("aws.Time(time.Unix(%d, 0))", int(v.Float()))
		}
	case "jsonvalue":
		v, err := json.Marshal(value)
		if err != nil {
			panic("failed to marshal JSONValue, " + err.Error())
		}
		const tmpl = `func() aws.JSONValue {
			var m aws.JSONValue
			if err := json.Unmarshal([]byte(%q), &m); err != nil {
				panic("failed to unmarshal JSONValue, "+err.Error())
			}
			return m
		}()`
		return fmt.Sprintf(tmpl, string(v))
	default:
		panic("Unhandled type " + shape.Type)
	}
	return ""
}

// paramsStructStruct returns the string representation of a structure
func (f paramFiller) paramsStructStruct(value map[string]interface{}, shape *Shape) string {
	out := "&" + f.typeName(shape)[1:] + "{\n"
	for _, n := range shape.MemberNames() {
		ref := shape.MemberRefs[n]
		name := findParamMember(value, n)

		if val := f.paramsStructAny(value[name], ref.Shape); val != "" {
			out += fmt.Sprintf("%s: %s,\n", n, val)
		}
	}
	out += "}"
	return out
}

// paramsStructMap returns the string representation of a map of values
func (f paramFiller) paramsStructMap(value map[string]interface{}, shape *Shape) string {
	out := f.typeName(shape) + "{\n"
	keys := util.SortedKeys(value)
	for _, k := range keys {
		v := value[k]
		out += fmt.Sprintf("%q: %s,\n", k, f.paramsStructAny(v, shape.ValueRef.Shape))
	}
	out += "}"
	return out
}

// paramsStructList returns the string representation of slice of values
func (f paramFiller) paramsStructList(value []interface{}, shape *Shape) string {
	out := f.typeName(shape) + "{\n"
	for _, v := range value {
		out += fmt.Sprintf("%s,\n", f.paramsStructAny(v, shape.MemberRef.Shape))
	}
	out += "}"
	return out
}

// findParamMember searches a map for a key ignoring case. Returns the map key if found.
func findParamMember(value map[string]interface{}, key string) string {
	for actualKey := range value {
		if strings.EqualFold(key, actualKey) {
			return actualKey
		}
	}
	return ""
}
