package treeprint

import (
	"reflect"
	"strings"
)

func isEmpty(v *reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func tagSpec(tag string) (name string, omit bool) {
	parts := strings.Split(tag, ",")
	if len(parts) < 2 {
		return tag, false
	}
	if parts[1] == "omitempty" {
		return parts[0], true
	}
	return parts[0], false
}

func filterTags(tag reflect.StructTag) string {
	tags := strings.Split(string(tag), " ")
	filtered := make([]string, 0, len(tags))
	for i := range tags {
		if strings.HasPrefix(tags[i], "tree:") {
			continue
		}
		filtered = append(filtered, tags[i])
	}
	return strings.Join(filtered, " ")
}
