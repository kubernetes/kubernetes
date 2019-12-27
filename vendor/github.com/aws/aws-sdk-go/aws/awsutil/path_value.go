package awsutil

import (
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/jmespath/go-jmespath"
)

var indexRe = regexp.MustCompile(`(.+)\[(-?\d+)?\]$`)

// rValuesAtPath returns a slice of values found in value v. The values
// in v are explored recursively so all nested values are collected.
func rValuesAtPath(v interface{}, path string, createPath, caseSensitive, nilTerm bool) []reflect.Value {
	pathparts := strings.Split(path, "||")
	if len(pathparts) > 1 {
		for _, pathpart := range pathparts {
			vals := rValuesAtPath(v, pathpart, createPath, caseSensitive, nilTerm)
			if len(vals) > 0 {
				return vals
			}
		}
		return nil
	}

	values := []reflect.Value{reflect.Indirect(reflect.ValueOf(v))}
	components := strings.Split(path, ".")
	for len(values) > 0 && len(components) > 0 {
		var index *int64
		var indexStar bool
		c := strings.TrimSpace(components[0])
		if c == "" { // no actual component, illegal syntax
			return nil
		} else if caseSensitive && c != "*" && strings.ToLower(c[0:1]) == c[0:1] {
			// TODO normalize case for user
			return nil // don't support unexported fields
		}

		// parse this component
		if m := indexRe.FindStringSubmatch(c); m != nil {
			c = m[1]
			if m[2] == "" {
				index = nil
				indexStar = true
			} else {
				i, _ := strconv.ParseInt(m[2], 10, 32)
				index = &i
				indexStar = false
			}
		}

		nextvals := []reflect.Value{}
		for _, value := range values {
			// pull component name out of struct member
			if value.Kind() != reflect.Struct {
				continue
			}

			if c == "*" { // pull all members
				for i := 0; i < value.NumField(); i++ {
					if f := reflect.Indirect(value.Field(i)); f.IsValid() {
						nextvals = append(nextvals, f)
					}
				}
				continue
			}

			value = value.FieldByNameFunc(func(name string) bool {
				if c == name {
					return true
				} else if !caseSensitive && strings.ToLower(name) == strings.ToLower(c) {
					return true
				}
				return false
			})

			if nilTerm && value.Kind() == reflect.Ptr && len(components[1:]) == 0 {
				if !value.IsNil() {
					value.Set(reflect.Zero(value.Type()))
				}
				return []reflect.Value{value}
			}

			if createPath && value.Kind() == reflect.Ptr && value.IsNil() {
				// TODO if the value is the terminus it should not be created
				// if the value to be set to its position is nil.
				value.Set(reflect.New(value.Type().Elem()))
				value = value.Elem()
			} else {
				value = reflect.Indirect(value)
			}

			if value.Kind() == reflect.Slice || value.Kind() == reflect.Map {
				if !createPath && value.IsNil() {
					value = reflect.ValueOf(nil)
				}
			}

			if value.IsValid() {
				nextvals = append(nextvals, value)
			}
		}
		values = nextvals

		if indexStar || index != nil {
			nextvals = []reflect.Value{}
			for _, valItem := range values {
				value := reflect.Indirect(valItem)
				if value.Kind() != reflect.Slice {
					continue
				}

				if indexStar { // grab all indices
					for i := 0; i < value.Len(); i++ {
						idx := reflect.Indirect(value.Index(i))
						if idx.IsValid() {
							nextvals = append(nextvals, idx)
						}
					}
					continue
				}

				// pull out index
				i := int(*index)
				if i >= value.Len() { // check out of bounds
					if createPath {
						// TODO resize slice
					} else {
						continue
					}
				} else if i < 0 { // support negative indexing
					i = value.Len() + i
				}
				value = reflect.Indirect(value.Index(i))

				if value.Kind() == reflect.Slice || value.Kind() == reflect.Map {
					if !createPath && value.IsNil() {
						value = reflect.ValueOf(nil)
					}
				}

				if value.IsValid() {
					nextvals = append(nextvals, value)
				}
			}
			values = nextvals
		}

		components = components[1:]
	}
	return values
}

// ValuesAtPath returns a list of values at the case insensitive lexical
// path inside of a structure.
func ValuesAtPath(i interface{}, path string) ([]interface{}, error) {
	result, err := jmespath.Search(path, i)
	if err != nil {
		return nil, err
	}

	v := reflect.ValueOf(result)
	if !v.IsValid() || (v.Kind() == reflect.Ptr && v.IsNil()) {
		return nil, nil
	}
	if s, ok := result.([]interface{}); ok {
		return s, err
	}
	if v.Kind() == reflect.Map && v.Len() == 0 {
		return nil, nil
	}
	if v.Kind() == reflect.Slice {
		out := make([]interface{}, v.Len())
		for i := 0; i < v.Len(); i++ {
			out[i] = v.Index(i).Interface()
		}
		return out, nil
	}

	return []interface{}{result}, nil
}

// SetValueAtPath sets a value at the case insensitive lexical path inside
// of a structure.
func SetValueAtPath(i interface{}, path string, v interface{}) {
	rvals := rValuesAtPath(i, path, true, false, v == nil)
	for _, rval := range rvals {
		if rval.Kind() == reflect.Ptr && rval.IsNil() {
			continue
		}
		setValue(rval, v)
	}
}

func setValue(dstVal reflect.Value, src interface{}) {
	if dstVal.Kind() == reflect.Ptr {
		dstVal = reflect.Indirect(dstVal)
	}
	srcVal := reflect.ValueOf(src)

	if !srcVal.IsValid() { // src is literal nil
		if dstVal.CanAddr() {
			// Convert to pointer so that pointer's value can be nil'ed
			//                     dstVal = dstVal.Addr()
		}
		dstVal.Set(reflect.Zero(dstVal.Type()))

	} else if srcVal.Kind() == reflect.Ptr {
		if srcVal.IsNil() {
			srcVal = reflect.Zero(dstVal.Type())
		} else {
			srcVal = reflect.ValueOf(src).Elem()
		}
		dstVal.Set(srcVal)
	} else {
		dstVal.Set(srcVal)
	}

}
