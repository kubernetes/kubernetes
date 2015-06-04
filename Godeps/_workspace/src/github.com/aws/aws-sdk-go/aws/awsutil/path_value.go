package awsutil

import (
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

var indexRe = regexp.MustCompile(`(.+)\[(-?\d+)?\]$`)

// rValuesAtPath returns a slice of values found in value v. The values
// in v are explored recursively so all nested values are collected.
func rValuesAtPath(v interface{}, path string, create bool, caseSensitive bool) []reflect.Value {
	pathparts := strings.Split(path, "||")
	if len(pathparts) > 1 {
		for _, pathpart := range pathparts {
			vals := rValuesAtPath(v, pathpart, create, caseSensitive)
			if vals != nil && len(vals) > 0 {
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

			if create && value.Kind() == reflect.Ptr && value.IsNil() {
				value.Set(reflect.New(value.Type().Elem()))
				value = value.Elem()
			} else {
				value = reflect.Indirect(value)
			}

			if value.IsValid() {
				nextvals = append(nextvals, value)
			}
		}
		values = nextvals

		if indexStar || index != nil {
			nextvals = []reflect.Value{}
			for _, value := range values {
				value := reflect.Indirect(value)
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
					if create {
						// TODO resize slice
					} else {
						continue
					}
				} else if i < 0 { // support negative indexing
					i = value.Len() + i
				}
				value = reflect.Indirect(value.Index(i))

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

// ValuesAtPath returns a list of objects at the lexical path inside of a structure
func ValuesAtPath(i interface{}, path string) []interface{} {
	if rvals := rValuesAtPath(i, path, false, true); rvals != nil {
		vals := make([]interface{}, len(rvals))
		for i, rval := range rvals {
			vals[i] = rval.Interface()
		}
		return vals
	}
	return nil
}

// ValuesAtAnyPath returns a list of objects at the case-insensitive lexical
// path inside of a structure
func ValuesAtAnyPath(i interface{}, path string) []interface{} {
	if rvals := rValuesAtPath(i, path, false, false); rvals != nil {
		vals := make([]interface{}, len(rvals))
		for i, rval := range rvals {
			vals[i] = rval.Interface()
		}
		return vals
	}
	return nil
}

// SetValueAtPath sets an object at the lexical path inside of a structure
func SetValueAtPath(i interface{}, path string, v interface{}) {
	if rvals := rValuesAtPath(i, path, true, true); rvals != nil {
		for _, rval := range rvals {
			rval.Set(reflect.ValueOf(v))
		}
	}
}

// SetValueAtAnyPath sets an object at the case insensitive lexical path inside
// of a structure
func SetValueAtAnyPath(i interface{}, path string, v interface{}) {
	if rvals := rValuesAtPath(i, path, true, false); rvals != nil {
		for _, rval := range rvals {
			rval.Set(reflect.ValueOf(v))
		}
	}
}
