//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"bytes"
	"fmt"

	"reflect"
)

// Stringify attempts to create a reasonable string representation of types in
// the GitHub library.  It does things like resolve pointers to their values
// and omits struct fields with nil values.
func Stringify(message interface{}) string {
	var buf bytes.Buffer
	v := reflect.ValueOf(message)
	stringifyValue(&buf, v)
	return buf.String()
}

// stringifyValue was heavily inspired by the goprotobuf library.
func stringifyValue(buf *bytes.Buffer, val reflect.Value) {
	if val.Kind() == reflect.Ptr && val.IsNil() {
		buf.WriteString("<nil>")
		return
	}

	v := reflect.Indirect(val)

	switch v.Kind() {
	case reflect.String:
		fmt.Fprintf(buf, `"%s"`, v)
	case reflect.Slice:
		buf.WriteByte('[')
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				buf.WriteByte(' ')
			}

			stringifyValue(buf, v.Index(i))
		}

		buf.WriteByte(']')
		return
	case reflect.Struct:
		if v.Type().Name() != "" {
			buf.WriteString(v.Type().String())
		}

		buf.WriteByte('{')

		var sep bool
		for i := 0; i < v.NumField(); i++ {
			fv := v.Field(i)
			if fv.Kind() == reflect.Ptr && fv.IsNil() {
				continue
			}
			if fv.Kind() == reflect.Slice && fv.IsNil() {
				continue
			}

			if sep {
				buf.WriteString(", ")
			} else {
				sep = true
			}

			buf.WriteString(v.Type().Field(i).Name)
			buf.WriteByte(':')
			stringifyValue(buf, fv)
		}

		buf.WriteByte('}')
	default:
		if v.CanInterface() {
			fmt.Fprint(buf, v.Interface())
		}
	}
}
