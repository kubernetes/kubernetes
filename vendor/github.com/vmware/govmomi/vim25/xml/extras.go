/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package xml

import (
	"reflect"
	"time"
)

var xmlSchemaInstance = Name{Space: "http://www.w3.org/2001/XMLSchema-instance", Local: "type"}

var xsiType = Name{Space: "xsi", Local: "type"}

var stringToTypeMap = map[string]reflect.Type{
	"xsd:boolean":       reflect.TypeOf((*bool)(nil)).Elem(),
	"xsd:byte":          reflect.TypeOf((*int8)(nil)).Elem(),
	"xsd:short":         reflect.TypeOf((*int16)(nil)).Elem(),
	"xsd:int":           reflect.TypeOf((*int32)(nil)).Elem(),
	"xsd:long":          reflect.TypeOf((*int64)(nil)).Elem(),
	"xsd:unsignedByte":  reflect.TypeOf((*uint8)(nil)).Elem(),
	"xsd:unsignedShort": reflect.TypeOf((*uint16)(nil)).Elem(),
	"xsd:unsignedInt":   reflect.TypeOf((*uint32)(nil)).Elem(),
	"xsd:unsignedLong":  reflect.TypeOf((*uint64)(nil)).Elem(),
	"xsd:float":         reflect.TypeOf((*float32)(nil)).Elem(),
	"xsd:double":        reflect.TypeOf((*float64)(nil)).Elem(),
	"xsd:string":        reflect.TypeOf((*string)(nil)).Elem(),
	"xsd:dateTime":      reflect.TypeOf((*time.Time)(nil)).Elem(),
	"xsd:base64Binary":  reflect.TypeOf((*[]byte)(nil)).Elem(),
}

// Return a reflect.Type for the specified type. Nil if unknown.
func stringToType(s string) reflect.Type {
	return stringToTypeMap[s]
}

// Return a string for the specified reflect.Type. Panic if unknown.
func typeToString(typ reflect.Type) string {
	switch typ.Kind() {
	case reflect.Bool:
		return "xsd:boolean"
	case reflect.Int8:
		return "xsd:byte"
	case reflect.Int16:
		return "xsd:short"
	case reflect.Int32:
		return "xsd:int"
	case reflect.Int, reflect.Int64:
		return "xsd:long"
	case reflect.Uint8:
		return "xsd:unsignedByte"
	case reflect.Uint16:
		return "xsd:unsignedShort"
	case reflect.Uint32:
		return "xsd:unsignedInt"
	case reflect.Uint, reflect.Uint64:
		return "xsd:unsignedLong"
	case reflect.Float32:
		return "xsd:float"
	case reflect.Float64:
		return "xsd:double"
	case reflect.String:
		name := typ.Name()
		if name == "string" {
			return "xsd:string"
		}
		return name
	case reflect.Struct:
		if typ == stringToTypeMap["xsd:dateTime"] {
			return "xsd:dateTime"
		}

		// Expect any other struct to be handled...
		return typ.Name()
	case reflect.Slice:
		if typ.Elem().Kind() == reflect.Uint8 {
			return "xsd:base64Binary"
		}
	case reflect.Array:
		if typ.Elem().Kind() == reflect.Uint8 {
			return "xsd:base64Binary"
		}
	}

	panic("don't know what to do for type: " + typ.String())
}
