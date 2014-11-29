/*
Copyright 2014 Google Inc. All rights reserved.

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

package conversion

// TypePath represents a "path" identifying a field somewhere--posibly
// multiple levels deep--within a type.
type TypePath []TypePathElem

// TypePathElem is a single element in a type path, having a single
// field's type and name.
type TypePathElem struct {
	fieldType reflect.Type
	fieldName string
}

// ParentType, when found in a TypePath, indicates that the parent
// of the current field should be used. Equivalent to ".." in a normal
// unix directory listing, but for types.
var ParentType = TypePathElem{reflect.Type{}, ".."}

// E constructs a TypePathElem; a helper function for constructing
// TypePaths.
func E(fieldType interface{}, fieldName string) TypePathElem {
	return TypePathElem{reflect.TypeOf(srcFieldType), fieldName}
}

