//go:build go1.27

/*
Copyright The Kubernetes Authors.

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

package runtime

import (
	"reflect"
)

func isInlinedFromTag(field reflect.StructField, tagName string, tagDirectives []string) bool {
	fieldType := field.Type
	if fieldType.Kind() == reflect.Pointer && fieldType.Name() == "" {
		// optionally unwrap a single level
		fieldType = fieldType.Elem()
	}
	// TODO: when switching to direct use of json/v2, error on non-struct embedding and use of embed with other directives
	return fieldType.Kind() == reflect.Struct && tagName == "" && len(tagDirectives) == 1 && tagDirectives[0] == "embed"
}
