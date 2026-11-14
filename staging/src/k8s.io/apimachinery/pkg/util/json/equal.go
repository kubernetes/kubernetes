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

package json

import (
	"bytes"
	"encoding/json/jsontext"
)

// EqualWhenNormalized reports whether oldInput and newInput are equal after normalizing JSON
// whitespace, string escaping, and object member order. Array order and number
// spellings remain significant, so 1, 1.0, and 1e0 are distinct. Neither input is
// modified.
//
// Duplicate object names are allowed and normalized by jsontext.Value.Format.
// Invalid UTF-8 in strings is replaced with the Unicode replacement character.
// Byte-identical inputs are equal without parsing; otherwise malformed JSON
// compares unequal. EqualWhenNormalized is not a substitute for JSON validation.
func EqualWhenNormalized(oldInput, newInput []byte) bool {
	if bytes.Equal(oldInput, newInput) {
		return true
	}

	oldJSON := jsontext.Value(bytes.Clone(oldInput))
	if err := oldJSON.Format(jsontext.ReorderRawObjects(true), jsontext.AllowDuplicateNames(true), jsontext.AllowInvalidUTF8(true)); err != nil {
		return false
	}
	newJSON := jsontext.Value(bytes.Clone(newInput))
	if err := newJSON.Format(jsontext.ReorderRawObjects(true), jsontext.AllowDuplicateNames(true), jsontext.AllowInvalidUTF8(true)); err != nil {
		return false
	}
	return bytes.Equal(oldJSON, newJSON)
}
