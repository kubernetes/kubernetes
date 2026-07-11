/*
Copyright 2024 The Kubernetes Authors.

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

package api

import (
	"encoding/json"
	"unique"
)

// NullUniqueString is a UniqueString which contains no string.
var NullUniqueString UniqueString

// UniqueString is a wrapper around [unique.Handle[string]].
type UniqueString unique.Handle[string]

// Returns the string that is stored in the UniqueString.
// If the UniqueString is null, the empty string is returned.
func (us UniqueString) String() string {
	if us == NullUniqueString {
		return ""
	}
	return unique.Handle[string](us).Value()
}

// MarshalJSON is primarily useful for pretty-printing as JSON or YAML.
func (us UniqueString) MarshalJSON() ([]byte, error) {
	return json.Marshal(us.String())
}

// MarshalText allows UniqueString to be used as the key in maps
// without causing problems for logging.
func (us UniqueString) MarshalText() ([]byte, error) {
	return []byte(us.String()), nil
}

// MakeUniqueString constructs a new unique string.
func MakeUniqueString(str string) UniqueString {
	return UniqueString(unique.Make(str))
}
