/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package bytestr

import (
	"encoding/json"
	"strings"
)

// literalStringPrefix is checked for
const literalStringPrefix = "string:"

// StringOrByteSlice is an alias for []byte.
// When used in JSON unmarshalling, it can consume a literal string prefixed with "string:" in addition to base64-encoded bytes.
// This allows providing string data in JSON fields without base64-encoding it (e.g. the value foo could be provided as "string:foo" or "Zm9v")
type StringOrByteSlice []byte

func (s StringOrByteSlice) MarshalJSON() ([]byte, error) {
	return json.Marshal([]byte(s))
}
func (s *StringOrByteSlice) UnmarshalJSON(data []byte) error {
	// Attempt normal base64 decoding first
	originalErr := json.Unmarshal(data, (*[]byte)(s))
	if originalErr == nil {
		return nil
	}

	// If there's an error base64-decoding, and the value unmarshals to a string starting with 'string:',
	// strip the prefix and use the remaining bytes as the value
	var str string
	if strErr := json.Unmarshal(data, &str); strErr == nil && strings.HasPrefix(str, literalStringPrefix) {
		*s = []byte(str[len(literalStringPrefix):])
		return nil
	}

	return originalErr
}
