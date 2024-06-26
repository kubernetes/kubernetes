// Copyright The OpenTelemetry Authors
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

package attribute // import "go.opentelemetry.io/otel/attribute"

// Filter supports removing certain attributes from attribute sets. When
// the filter returns true, the attribute will be kept in the filtered
// attribute set. When the filter returns false, the attribute is excluded
// from the filtered attribute set, and the attribute instead appears in
// the removed list of excluded attributes.
type Filter func(KeyValue) bool

// NewAllowKeysFilter returns a Filter that only allows attributes with one of
// the provided keys.
//
// If keys is empty a deny-all filter is returned.
func NewAllowKeysFilter(keys ...Key) Filter {
	if len(keys) <= 0 {
		return func(kv KeyValue) bool { return false }
	}

	allowed := make(map[Key]struct{})
	for _, k := range keys {
		allowed[k] = struct{}{}
	}
	return func(kv KeyValue) bool {
		_, ok := allowed[kv.Key]
		return ok
	}
}

// NewDenyKeysFilter returns a Filter that only allows attributes
// that do not have one of the provided keys.
//
// If keys is empty an allow-all filter is returned.
func NewDenyKeysFilter(keys ...Key) Filter {
	if len(keys) <= 0 {
		return func(kv KeyValue) bool { return true }
	}

	forbid := make(map[Key]struct{})
	for _, k := range keys {
		forbid[k] = struct{}{}
	}
	return func(kv KeyValue) bool {
		_, ok := forbid[kv.Key]
		return !ok
	}
}
