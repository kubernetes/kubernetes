// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

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

	allowed := make(map[Key]struct{}, len(keys))
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

	forbid := make(map[Key]struct{}, len(keys))
	for _, k := range keys {
		forbid[k] = struct{}{}
	}
	return func(kv KeyValue) bool {
		_, ok := forbid[kv.Key]
		return !ok
	}
}
