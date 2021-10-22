// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package descfmt

import (
	"testing"
)

// TestDescriptorAccessors tests that descriptorAccessors is up-to-date.
func TestDescriptorAccessors(t *testing.T) {
	ignore := map[string]bool{
		"ParentFile":    true,
		"Parent":        true,
		"Index":         true,
		"Syntax":        true,
		"Name":          true,
		"FullName":      true,
		"IsPlaceholder": true,
		"Options":       true,
		"ProtoInternal": true,
		"ProtoType":     true,

		"TextName":           true, // derived from other fields
		"HasOptionalKeyword": true, // captured by HasPresence
		"IsSynthetic":        true, // captured by HasPresence

		"SourceLocations":       true, // specific to FileDescriptor
		"ExtensionRangeOptions": true, // specific to MessageDescriptor
		"DefaultEnumValue":      true, // specific to FieldDescriptor
		"MapKey":                true, // specific to FieldDescriptor
		"MapValue":              true, // specific to FieldDescriptor
	}

	for rt, m := range descriptorAccessors {
		got := map[string]bool{}
		for _, s := range m {
			got[s] = true
		}
		want := map[string]bool{}
		for i := 0; i < rt.NumMethod(); i++ {
			want[rt.Method(i).Name] = true
		}

		// Check if descriptorAccessors contains a non-existent accessor.
		// If this test fails, remove the accessor from descriptorAccessors.
		for s := range got {
			if !want[s] && !ignore[s] {
				t.Errorf("%v.%v does not exist", rt, s)
			}
		}

		// Check if there are new protoreflect interface methods that are not
		// handled by the formatter. If this fails, either add the method to
		// ignore or add them to descriptorAccessors.
		for s := range want {
			if !got[s] && !ignore[s] {
				t.Errorf("%v.%v is not called by formatter", rt, s)
			}
		}
	}
}
