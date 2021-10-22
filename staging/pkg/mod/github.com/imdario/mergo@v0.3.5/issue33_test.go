package mergo

import (
	"testing"
)

type Foo struct {
	Str    string
	Bslice []byte
}

func TestIssue33Merge(t *testing.T) {
	dest := Foo{Str: "a"}
	toMerge := Foo{
		Str:    "b",
		Bslice: []byte{1, 2},
	}
	if err := Merge(&dest, toMerge); err != nil {
		t.Errorf("Error while merging: %s", err)
	}
	// Merge doesn't overwrite an attribute if in destination it doesn't have a zero value.
	// In this case, Str isn't a zero value string.
	if dest.Str != "a" {
		t.Errorf("dest.Str should have not been override as it has a non-zero value: dest.Str(%v) != 'a'", dest.Str)
	}
	// If we want to override, we must use MergeWithOverwrite or Merge using WithOverride.
	if err := Merge(&dest, toMerge, WithOverride); err != nil {
		t.Errorf("Error while merging: %s", err)
	}
	if dest.Str != toMerge.Str {
		t.Errorf("dest.Str should have been override: dest.Str(%v) != toMerge.Str(%v)", dest.Str, toMerge.Str)
	}
}
