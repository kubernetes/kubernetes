package json

import (
	"reflect"
	"testing"
)

func TestLookupPtrToStruct(t *testing.T) {
	type Elem struct {
		Key   string
		Value string
	}
	type Outer struct {
		Inner []Elem `json:"inner" patchStrategy:"merge" patchMergeKey:"key"`
	}
	outer := &Outer{}
	elemType, patchStrategies, patchMergeKey, err := LookupPatchMetadata(reflect.TypeOf(outer), "inner")
	if err != nil {
		t.Fatal(err)
	}
	if elemType != reflect.TypeOf([]Elem{}) {
		t.Errorf("elemType = %v, want: %v", elemType, reflect.TypeOf([]Elem{}))
	}
	if !reflect.DeepEqual(patchStrategies, []string{"merge"}) {
		t.Errorf("patchStrategies = %v, want: %v", patchStrategies, []string{"merge"})
	}
	if patchMergeKey != "key" {
		t.Errorf("patchMergeKey = %v, want: %v", patchMergeKey, "key")
	}
}
