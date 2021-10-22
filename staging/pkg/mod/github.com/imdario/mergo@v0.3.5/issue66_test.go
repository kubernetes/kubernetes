package mergo

import (
	"testing"
)

type PrivateSliceTest66 struct {
	PublicStrings  []string
	privateStrings []string
}

func TestPrivateSlice(t *testing.T) {
	p1 := PrivateSliceTest66{
		PublicStrings:  []string{"one", "two", "three"},
		privateStrings: []string{"four", "five"},
	}
	p2 := PrivateSliceTest66{
		PublicStrings: []string{"six", "seven"},
	}
	if err := Merge(&p1, p2); err != nil {
		t.Fatalf("Error during the merge: %v", err)
	}
	if len(p1.PublicStrings) != 3 {
		t.Error("5 elements should be in 'PublicStrings' field")
	}
	if len(p1.privateStrings) != 2 {
		t.Error("2 elements should be in 'privateStrings' field")
	}
}

func TestPrivateSliceWithAppendSlice(t *testing.T) {
	p1 := PrivateSliceTest66{
		PublicStrings:  []string{"one", "two", "three"},
		privateStrings: []string{"four", "five"},
	}
	p2 := PrivateSliceTest66{
		PublicStrings: []string{"six", "seven"},
	}
	if err := Merge(&p1, p2, WithAppendSlice); err != nil {
		t.Fatalf("Error during the merge: %v", err)
	}
	if len(p1.PublicStrings) != 5 {
		t.Error("5 elements should be in 'PublicStrings' field")
	}
	if len(p1.privateStrings) != 2 {
		t.Error("2 elements should be in 'privateStrings' field")
	}
}
