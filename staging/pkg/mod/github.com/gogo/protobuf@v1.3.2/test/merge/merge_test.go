package merge

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/gogo/protobuf/proto"
)

func TestClone1(t *testing.T) {
	f1 := &A{}
	proto.Clone(f1)
}

func TestClone2(t *testing.T) {
	f1 := &A{C: []C{{D: 1}, {D: 2}}}
	f2 := proto.Clone(f1).(*A)
	if !reflect.DeepEqual(f1.C, f2.C) {
		t.Fatalf("got %v want %v", f2, f1)
	}
	if fmt.Sprintf("%p", f1.C) == fmt.Sprintf("%p", f2.C) {
		t.Fatalf("slice is not deep copied")
	}
}

func TestMerge1(t *testing.T) {
	f1 := &A{}
	f2 := &A{}
	proto.Merge(f1, f2)
}

func TestMerge2(t *testing.T) {
	f1 := &A{B: B{C: 1}}
	f2 := &A{}
	proto.Merge(f1, f2)
	if f1.B.C != 1 {
		t.Fatalf("got %d want %d", f1.B.C, 1)
	}
}

func TestMerge3(t *testing.T) {
	f1 := &A{}
	f2 := &A{B: B{C: 1}}
	proto.Merge(f1, f2)
	if f1.B.C != 1 {
		t.Fatalf("got %d want %d", f1.B.C, 1)
	}
}

func TestMerge4(t *testing.T) {
	f1 := &A{}
	f2 := &A{C: []C{}}
	proto.Merge(f1, f2)
	if f1.C == nil {
		t.Fatalf("got %v want %v", f1, []C{})
	}
}

func TestMerge5(t *testing.T) {
	f1 := &A{C: []C{{D: 1}, {D: 2}}}
	f2 := &A{C: []C{{D: 3}, {D: 4}}}
	f3 := &A{C: []C{{D: 1}, {D: 2}, {D: 3}, {D: 4}}}
	proto.Merge(f1, f2)
	if !reflect.DeepEqual(f1, f3) {
		t.Fatalf("got %v want %v", f1, f3)
	}
}
