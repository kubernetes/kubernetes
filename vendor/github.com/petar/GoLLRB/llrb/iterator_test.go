package llrb

import (
	"reflect"
	"testing"
)

func TestAscendGreaterOrEqual(t *testing.T) {
	tree := New()
	tree.InsertNoReplace(Int(4))
	tree.InsertNoReplace(Int(6))
	tree.InsertNoReplace(Int(1))
	tree.InsertNoReplace(Int(3))
	var ary []Item
	tree.AscendGreaterOrEqual(Int(-1), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected := []Item{Int(1), Int(3), Int(4), Int(6)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
	ary = nil
	tree.AscendGreaterOrEqual(Int(3), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected = []Item{Int(3), Int(4), Int(6)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
	ary = nil
	tree.AscendGreaterOrEqual(Int(2), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected = []Item{Int(3), Int(4), Int(6)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
}

func TestDescendLessOrEqual(t *testing.T) {
	tree := New()
	tree.InsertNoReplace(Int(4))
	tree.InsertNoReplace(Int(6))
	tree.InsertNoReplace(Int(1))
	tree.InsertNoReplace(Int(3))
	var ary []Item
	tree.DescendLessOrEqual(Int(10), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected := []Item{Int(6), Int(4), Int(3), Int(1)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
	ary = nil
	tree.DescendLessOrEqual(Int(4), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected = []Item{Int(4), Int(3), Int(1)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
	ary = nil
	tree.DescendLessOrEqual(Int(5), func(i Item) bool {
		ary = append(ary, i)
		return true
	})
	expected = []Item{Int(4), Int(3), Int(1)}
	if !reflect.DeepEqual(ary, expected) {
		t.Errorf("expected %v but got %v", expected, ary)
	}
}
