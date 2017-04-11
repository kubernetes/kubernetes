package topk

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func TestTopK(t *testing.T) {
	stream := New(10)
	ss := []*Stream{New(10), New(10), New(10)}
	m := make(map[string]int)
	for _, s := range ss {
		for i := 0; i < 1e6; i++ {
			v := fmt.Sprintf("%x", int8(rand.ExpFloat64()))
			s.Insert(v)
			m[v]++
		}
		stream.Merge(s.Query())
	}

	var sm Samples
	for x, s := range m {
		sm = append(sm, &Element{x, s})
	}
	sort.Sort(sort.Reverse(sm))

	g := stream.Query()
	if len(g) != 10 {
		t.Fatalf("got %d, want 10", len(g))
	}
	for i, e := range g {
		if sm[i].Value != e.Value {
			t.Errorf("at %d: want %q, got %q", i, sm[i].Value, e.Value)
		}
	}
}

func TestQuery(t *testing.T) {
	queryTests := []struct {
		value string
		expected int
	}{
		{"a", 1},
		{"b", 2},
		{"c", 2},
	}

	stream := New(2)
	for _, tt := range queryTests {
		stream.Insert(tt.value)
		if n := len(stream.Query()); n != tt.expected {
			t.Errorf("want %d, got %d", tt.expected, n)
		}
	}
}
