/*
Copyright 2018 The Kubernetes Authors.

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

package fieldpath

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"sigs.k8s.io/structured-merge-diff/value"
)

type randomPathAlphabet []PathElement

func (a randomPathAlphabet) makePath(minLen, maxLen int) Path {
	n := minLen
	if minLen < maxLen {
		n += rand.Intn(maxLen - minLen)
	}
	var p Path
	for i := 0; i < n; i++ {
		p = append(p, a[rand.Intn(len(a))])
	}
	return p
}

var randomPathMaker = randomPathAlphabet(MakePathOrDie(
	"aaa",
	"aab",
	"aac",
	"aad",
	"aae",
	"aaf",
	KeyByFields("name", value.StringValue("first")),
	KeyByFields("name", value.StringValue("second")),
	KeyByFields("port", value.IntValue(443), "protocol", value.StringValue("tcp")),
	KeyByFields("port", value.IntValue(443), "protocol", value.StringValue("udp")),
	value.IntValue(1),
	value.IntValue(2),
	value.IntValue(3),
	value.StringValue("aa"),
	value.StringValue("ab"),
	value.BooleanValue(true),
	1, 2, 3, 4,
))

func BenchmarkFieldSet(b *testing.B) {
	cases := []struct {
		size       int
		minPathLen int
		maxPathLen int
	}{
		//{10, 1, 2},
		{20, 2, 3},
		{50, 2, 4},
		{100, 3, 6},
		{500, 3, 7},
		{1000, 3, 8},
	}
	for i := range cases {
		here := cases[i]
		makeSet := func() *Set {
			x := NewSet()
			for j := 0; j < here.size; j++ {
				x.Insert(randomPathMaker.makePath(here.minPathLen, here.maxPathLen))
			}
			return x
		}
		operands := make([]*Set, 500)
		serialized := make([][]byte, len(operands))
		for i := range operands {
			operands[i] = makeSet()
			serialized[i], _ = operands[i].ToJSON()
		}
		randOperand := func() *Set { return operands[rand.Intn(len(operands))] }

		b.Run(fmt.Sprintf("insert-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				makeSet()
			}
		})
		b.Run(fmt.Sprintf("has-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				randOperand().Has(randomPathMaker.makePath(here.minPathLen, here.maxPathLen))
			}
		})
		b.Run(fmt.Sprintf("serialize-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				randOperand().ToJSON()
			}
		})
		b.Run(fmt.Sprintf("deserialize-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			s := NewSet()
			for i := 0; i < b.N; i++ {
				s.FromJSON(bytes.NewReader(serialized[rand.Intn(len(serialized))]))
			}
		})

		b.Run(fmt.Sprintf("union-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				randOperand().Union(randOperand())
			}
		})
		b.Run(fmt.Sprintf("intersection-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				randOperand().Intersection(randOperand())
			}
		})
		b.Run(fmt.Sprintf("difference-%v", here.size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				randOperand().Difference(randOperand())
			}
		})
	}
}

func TestSetInsertHas(t *testing.T) {
	s1 := NewSet(
		MakePathOrDie("foo", 0, "bar", "baz"),
		MakePathOrDie("foo", 0, "bar"),
		MakePathOrDie("foo", 0),
		MakePathOrDie("foo", 1, "bar", "baz"),
		MakePathOrDie("foo", 1, "bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first")), "bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("second")), "bar"),
		MakePathOrDie("canonicalOrder", KeyByFields(
			"a", value.StringValue("a"),
			"b", value.StringValue("a"),
			"c", value.StringValue("a"),
			"d", value.StringValue("a"),
			"e", value.StringValue("a"),
			"f", value.StringValue("a"),
		)),
	)

	table := []struct {
		set              *Set
		check            Path
		expectMembership bool
	}{
		{s1, MakePathOrDie("qux", KeyByFields("name", value.StringValue("second"))), false},
		{s1, MakePathOrDie("qux", KeyByFields("name", value.StringValue("second")), "bar"), true},
		{s1, MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))), true},
		{s1, MakePathOrDie("xuq", KeyByFields("name", value.StringValue("first"))), false},
		{s1, MakePathOrDie("foo", 0), true},
		{s1, MakePathOrDie("foo", 0, "bar"), true},
		{s1, MakePathOrDie("foo", 0, "bar", "baz"), true},
		{s1, MakePathOrDie("foo", 1), false},
		{s1, MakePathOrDie("foo", 1, "bar"), true},
		{s1, MakePathOrDie("foo", 1, "bar", "baz"), true},
		{s1, MakePathOrDie("canonicalOrder", KeyByFields(
			"f", value.StringValue("a"),
			"e", value.StringValue("a"),
			"d", value.StringValue("a"),
			"c", value.StringValue("a"),
			"b", value.StringValue("a"),
			"a", value.StringValue("a"),
		)), true},
	}

	for _, tt := range table {
		got := tt.set.Has(tt.check)
		if e, a := tt.expectMembership, got; e != a {
			t.Errorf("%v: wanted %v, got %v", tt.check.String(), e, a)
		}
	}

	if NewSet().Has(Path{}) {
		t.Errorf("empty set should not include the empty path")
	}
	if NewSet(Path{}).Has(Path{}) {
		t.Errorf("empty set should not include the empty path")
	}
}

func TestSetString(t *testing.T) {
	p := MakePathOrDie("foo", PathElement{Key: KeyByFields("name", value.StringValue("first"))})
	s1 := NewSet(p)

	if p.String() != s1.String() {
		t.Errorf("expected single entry set to just call the path's string, but got %s %s", p, s1)
	}
}

func TestSetIterSize(t *testing.T) {
	s1 := NewSet(
		MakePathOrDie("foo", 0, "bar", "baz"),
		MakePathOrDie("foo", 0, "bar", "zot"),
		MakePathOrDie("foo", 0, "bar"),
		MakePathOrDie("foo", 0),
		MakePathOrDie("foo", 1, "bar", "baz"),
		MakePathOrDie("foo", 1, "bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first")), "bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("second")), "bar"),
	)

	s2 := NewSet()

	addedCount := 0
	s1.Iterate(func(p Path) {
		if s2.Size() != addedCount {
			t.Errorf("added %v items to set, but size is %v", addedCount, s2.Size())
		}
		if addedCount > 0 == s2.Empty() {
			t.Errorf("added %v items to set, but s2.Empty() is %v", addedCount, s2.Empty())
		}
		s2.Insert(p)
		addedCount++
	})

	if !s1.Equals(s2) {
		// No point in using String() if iterate is broken...
		t.Errorf("Iterate missed something?\n%#v\n%#v", s1, s2)
	}
}

func TestSetEquals(t *testing.T) {
	table := []struct {
		a     *Set
		b     *Set
		equal bool
	}{
		{
			a:     NewSet(MakePathOrDie("foo")),
			b:     NewSet(MakePathOrDie("bar")),
			equal: false,
		},
		{
			a:     NewSet(MakePathOrDie("foo")),
			b:     NewSet(MakePathOrDie("foo")),
			equal: true,
		},
		{
			a:     NewSet(),
			b:     NewSet(MakePathOrDie(0, "foo")),
			equal: false,
		},
		{
			a:     NewSet(MakePathOrDie(1, "foo")),
			b:     NewSet(MakePathOrDie(0, "foo")),
			equal: false,
		},
		{
			a:     NewSet(MakePathOrDie(1, "foo")),
			b:     NewSet(MakePathOrDie(1, "foo", "bar")),
			equal: false,
		},
		{
			a: NewSet(
				MakePathOrDie(0),
				MakePathOrDie(1),
			),
			b: NewSet(
				MakePathOrDie(1),
				MakePathOrDie(0),
			),
			equal: true,
		},
		{
			a: NewSet(
				MakePathOrDie("foo", 0),
				MakePathOrDie("foo", 1),
			),
			b: NewSet(
				MakePathOrDie("foo", 1),
				MakePathOrDie("foo", 0),
			),
			equal: true,
		},
		{
			a: NewSet(
				MakePathOrDie("foo", 0),
				MakePathOrDie("foo"),
				MakePathOrDie("bar", "baz"),
				MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))),
			),
			b: NewSet(
				MakePathOrDie("foo", 1),
				MakePathOrDie("bar", "baz"),
				MakePathOrDie("bar"),
				MakePathOrDie("qux", KeyByFields("name", value.StringValue("second"))),
			),
			equal: false,
		},
	}

	for _, tt := range table {
		if e, a := tt.equal, tt.a.Equals(tt.b); e != a {
			t.Errorf("expected %v, got %v for:\na=\n%v\nb=\n%v", e, a, tt.a, tt.b)
		}
	}
}

func TestSetUnion(t *testing.T) {
	// Even though this is not a table driven test, since the thing under
	// test is recursive, we should be able to craft a single input that is
	// sufficient to check all code paths.

	s1 := NewSet(
		MakePathOrDie("foo", 0),
		MakePathOrDie("foo"),
		MakePathOrDie("bar", "baz"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))),
		MakePathOrDie("parent", "child", "grandchild"),
	)

	s2 := NewSet(
		MakePathOrDie("foo", 1),
		MakePathOrDie("bar", "baz"),
		MakePathOrDie("bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("second"))),
		MakePathOrDie("parent", "child"),
	)

	u := NewSet(
		MakePathOrDie("foo", 0),
		MakePathOrDie("foo", 1),
		MakePathOrDie("foo"),
		MakePathOrDie("bar", "baz"),
		MakePathOrDie("bar"),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("first"))),
		MakePathOrDie("qux", KeyByFields("name", value.StringValue("second"))),
		MakePathOrDie("parent", "child"),
		MakePathOrDie("parent", "child", "grandchild"),
	)

	got := s1.Union(s2)

	if !got.Equals(u) {
		t.Errorf("union: expected: \n%v\n, got \n%v\n", u, got)
	}
}

func TestSetIntersectionDifference(t *testing.T) {
	// Even though this is not a table driven test, since the thing under
	// test is recursive, we should be able to craft a single input that is
	// sufficient to check all code paths.

	nameFirst := KeyByFields("name", value.StringValue("first"))
	s1 := NewSet(
		MakePathOrDie("a0"),
		MakePathOrDie("a1"),
		MakePathOrDie("foo", 0),
		MakePathOrDie("foo", 1),
		MakePathOrDie("b0", nameFirst),
		MakePathOrDie("b1", nameFirst),
		MakePathOrDie("bar", "c0"),

		MakePathOrDie("cp", nameFirst, "child"),
	)

	s2 := NewSet(
		MakePathOrDie("a1"),
		MakePathOrDie("a2"),
		MakePathOrDie("foo", 1),
		MakePathOrDie("foo", 2),
		MakePathOrDie("b1", nameFirst),
		MakePathOrDie("b2", nameFirst),
		MakePathOrDie("bar", "c2"),

		MakePathOrDie("cp", nameFirst),
	)
	t.Logf("s1:\n%v\n", s1)
	t.Logf("s2:\n%v\n", s2)

	t.Run("intersection", func(t *testing.T) {
		i := NewSet(
			MakePathOrDie("a1"),
			MakePathOrDie("foo", 1),
			MakePathOrDie("b1", nameFirst),
		)

		got := s1.Intersection(s2)
		if !got.Equals(i) {
			t.Errorf("expected: \n%v\n, got \n%v\n", i, got)
		}
	})

	t.Run("s1 - s2", func(t *testing.T) {
		sDiffS2 := NewSet(
			MakePathOrDie("a0"),
			MakePathOrDie("foo", 0),
			MakePathOrDie("b0", nameFirst),
			MakePathOrDie("bar", "c0"),
			MakePathOrDie("cp", nameFirst, "child"),
		)

		got := s1.Difference(s2)
		if !got.Equals(sDiffS2) {
			t.Errorf("expected: \n%v\n, got \n%v\n", sDiffS2, got)
		}
	})

	t.Run("s2 - s1", func(t *testing.T) {
		s2DiffS := NewSet(
			MakePathOrDie("a2"),
			MakePathOrDie("foo", 2),
			MakePathOrDie("b2", nameFirst),
			MakePathOrDie("bar", "c2"),
			MakePathOrDie("cp", nameFirst),
		)

		got := s2.Difference(s1)
		if !got.Equals(s2DiffS) {
			t.Errorf("expected: \n%v\n, got \n%v\n", s2DiffS, got)
		}
	})

	t.Run("intersection (the hard way)", func(t *testing.T) {
		i := NewSet(
			MakePathOrDie("a1"),
			MakePathOrDie("foo", 1),
			MakePathOrDie("b1", nameFirst),
		)

		// We can construct Intersection out of two union and
		// three difference calls.
		u := s1.Union(s2)
		t.Logf("s1 u s2:\n%v\n", u)
		notIntersection := s2.Difference(s1).Union(s1.Difference(s2))
		t.Logf("s1 !i s2:\n%v\n", notIntersection)
		got := u.Difference(notIntersection)
		if !got.Equals(i) {
			t.Errorf("expected: \n%v\n, got \n%v\n", i, got)
		}
	})
}

func TestSetNodeMapIterate(t *testing.T) {
	set := &SetNodeMap{}
	toAdd := 5
	addedElements := make([]string, toAdd)
	for i := 0; i < toAdd; i++ {
		p := i
		pe := PathElement{Index: &p}
		addedElements[i] = pe.String()
		_ = set.Descend(pe)
	}

	iteratedElements := make(map[string]bool, toAdd)
	set.Iterate(func(pe PathElement) {
		iteratedElements[pe.String()] = true
	})

	if len(iteratedElements) != toAdd {
		t.Errorf("expected %v elements to be iterated over, got %v", toAdd, len(iteratedElements))
	}
	for _, pe := range addedElements {
		if _, ok := iteratedElements[pe]; !ok {
			t.Errorf("expected to have iterated over %v, but never did", pe)
		}
	}
}
