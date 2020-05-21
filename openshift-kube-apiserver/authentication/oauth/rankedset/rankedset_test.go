package rankedset

import "testing"

func TestRankedSet(t *testing.T) {
	s := New()
	a := newTestSetItem("A", 5, "AD")
	b := newTestSetItem("B", 6, "BD")
	c := newTestSetItem("C", 4, "CD")
	d := newTestSetItem("D", 6, "DD")
	e := newTestSetItem("E", 1, "ED")

	for _, tc := range []struct {
		name string
		f    func(*testing.T)
	}{
		{
			name: "insert",
			f: func(t *testing.T) {
				assertLen(s, 0, t)
				s.Insert(a)
				assertLen(s, 1, t)
				s.Insert(b)
				assertLen(s, 2, t)
				s.Insert(c)
				assertLen(s, 3, t)
				s.Insert(d)
				assertLen(s, 4, t)
				s.Insert(e)
				assertLen(s, 5, t)
			},
		},
		{
			name: "list order",
			f: func(t *testing.T) {
				assertOrder(s.List(false), t, e, c, a, b, d)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "delete list order 1",
			f: func(t *testing.T) {
				assertItem(a, s.Delete(a), t)
				assertOrder(s.List(false), t, e, c, b, d)
				assertLen(s, 4, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "delete list order 2",
			f: func(t *testing.T) {
				assertItem(b, s.Delete(b), t)
				assertOrder(s.List(false), t, e, c, d)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "has",
			f: func(t *testing.T) {
				assertHas("A", false, s, t)
				assertHas("B", false, s, t)
				assertHas("C", true, s, t)
				assertHas("D", true, s, t)
				assertHas("E", true, s, t)
				assertHas("F", false, s, t)
			},
		},
		{
			name: "get",
			f: func(t *testing.T) {
				assertItem(nil, s.Get(StringItem("A")), t)
				assertItem(nil, s.Get(StringItem("B")), t)
				assertItem(c, s.Get(StringItem("C")), t)
				assertItem(d, s.Get(StringItem("D")), t)
				assertItem(e, s.Get(StringItem("E")), t)
				assertItem(nil, s.Get(StringItem("F")), t)
			},
		},
		{
			name: "delete list order 3",
			f: func(t *testing.T) {
				assertItem(nil, s.Delete(b), t)
				assertOrder(s.List(false), t, e, c, d)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "delete list order 4",
			f: func(t *testing.T) {
				assertItem(c, s.Delete(c), t)
				assertOrder(s.List(false), t, e, d)
				assertLen(s, 2, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "insert list order",
			f: func(t *testing.T) {
				assertItem(nil, s.Insert(a), t)
				assertOrder(s.List(false), t, e, a, d)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "less than order",
			f: func(t *testing.T) {
				assertOrder(s.LessThan(6, false), t, e, a)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "less than order delete",
			f: func(t *testing.T) {
				assertOrder(s.LessThan(6, true), t, e, a)
				assertLen(s, 1, t)
				assertItem(d, s.Min(), t)
				assertItem(d, s.Max(), t)
			},
		},
		{
			name: "list order delete",
			f: func(t *testing.T) {
				assertOrder(s.List(true), t, d)
				assertLen(s, 0, t)
				assertItem(nil, s.Min(), t)
				assertItem(nil, s.Max(), t)
			},
		},
		{
			name: "insert min max",
			f: func(t *testing.T) {
				assertItem(nil, s.Insert(b), t)
				assertItem(nil, s.Insert(a), t)
				assertItem(nil, s.Insert(e), t)
				assertOrder(s.List(false), t, e, a, b)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(b, s.Max(), t)
				assertItem(e, s.Delete(e), t)
				assertLen(s, 2, t)
				assertItem(a, s.Min(), t)
				assertItem(b, s.Max(), t)
			},
		},
		{
			name: "insert replace",
			f: func(t *testing.T) {
				a0 := newTestSetItem("A", 1, "AD0")
				a1 := newTestSetItem("A", 2, "AD1")
				a2 := newTestSetItem("A", 3, "AD2")

				assertItem(nil, s.Insert(e), t)
				assertOrder(s.List(false), t, e, a, b)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(b, s.Max(), t)

				assertItem(a, s.Insert(a0), t)
				assertOrder(s.List(false), t, a0, e, b)
				assertLen(s, 3, t)
				assertItem(a0, s.Min(), t)
				assertItem(b, s.Max(), t)

				assertItem(a0, s.Insert(a1), t)
				assertOrder(s.List(false), t, e, a1, b)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(b, s.Max(), t)

				assertItem(a1, s.Insert(a2), t)
				assertOrder(s.List(false), t, e, a2, b)
				assertLen(s, 3, t)
				assertItem(e, s.Min(), t)
				assertItem(b, s.Max(), t)
			},
		},
	} {
		t.Run(tc.name, tc.f)
	}
}

func assertLen(s *RankedSet, length int, t *testing.T) {
	if s.Len() != length {
		t.Errorf("%s expected len: %d got %d for %v", t.Name(), length, s.Len(), noPointerItems(s.List(false)))
	}
}

func assertOrder(actual []Item, t *testing.T, items ...*testSetItem) {
	if len(items) != len(actual) {
		t.Errorf("%s expected len: %d got %d for %v and %v", t.Name(), len(items), len(actual), noPointers(items), noPointerItems(actual))
		return
	}
	for i, item := range items {
		if actualItem := actual[i].(*testSetItem); *item != *actualItem {
			t.Errorf("%s expected item: %v got %v for idx %d", t.Name(), *item, *actualItem, i)
		}
	}
}

func assertItem(item *testSetItem, actual Item, t *testing.T) {
	itemNil := item == nil
	actualNil := actual == nil

	if itemNil != actualNil {
		t.Errorf("%s expected or actual is nil: %v vs %v", t.Name(), item, actual)
		return
	}

	if itemNil {
		return
	}

	if actualItem := actual.(*testSetItem); *item != *actualItem {
		t.Errorf("%s expected item: %v got %v", t.Name(), *item, *actualItem)
	}
}

func assertHas(key string, expected bool, s *RankedSet, t *testing.T) {
	if expected != s.Has(StringItem(key)) {
		t.Errorf("%s expected %v for %s with %v", t.Name(), expected, key, noPointerItems(s.List(false)))
	}
}

func newTestSetItem(key string, rank int64, data string) *testSetItem {
	return &testSetItem{
		key:  key,
		rank: rank,
		data: data,
	}
}

type testSetItem struct {
	key  string
	rank int64
	data string
}

func (i *testSetItem) Key() string {
	return i.key
}

func (i *testSetItem) Rank() int64 {
	return i.rank
}

// funcs below make the printing of these slices better

func noPointers(items []*testSetItem) []testSetItem {
	var out []testSetItem
	for _, item := range items {
		out = append(out, *item)
	}
	return out
}

func noPointerItems(items []Item) []testSetItem {
	var out []testSetItem
	for _, item := range items {
		out = append(out, *(item.(*testSetItem)))
	}
	return out
}
