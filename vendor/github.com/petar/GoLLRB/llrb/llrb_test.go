// Copyright 2010 Petar Maymounkov. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package llrb

import (
	"math"
	"math/rand"
	"testing"
)

func TestCases(t *testing.T) {
	tree := New()
	tree.ReplaceOrInsert(Int(1))
	tree.ReplaceOrInsert(Int(1))
	if tree.Len() != 1 {
		t.Errorf("expecting len 1")
	}
	if !tree.Has(Int(1)) {
		t.Errorf("expecting to find key=1")
	}

	tree.Delete(Int(1))
	if tree.Len() != 0 {
		t.Errorf("expecting len 0")
	}
	if tree.Has(Int(1)) {
		t.Errorf("not expecting to find key=1")
	}

	tree.Delete(Int(1))
	if tree.Len() != 0 {
		t.Errorf("expecting len 0")
	}
	if tree.Has(Int(1)) {
		t.Errorf("not expecting to find key=1")
	}
}

func TestReverseInsertOrder(t *testing.T) {
	tree := New()
	n := 100
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(n - i))
	}
	i := 0
	tree.AscendGreaterOrEqual(Int(0), func(item Item) bool {
		i++
		if item.(Int) != Int(i) {
			t.Errorf("bad order: got %d, expect %d", item.(Int), i)
		}
		return true
	})
}

func TestRange(t *testing.T) {
	tree := New()
	order := []String{
		"ab", "aba", "abc", "a", "aa", "aaa", "b", "a-", "a!",
	}
	for _, i := range order {
		tree.ReplaceOrInsert(i)
	}
	k := 0
	tree.AscendRange(String("ab"), String("ac"), func(item Item) bool {
		if k > 3 {
			t.Fatalf("returned more items than expected")
		}
		i1 := order[k]
		i2 := item.(String)
		if i1 != i2 {
			t.Errorf("expecting %s, got %s", i1, i2)
		}
		k++
		return true
	})
}

func TestRandomInsertOrder(t *testing.T) {
	tree := New()
	n := 1000
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	j := 0
	tree.AscendGreaterOrEqual(Int(0), func(item Item) bool {
		if item.(Int) != Int(j) {
			t.Fatalf("bad order")
		}
		j++
		return true
	})
}

func TestRandomReplace(t *testing.T) {
	tree := New()
	n := 100
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	perm = rand.Perm(n)
	for i := 0; i < n; i++ {
		if replaced := tree.ReplaceOrInsert(Int(perm[i])); replaced == nil || replaced.(Int) != Int(perm[i]) {
			t.Errorf("error replacing")
		}
	}
}

func TestRandomInsertSequentialDelete(t *testing.T) {
	tree := New()
	n := 1000
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	for i := 0; i < n; i++ {
		tree.Delete(Int(i))
	}
}

func TestRandomInsertDeleteNonExistent(t *testing.T) {
	tree := New()
	n := 100
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	if tree.Delete(Int(200)) != nil {
		t.Errorf("deleted non-existent item")
	}
	if tree.Delete(Int(-2)) != nil {
		t.Errorf("deleted non-existent item")
	}
	for i := 0; i < n; i++ {
		if u := tree.Delete(Int(i)); u == nil || u.(Int) != Int(i) {
			t.Errorf("delete failed")
		}
	}
	if tree.Delete(Int(200)) != nil {
		t.Errorf("deleted non-existent item")
	}
	if tree.Delete(Int(-2)) != nil {
		t.Errorf("deleted non-existent item")
	}
}

func TestRandomInsertPartialDeleteOrder(t *testing.T) {
	tree := New()
	n := 100
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	for i := 1; i < n-1; i++ {
		tree.Delete(Int(i))
	}
	j := 0
	tree.AscendGreaterOrEqual(Int(0), func(item Item) bool {
		switch j {
		case 0:
			if item.(Int) != Int(0) {
				t.Errorf("expecting 0")
			}
		case 1:
			if item.(Int) != Int(n-1) {
				t.Errorf("expecting %d", n-1)
			}
		}
		j++
		return true
	})
}

func TestRandomInsertStats(t *testing.T) {
	tree := New()
	n := 100000
	perm := rand.Perm(n)
	for i := 0; i < n; i++ {
		tree.ReplaceOrInsert(Int(perm[i]))
	}
	avg, _ := tree.HeightStats()
	expAvg := math.Log2(float64(n)) - 1.5
	if math.Abs(avg-expAvg) >= 2.0 {
		t.Errorf("too much deviation from expected average height")
	}
}

func BenchmarkInsert(b *testing.B) {
	tree := New()
	for i := 0; i < b.N; i++ {
		tree.ReplaceOrInsert(Int(b.N - i))
	}
}

func BenchmarkDelete(b *testing.B) {
	b.StopTimer()
	tree := New()
	for i := 0; i < b.N; i++ {
		tree.ReplaceOrInsert(Int(b.N - i))
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tree.Delete(Int(i))
	}
}

func BenchmarkDeleteMin(b *testing.B) {
	b.StopTimer()
	tree := New()
	for i := 0; i < b.N; i++ {
		tree.ReplaceOrInsert(Int(b.N - i))
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tree.DeleteMin()
	}
}

func TestInsertNoReplace(t *testing.T) {
	tree := New()
	n := 1000
	for q := 0; q < 2; q++ {
		perm := rand.Perm(n)
		for i := 0; i < n; i++ {
			tree.InsertNoReplace(Int(perm[i]))
		}
	}
	j := 0
	tree.AscendGreaterOrEqual(Int(0), func(item Item) bool {
		if item.(Int) != Int(j/2) {
			t.Fatalf("bad order")
		}
		j++
		return true
	})
}
