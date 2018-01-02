// Copyright 2014 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package btree

import (
	"flag"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"
)

func init() {
	seed := time.Now().Unix()
	fmt.Println(seed)
	rand.Seed(seed)
}

// perm returns a random permutation of n Int items in the range [0, n).
func perm(n int) (out []Item) {
	for _, v := range rand.Perm(n) {
		out = append(out, Int(v))
	}
	return
}

// rang returns an ordered list of Int items in the range [0, n).
func rang(n int) (out []Item) {
	for i := 0; i < n; i++ {
		out = append(out, Int(i))
	}
	return
}

// all extracts all items from a tree in order as a slice.
func all(t *BTree) (out []Item) {
	t.Ascend(func(a Item) bool {
		out = append(out, a)
		return true
	})
	return
}

var btreeDegree = flag.Int("degree", 32, "B-Tree degree")

func TestBTree(t *testing.T) {
	tr := New(*btreeDegree)
	const treeSize = 10000
	for i := 0; i < 10; i++ {
		if min := tr.Min(); min != nil {
			t.Fatalf("empty min, got %+v", min)
		}
		if max := tr.Max(); max != nil {
			t.Fatalf("empty max, got %+v", max)
		}
		for _, item := range perm(treeSize) {
			if x := tr.ReplaceOrInsert(item); x != nil {
				t.Fatal("insert found item", item)
			}
		}
		for _, item := range perm(treeSize) {
			if x := tr.ReplaceOrInsert(item); x == nil {
				t.Fatal("insert didn't find item", item)
			}
		}
		if min, want := tr.Min(), Item(Int(0)); min != want {
			t.Fatalf("min: want %+v, got %+v", want, min)
		}
		if max, want := tr.Max(), Item(Int(treeSize-1)); max != want {
			t.Fatalf("max: want %+v, got %+v", want, max)
		}
		got := all(tr)
		want := rang(treeSize)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("mismatch:\n got: %v\nwant: %v", got, want)
		}
		for _, item := range perm(treeSize) {
			if x := tr.Delete(item); x == nil {
				t.Fatalf("didn't find %v", item)
			}
		}
		if got = all(tr); len(got) > 0 {
			t.Fatalf("some left!: %v", got)
		}
	}
}

func ExampleBTree() {
	tr := New(*btreeDegree)
	for i := Int(0); i < 10; i++ {
		tr.ReplaceOrInsert(i)
	}
	fmt.Println("len:       ", tr.Len())
	fmt.Println("get3:      ", tr.Get(Int(3)))
	fmt.Println("get100:    ", tr.Get(Int(100)))
	fmt.Println("del4:      ", tr.Delete(Int(4)))
	fmt.Println("del100:    ", tr.Delete(Int(100)))
	fmt.Println("replace5:  ", tr.ReplaceOrInsert(Int(5)))
	fmt.Println("replace100:", tr.ReplaceOrInsert(Int(100)))
	fmt.Println("min:       ", tr.Min())
	fmt.Println("delmin:    ", tr.DeleteMin())
	fmt.Println("max:       ", tr.Max())
	fmt.Println("delmax:    ", tr.DeleteMax())
	fmt.Println("len:       ", tr.Len())
	// Output:
	// len:        10
	// get3:       3
	// get100:     <nil>
	// del4:       4
	// del100:     <nil>
	// replace5:   5
	// replace100: <nil>
	// min:        0
	// delmin:     0
	// max:        100
	// delmax:     100
	// len:        8
}

func TestDeleteMin(t *testing.T) {
	tr := New(3)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []Item
	for v := tr.DeleteMin(); v != nil; v = tr.DeleteMin() {
		got = append(got, v)
	}
	if want := rang(100); !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestDeleteMax(t *testing.T) {
	tr := New(3)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []Item
	for v := tr.DeleteMax(); v != nil; v = tr.DeleteMax() {
		got = append(got, v)
	}
	// Reverse our list.
	for i := 0; i < len(got)/2; i++ {
		got[i], got[len(got)-i-1] = got[len(got)-i-1], got[i]
	}
	if want := rang(100); !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestAscendRange(t *testing.T) {
	tr := New(2)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []Item
	tr.AscendRange(Int(40), Int(60), func(a Item) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:60]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendRange(Int(40), Int(60), func(a Item) bool {
		if a.(Int) > 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestAscendLessThan(t *testing.T) {
	tr := New(*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []Item
	tr.AscendLessThan(Int(60), func(a Item) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[:60]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendLessThan(Int(60), func(a Item) bool {
		if a.(Int) > 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

func TestAscendGreaterOrEqual(t *testing.T) {
	tr := New(*btreeDegree)
	for _, v := range perm(100) {
		tr.ReplaceOrInsert(v)
	}
	var got []Item
	tr.AscendGreaterOrEqual(Int(40), func(a Item) bool {
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
	got = got[:0]
	tr.AscendGreaterOrEqual(Int(40), func(a Item) bool {
		if a.(Int) > 50 {
			return false
		}
		got = append(got, a)
		return true
	})
	if want := rang(100)[40:51]; !reflect.DeepEqual(got, want) {
		t.Fatalf("ascendrange:\n got: %v\nwant: %v", got, want)
	}
}

const benchmarkTreeSize = 10000

func BenchmarkInsert(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		tr := New(*btreeDegree)
		for _, item := range insertP {
			tr.ReplaceOrInsert(item)
			i++
			if i >= b.N {
				return
			}
		}
	}
}

func BenchmarkDelete(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New(*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Delete(item)
			i++
			if i >= b.N {
				return
			}
		}
		if tr.Len() > 0 {
			panic(tr.Len())
		}
	}
}

func BenchmarkGet(b *testing.B) {
	b.StopTimer()
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	b.StartTimer()
	i := 0
	for i < b.N {
		b.StopTimer()
		tr := New(*btreeDegree)
		for _, v := range insertP {
			tr.ReplaceOrInsert(v)
		}
		b.StartTimer()
		for _, item := range removeP {
			tr.Get(item)
			i++
			if i >= b.N {
				return
			}
		}
	}
}
