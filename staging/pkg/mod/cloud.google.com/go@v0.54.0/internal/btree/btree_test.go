// Copyright 2014 Google LLC
// Modified 2018 by Jonathan Amsterdam (jbamsterdam@gmail.com)
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
	"math/rand"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func init() {
	seed := time.Now().Unix()
	rand.Seed(seed)
}

type itemWithIndex struct {
	Key   Key
	Value Value
	Index int
}

// perm returns a random permutation of n Int items in the range [0, n).
func perm(n int) []itemWithIndex {
	var out []itemWithIndex
	for _, v := range rand.Perm(n) {
		out = append(out, itemWithIndex{v, v, v})
	}
	return out
}

// rang returns an ordered list of Int items in the range [0, n).
func rang(n int) []itemWithIndex {
	var out []itemWithIndex
	for i := 0; i < n; i++ {
		out = append(out, itemWithIndex{i, i, i})
	}
	return out
}

// all extracts all items from an iterator.
func all(it *Iterator) []itemWithIndex {
	var out []itemWithIndex
	for it.Next() {
		out = append(out, itemWithIndex{it.Key, it.Value, it.Index})
	}
	return out
}

func reverse(s []itemWithIndex) {
	for i := 0; i < len(s)/2; i++ {
		s[i], s[len(s)-i-1] = s[len(s)-i-1], s[i]
	}
}

var btreeDegree = flag.Int("degree", 32, "B-Tree degree")

func TestBTree(t *testing.T) {
	tr := New(*btreeDegree, less)
	const treeSize = 10000
	for i := 0; i < 10; i++ {
		if min, _ := tr.Min(); min != nil {
			t.Fatalf("empty min, got %+v", min)
		}
		if max, _ := tr.Max(); max != nil {
			t.Fatalf("empty max, got %+v", max)
		}
		for _, m := range perm(treeSize) {
			if _, ok := tr.Set(m.Key, m.Value); ok {
				t.Fatal("set found item", m)
			}
		}
		for _, m := range perm(treeSize) {
			_, ok, idx := tr.SetWithIndex(m.Key, m.Value)
			if !ok {
				t.Fatal("set didn't find item", m)
			}
			if idx != m.Index {
				t.Fatalf("got index %d, want %d", idx, m.Index)
			}
		}
		mink, minv := tr.Min()
		if want := 0; mink != want || minv != want {
			t.Fatalf("min: want %+v, got %+v, %+v", want, mink, minv)
		}
		maxk, maxv := tr.Max()
		if want := treeSize - 1; maxk != want || maxv != want {
			t.Fatalf("max: want %+v, got %+v, %+v", want, maxk, maxv)
		}
		got := all(tr.BeforeIndex(0))
		want := rang(treeSize)
		if !cmp.Equal(got, want) {
			t.Fatalf("mismatch:\n got: %v\nwant: %v", got, want)
		}

		for _, m := range perm(treeSize) {
			if _, removed := tr.Delete(m.Key); !removed {
				t.Fatalf("didn't find %v", m)
			}
		}
		if got = all(tr.BeforeIndex(0)); len(got) > 0 {
			t.Fatalf("some left!: %v", got)
		}
	}
}

func TestAt(t *testing.T) {
	tr := New(*btreeDegree, less)
	for _, m := range perm(100) {
		tr.Set(m.Key, m.Value)
	}
	for i := 0; i < tr.Len(); i++ {
		gotk, gotv := tr.At(i)
		if want := i; gotk != want || gotv != want {
			t.Fatalf("At(%d) = (%v, %v), want (%v, %v)", i, gotk, gotv, want, want)
		}
	}
}

func TestGetWithIndex(t *testing.T) {
	tr := New(*btreeDegree, less)
	for _, m := range perm(100) {
		tr.Set(m.Key, m.Value)
	}
	for i := 0; i < tr.Len(); i++ {
		gotv, goti := tr.GetWithIndex(i)
		wantv, wanti := i, i
		if gotv != wantv || goti != wanti {
			t.Errorf("GetWithIndex(%d) = (%v, %v), want (%v, %v)",
				i, gotv, goti, wantv, wanti)
		}
	}
	_, got := tr.GetWithIndex(100)
	if want := -1; got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestSetWithIndex(t *testing.T) {
	tr := New(4, less) // use a small degree to cover more cases
	var contents []int
	for _, m := range perm(100) {
		_, _, idx := tr.SetWithIndex(m.Key, m.Value)
		contents = append(contents, m.Index)
		sort.Ints(contents)
		want := -1
		for i, c := range contents {
			if c == m.Index {
				want = i
				break
			}
		}
		if idx != want {
			t.Fatalf("got %d, want %d", idx, want)
		}
	}
}

func TestDeleteMin(t *testing.T) {
	tr := New(3, less)
	for _, m := range perm(100) {
		tr.Set(m.Key, m.Value)
	}
	var got []itemWithIndex
	for i := 0; tr.Len() > 0; i++ {
		k, v := tr.DeleteMin()
		got = append(got, itemWithIndex{k, v, i})
	}
	if want := rang(100); !cmp.Equal(got, want) {
		t.Fatalf("got: %v\nwant: %v", got, want)
	}
}

func TestDeleteMax(t *testing.T) {
	tr := New(3, less)
	for _, m := range perm(100) {
		tr.Set(m.Key, m.Value)
	}
	var got []itemWithIndex
	for tr.Len() > 0 {
		k, v := tr.DeleteMax()
		got = append(got, itemWithIndex{k, v, tr.Len()})
	}
	reverse(got)
	if want := rang(100); !cmp.Equal(got, want) {
		t.Fatalf("got: %v\nwant: %v", got, want)
	}
}

func TestIterator(t *testing.T) {
	const size = 10

	tr := New(2, less)
	// Empty tree.
	for i, it := range []*Iterator{
		tr.BeforeIndex(0),
		tr.Before(3),
		tr.After(3),
	} {
		if got, want := it.Next(), false; got != want {
			t.Errorf("empty, #%d: got %t, want %t", i, got, want)
		}
	}

	// Root with zero children.
	tr.Set(1, nil)
	tr.Delete(1)
	if !(tr.root != nil && len(tr.root.children) == 0 && len(tr.root.items) == 0) {
		t.Fatal("wrong shape tree")
	}
	for i, it := range []*Iterator{
		tr.BeforeIndex(0),
		tr.Before(3),
		tr.After(3),
	} {
		if got, want := it.Next(), false; got != want {
			t.Errorf("zero root, #%d: got %t, want %t", i, got, want)
		}
	}

	// Tree with size elements.
	p := perm(size)
	for _, v := range p {
		tr.Set(v.Key, v.Value)
	}

	it := tr.BeforeIndex(0)
	got := all(it)
	want := rang(size)
	if !cmp.Equal(got, want) {
		t.Fatalf("got %+v\nwant %+v\n", got, want)
	}

	for i, w := range want {
		it := tr.Before(w.Key)
		got = all(it)
		wn := want[w.Key.(int):]
		if !cmp.Equal(got, wn) {
			t.Fatalf("got %+v\nwant %+v\n", got, wn)
		}

		it = tr.BeforeIndex(i)
		got = all(it)
		if !cmp.Equal(got, wn) {
			t.Fatalf("got %+v\nwant %+v\n", got, wn)
		}

		it = tr.After(w.Key)
		got = all(it)
		wn = append([]itemWithIndex(nil), want[:w.Key.(int)+1]...)
		reverse(wn)
		if !cmp.Equal(got, wn) {
			t.Fatalf("got %+v\nwant %+v\n", got, wn)
		}

		it = tr.AfterIndex(i)
		got = all(it)
		if !cmp.Equal(got, wn) {
			t.Fatalf("got %+v\nwant %+v\n", got, wn)
		}
	}

	// Non-existent keys.
	tr = New(2, less)
	for _, v := range p {
		tr.Set(v.Key.(int)*2, v.Value)
	}
	// tr has only even keys: 0, 2, 4, ... Iterate from odd keys.
	for i := -1; i <= size+1; i += 2 {
		it := tr.Before(i)
		got := all(it)
		var want []itemWithIndex
		for j := (i + 1) / 2; j < size; j++ {
			want = append(want, itemWithIndex{j * 2, j, j})
		}
		if !cmp.Equal(got, want) {
			tr.print(os.Stdout)
			t.Fatalf("%d: got %+v\nwant %+v\n", i, got, want)
		}

		it = tr.After(i)
		got = all(it)
		want = nil
		for j := (i - 1) / 2; j >= 0; j-- {
			want = append(want, itemWithIndex{j * 2, j, j})
		}
		if !cmp.Equal(got, want) {
			t.Fatalf("%d: got %+v\nwant %+v\n", i, got, want)
		}
	}
}

func TestMixed(t *testing.T) {
	// Test random, mixed insertions and deletions.
	const maxSize = 1000
	tr := New(3, less)
	has := map[int]bool{}
	for i := 0; i < 10000; i++ {
		r := rand.Intn(maxSize)
		if r >= tr.Len() {
			old, ok := tr.Set(r, r)
			if has[r] != ok {
				t.Fatalf("%d: has=%t, ok=%t", r, has[r], ok)
			}
			if ok && old.(int) != r {
				t.Fatalf("%d: bad old", r)
			}
			has[r] = true
			if got, want := tr.Get(r), r; got != want {
				t.Fatalf("Get(%d) = %d, want %d", r, got, want)
			}
		} else {
			// Expoit random map iteration order.
			var d int
			for d = range has {
				break
			}
			old, removed := tr.Delete(d)
			if !removed {
				t.Fatalf("%d not found", d)
			}
			if old.(int) != d {
				t.Fatalf("%d: bad old", d)
			}
			delete(has, d)
		}
	}
}

const cloneTestSize = 10000

func cloneTest(t *testing.T, b *BTree, start int, p []itemWithIndex, wg *sync.WaitGroup, treec chan<- *BTree) {
	treec <- b
	for i := start; i < cloneTestSize; i++ {
		b.Set(p[i].Key, p[i].Value)
		if i%(cloneTestSize/5) == 0 {
			wg.Add(1)
			go cloneTest(t, b.Clone(), i+1, p, wg, treec)
		}
	}
	wg.Done()
}

func TestCloneConcurrentOperations(t *testing.T) {
	b := New(*btreeDegree, less)
	treec := make(chan *BTree)
	p := perm(cloneTestSize)
	var wg sync.WaitGroup
	wg.Add(1)
	go cloneTest(t, b, 0, p, &wg, treec)
	var trees []*BTree
	donec := make(chan struct{})
	go func() {
		for t := range treec {
			trees = append(trees, t)
		}
		close(donec)
	}()
	wg.Wait()
	close(treec)
	<-donec
	want := rang(cloneTestSize)
	for i, tree := range trees {
		if !cmp.Equal(want, all(tree.BeforeIndex(0))) {
			t.Errorf("tree %v mismatch", i)
		}
	}
	toRemove := rang(cloneTestSize)[cloneTestSize/2:]
	for i := 0; i < len(trees)/2; i++ {
		tree := trees[i]
		wg.Add(1)
		go func() {
			for _, m := range toRemove {
				tree.Delete(m.Key)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	for i, tree := range trees {
		var wantpart []itemWithIndex
		if i < len(trees)/2 {
			wantpart = want[:cloneTestSize/2]
		} else {
			wantpart = want
		}
		if got := all(tree.BeforeIndex(0)); !cmp.Equal(wantpart, got) {
			t.Errorf("tree %v mismatch, want %v got %v", i, len(want), len(got))
		}
	}
}

func less(a, b interface{}) bool { return a.(int) < b.(int) }
