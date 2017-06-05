// Copyright 2015 The etcd Authors
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

package mvcc

import (
	"reflect"
	"testing"
)

func TestKeyIndexGet(t *testing.T) {
	// key: "foo"
	// rev: 16
	// generations:
	//    {empty}
	//    {{14, 0}[1], {14, 1}[2], {16, 0}(t)[3]}
	//    {{8, 0}[1], {10, 0}[2], {12, 0}(t)[3]}
	//    {{2, 0}[1], {4, 0}[2], {6, 0}(t)[3]}
	ki := newTestKeyIndex()
	ki.compact(4, make(map[revision]struct{}))

	tests := []struct {
		rev int64

		wmod   revision
		wcreat revision
		wver   int64
		werr   error
	}{
		{17, revision{}, revision{}, 0, ErrRevisionNotFound},
		{16, revision{}, revision{}, 0, ErrRevisionNotFound},

		// get on generation 3
		{15, revision{14, 1}, revision{14, 0}, 2, nil},
		{14, revision{14, 1}, revision{14, 0}, 2, nil},

		{13, revision{}, revision{}, 0, ErrRevisionNotFound},
		{12, revision{}, revision{}, 0, ErrRevisionNotFound},

		// get on generation 2
		{11, revision{10, 0}, revision{8, 0}, 2, nil},
		{10, revision{10, 0}, revision{8, 0}, 2, nil},
		{9, revision{8, 0}, revision{8, 0}, 1, nil},
		{8, revision{8, 0}, revision{8, 0}, 1, nil},

		{7, revision{}, revision{}, 0, ErrRevisionNotFound},
		{6, revision{}, revision{}, 0, ErrRevisionNotFound},

		// get on generation 1
		{5, revision{4, 0}, revision{2, 0}, 2, nil},
		{4, revision{4, 0}, revision{2, 0}, 2, nil},

		{3, revision{}, revision{}, 0, ErrRevisionNotFound},
		{2, revision{}, revision{}, 0, ErrRevisionNotFound},
		{1, revision{}, revision{}, 0, ErrRevisionNotFound},
		{0, revision{}, revision{}, 0, ErrRevisionNotFound},
	}

	for i, tt := range tests {
		mod, creat, ver, err := ki.get(tt.rev)
		if err != tt.werr {
			t.Errorf("#%d: err = %v, want %v", i, err, tt.werr)
		}
		if mod != tt.wmod {
			t.Errorf("#%d: modified = %+v, want %+v", i, mod, tt.wmod)
		}
		if creat != tt.wcreat {
			t.Errorf("#%d: created = %+v, want %+v", i, creat, tt.wcreat)
		}
		if ver != tt.wver {
			t.Errorf("#%d: version = %d, want %d", i, ver, tt.wver)
		}
	}
}

func TestKeyIndexSince(t *testing.T) {
	ki := newTestKeyIndex()
	ki.compact(4, make(map[revision]struct{}))

	allRevs := []revision{{4, 0}, {6, 0}, {8, 0}, {10, 0}, {12, 0}, {14, 1}, {16, 0}}
	tests := []struct {
		rev int64

		wrevs []revision
	}{
		{17, nil},
		{16, allRevs[6:]},
		{15, allRevs[6:]},
		{14, allRevs[5:]},
		{13, allRevs[5:]},
		{12, allRevs[4:]},
		{11, allRevs[4:]},
		{10, allRevs[3:]},
		{9, allRevs[3:]},
		{8, allRevs[2:]},
		{7, allRevs[2:]},
		{6, allRevs[1:]},
		{5, allRevs[1:]},
		{4, allRevs},
		{3, allRevs},
		{2, allRevs},
		{1, allRevs},
		{0, allRevs},
	}

	for i, tt := range tests {
		revs := ki.since(tt.rev)
		if !reflect.DeepEqual(revs, tt.wrevs) {
			t.Errorf("#%d: revs = %+v, want %+v", i, revs, tt.wrevs)
		}
	}
}

func TestKeyIndexPut(t *testing.T) {
	ki := &keyIndex{key: []byte("foo")}
	ki.put(5, 0)

	wki := &keyIndex{
		key:         []byte("foo"),
		modified:    revision{5, 0},
		generations: []generation{{created: revision{5, 0}, ver: 1, revs: []revision{{main: 5}}}},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}

	ki.put(7, 0)

	wki = &keyIndex{
		key:         []byte("foo"),
		modified:    revision{7, 0},
		generations: []generation{{created: revision{5, 0}, ver: 2, revs: []revision{{main: 5}, {main: 7}}}},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}
}

func TestKeyIndexRestore(t *testing.T) {
	ki := &keyIndex{key: []byte("foo")}
	ki.restore(revision{5, 0}, revision{7, 0}, 2)

	wki := &keyIndex{
		key:         []byte("foo"),
		modified:    revision{7, 0},
		generations: []generation{{created: revision{5, 0}, ver: 2, revs: []revision{{main: 7}}}},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}
}

func TestKeyIndexTombstone(t *testing.T) {
	ki := &keyIndex{key: []byte("foo")}
	ki.put(5, 0)

	err := ki.tombstone(7, 0)
	if err != nil {
		t.Errorf("unexpected tombstone error: %v", err)
	}

	wki := &keyIndex{
		key:         []byte("foo"),
		modified:    revision{7, 0},
		generations: []generation{{created: revision{5, 0}, ver: 2, revs: []revision{{main: 5}, {main: 7}}}, {}},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}

	ki.put(8, 0)
	ki.put(9, 0)
	err = ki.tombstone(15, 0)
	if err != nil {
		t.Errorf("unexpected tombstone error: %v", err)
	}

	wki = &keyIndex{
		key:      []byte("foo"),
		modified: revision{15, 0},
		generations: []generation{
			{created: revision{5, 0}, ver: 2, revs: []revision{{main: 5}, {main: 7}}},
			{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 9}, {main: 15}}},
			{},
		},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}

	err = ki.tombstone(16, 0)
	if err != ErrRevisionNotFound {
		t.Errorf("tombstone error = %v, want %v", err, ErrRevisionNotFound)
	}
}

func TestKeyIndexCompact(t *testing.T) {
	tests := []struct {
		compact int64

		wki *keyIndex
		wam map[revision]struct{}
	}{
		{
			1,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{2, 0}, ver: 3, revs: []revision{{main: 2}, {main: 4}, {main: 6}}},
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{},
		},
		{
			2,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{2, 0}, ver: 3, revs: []revision{{main: 2}, {main: 4}, {main: 6}}},
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 2}: {},
			},
		},
		{
			3,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{2, 0}, ver: 3, revs: []revision{{main: 2}, {main: 4}, {main: 6}}},
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 2}: {},
			},
		},
		{
			4,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{2, 0}, ver: 3, revs: []revision{{main: 4}, {main: 6}}},
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 4}: {},
			},
		},
		{
			5,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{2, 0}, ver: 3, revs: []revision{{main: 4}, {main: 6}}},
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 4}: {},
			},
		},
		{
			6,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{},
		},
		{
			7,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{},
		},
		{
			8,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 8}: {},
			},
		},
		{
			9,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 8}, {main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 8}: {},
			},
		},
		{
			10,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 10}: {},
			},
		},
		{
			11,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{8, 0}, ver: 3, revs: []revision{{main: 10}, {main: 12}}},
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 10}: {},
			},
		},
		{
			12,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{},
		},
		{
			13,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14}, {main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{},
		},
		{
			14,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 14, sub: 1}: {},
			},
		},
		{
			15,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{created: revision{14, 0}, ver: 3, revs: []revision{{main: 14, sub: 1}, {main: 16}}},
					{},
				},
			},
			map[revision]struct{}{
				revision{main: 14, sub: 1}: {},
			},
		},
		{
			16,
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{16, 0},
				generations: []generation{
					{},
				},
			},
			map[revision]struct{}{},
		},
	}

	// Continuous Compaction
	ki := newTestKeyIndex()
	for i, tt := range tests {
		am := make(map[revision]struct{})
		ki.compact(tt.compact, am)
		if !reflect.DeepEqual(ki, tt.wki) {
			t.Errorf("#%d: ki = %+v, want %+v", i, ki, tt.wki)
		}
		if !reflect.DeepEqual(am, tt.wam) {
			t.Errorf("#%d: am = %+v, want %+v", i, am, tt.wam)
		}
	}

	// Jump Compaction
	ki = newTestKeyIndex()
	for i, tt := range tests {
		if (i%2 == 0 && i < 6) || (i%2 == 1 && i > 6) {
			am := make(map[revision]struct{})
			ki.compact(tt.compact, am)
			if !reflect.DeepEqual(ki, tt.wki) {
				t.Errorf("#%d: ki = %+v, want %+v", i, ki, tt.wki)
			}
			if !reflect.DeepEqual(am, tt.wam) {
				t.Errorf("#%d: am = %+v, want %+v", i, am, tt.wam)
			}
		}
	}

	// Once Compaction
	for i, tt := range tests {
		ki := newTestKeyIndex()
		am := make(map[revision]struct{})
		ki.compact(tt.compact, am)
		if !reflect.DeepEqual(ki, tt.wki) {
			t.Errorf("#%d: ki = %+v, want %+v", i, ki, tt.wki)
		}
		if !reflect.DeepEqual(am, tt.wam) {
			t.Errorf("#%d: am = %+v, want %+v", i, am, tt.wam)
		}
	}
}

// test that compact on version that higher than last modified version works well
func TestKeyIndexCompactOnFurtherRev(t *testing.T) {
	ki := &keyIndex{key: []byte("foo")}
	ki.put(1, 0)
	ki.put(2, 0)
	am := make(map[revision]struct{})
	ki.compact(3, am)

	wki := &keyIndex{
		key:      []byte("foo"),
		modified: revision{2, 0},
		generations: []generation{
			{created: revision{1, 0}, ver: 2, revs: []revision{{main: 2}}},
		},
	}
	wam := map[revision]struct{}{
		revision{main: 2}: {},
	}
	if !reflect.DeepEqual(ki, wki) {
		t.Errorf("ki = %+v, want %+v", ki, wki)
	}
	if !reflect.DeepEqual(am, wam) {
		t.Errorf("am = %+v, want %+v", am, wam)
	}
}

func TestKeyIndexIsEmpty(t *testing.T) {
	tests := []struct {
		ki *keyIndex
		w  bool
	}{
		{
			&keyIndex{
				key:         []byte("foo"),
				generations: []generation{{}},
			},
			true,
		},
		{
			&keyIndex{
				key:      []byte("foo"),
				modified: revision{2, 0},
				generations: []generation{
					{created: revision{1, 0}, ver: 2, revs: []revision{{main: 2}}},
				},
			},
			false,
		},
	}
	for i, tt := range tests {
		g := tt.ki.isEmpty()
		if g != tt.w {
			t.Errorf("#%d: isEmpty = %v, want %v", i, g, tt.w)
		}
	}
}

func TestKeyIndexFindGeneration(t *testing.T) {
	ki := newTestKeyIndex()

	tests := []struct {
		rev int64
		wg  *generation
	}{
		{0, nil},
		{1, nil},
		{2, &ki.generations[0]},
		{3, &ki.generations[0]},
		{4, &ki.generations[0]},
		{5, &ki.generations[0]},
		{6, nil},
		{7, nil},
		{8, &ki.generations[1]},
		{9, &ki.generations[1]},
		{10, &ki.generations[1]},
		{11, &ki.generations[1]},
		{12, nil},
		{13, nil},
	}
	for i, tt := range tests {
		g := ki.findGeneration(tt.rev)
		if g != tt.wg {
			t.Errorf("#%d: generation = %+v, want %+v", i, g, tt.wg)
		}
	}
}

func TestKeyIndexLess(t *testing.T) {
	ki := &keyIndex{key: []byte("foo")}

	tests := []struct {
		ki *keyIndex
		w  bool
	}{
		{&keyIndex{key: []byte("doo")}, false},
		{&keyIndex{key: []byte("foo")}, false},
		{&keyIndex{key: []byte("goo")}, true},
	}
	for i, tt := range tests {
		g := ki.Less(tt.ki)
		if g != tt.w {
			t.Errorf("#%d: Less = %v, want %v", i, g, tt.w)
		}
	}
}

func TestGenerationIsEmpty(t *testing.T) {
	tests := []struct {
		g *generation
		w bool
	}{
		{nil, true},
		{&generation{}, true},
		{&generation{revs: []revision{{main: 1}}}, false},
	}
	for i, tt := range tests {
		g := tt.g.isEmpty()
		if g != tt.w {
			t.Errorf("#%d: isEmpty = %v, want %v", i, g, tt.w)
		}
	}
}

func TestGenerationWalk(t *testing.T) {
	g := &generation{
		ver:     3,
		created: revision{2, 0},
		revs:    []revision{{main: 2}, {main: 4}, {main: 6}},
	}
	tests := []struct {
		f  func(rev revision) bool
		wi int
	}{
		{func(rev revision) bool { return rev.main >= 7 }, 2},
		{func(rev revision) bool { return rev.main >= 6 }, 1},
		{func(rev revision) bool { return rev.main >= 5 }, 1},
		{func(rev revision) bool { return rev.main >= 4 }, 0},
		{func(rev revision) bool { return rev.main >= 3 }, 0},
		{func(rev revision) bool { return rev.main >= 2 }, -1},
	}
	for i, tt := range tests {
		idx := g.walk(tt.f)
		if idx != tt.wi {
			t.Errorf("#%d: index = %d, want %d", i, idx, tt.wi)
		}
	}
}

func newTestKeyIndex() *keyIndex {
	// key: "foo"
	// rev: 16
	// generations:
	//    {empty}
	//    {{14, 0}[1], {14, 1}[2], {16, 0}(t)[3]}
	//    {{8, 0}[1], {10, 0}[2], {12, 0}(t)[3]}
	//    {{2, 0}[1], {4, 0}[2], {6, 0}(t)[3]}

	ki := &keyIndex{key: []byte("foo")}
	ki.put(2, 0)
	ki.put(4, 0)
	ki.tombstone(6, 0)
	ki.put(8, 0)
	ki.put(10, 0)
	ki.tombstone(12, 0)
	ki.put(14, 0)
	ki.put(14, 1)
	ki.tombstone(16, 0)
	return ki
}
