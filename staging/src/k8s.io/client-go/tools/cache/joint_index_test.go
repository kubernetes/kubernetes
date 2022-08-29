/*
Copyright 2022 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestJointIndexer(t *testing.T) {
	testIndexer1 := "test1I_a"
	testIndexer2 := "test2I-a"
	testIndexer3 := "test3i_b"
	testIndexer4 := "test4I-b"
	testIndexer5 := "test5i_c"

	indexers := Indexers{
		testIndexer1: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[:4]}
			return indexes, nil
		},
		testIndexer2: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[4:5]}
			return indexes, nil
		},
		testIndexer3: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[5:6]}
			return indexes, nil
		},
		testIndexer4: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[6:7]}
			return indexes, nil
		},
		testIndexer5: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[7:]}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	store.Add(testIndexer1, testIndexer1)
	store.Add(testIndexer2, testIndexer2)
	store.Add(testIndexer3, testIndexer3)
	store.Add(testIndexer4, testIndexer4)
	store.Add(testIndexer5, testIndexer5)

	tests := []struct {
		name    string
		in1     IndexConditions
		in2     *JointIndexOptions
		out     []interface{}
		wantErr bool
	}{
		{
			name: "[equal][default]",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer2, IndexedValue: "1"},
				{Operator: selection.Equals, IndexName: testIndexer3, IndexedValue: "i"},
				{Operator: selection.Equals, IndexName: testIndexer4, IndexedValue: "-"},
			},
			in2: &JointIndexOptions{},
			out: []interface{}{testIndexer1, testIndexer2, testIndexer3, testIndexer4, testIndexer5},
		},
		{
			name: "[equal][union]",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer2, IndexedValue: "1"},
				{Operator: selection.Equals, IndexName: testIndexer3, IndexedValue: "i"},
				{Operator: selection.Equals, IndexName: testIndexer4, IndexedValue: "-"},
			},
			in2: &JointIndexOptions{IndexType: UnionIndexType},
			out: []interface{}{testIndexer1, testIndexer2, testIndexer3, testIndexer4, testIndexer5},
		},
		{
			name: "[equal][intersection]",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexedValue: "test"},
				{Operator: selection.Equals, IndexName: testIndexer3, IndexedValue: "I"},
				{Operator: selection.Equals, IndexName: testIndexer4, IndexedValue: "-"},
			},
			in2: &JointIndexOptions{IndexType: IntersectionIndexType},
			out: []interface{}{testIndexer2, testIndexer4},
		},
		{
			name: "[double-equal][symmetric-difference]",
			in1: IndexConditions{
				{Operator: selection.DoubleEquals, IndexName: testIndexer1, IndexedValue: "test"},
				{Operator: selection.DoubleEquals, IndexName: testIndexer2, IndexedValue: "1"},
				{Operator: selection.DoubleEquals, IndexName: testIndexer3, IndexedValue: "i"},
			},
			in2: &JointIndexOptions{IndexType: SymmetricDifferenceIndexType},
			out: []interface{}{testIndexer2, testIndexer4},
		},
		{
			name: "[equal and not-equal][intersection]",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexedValue: "test"},
				{Operator: selection.NotEquals, IndexName: testIndexer2, IndexedValue: "1"},
			},
			in2: &JointIndexOptions{IndexType: IntersectionIndexType},
			out: []interface{}{testIndexer2, testIndexer3, testIndexer4, testIndexer5},
		},
		{
			name: "[equal][customize-index-func]",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexedValue: "test"},
				{Operator: selection.Equals, IndexName: testIndexer2, IndexedValue: "1"},
				{Operator: selection.Equals, IndexName: testIndexer3, IndexedValue: "i"},
				{Operator: selection.Equals, IndexName: testIndexer4, IndexedValue: "_"},
			},
			in2: &JointIndexOptions{IndexFunc: func(conds IndexConditions) (sets.String, error) {
				if len(conds) != 4 {
					return sets.String{}, fmt.Errorf("length of conds is not same")
				}
				c1, c2, c3, c4 := conds[0], conds[1], conds[2], conds[3]
				return c1.Intersection(c2).Union(c4.Difference(c3)).Complete(), nil
			}},
			out: []interface{}{testIndexer1},
		},
		// error scenes bellow
		{
			name: "not support operator",
			in1: IndexConditions{
				{Operator: selection.Exists, IndexName: testIndexer1, IndexedValue: "test"},
			},
			wantErr: true,
		},
		{
			name: "not support index type",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexedValue: "test"},
			},
			in2:     &JointIndexOptions{IndexType: JointIndexType("not support type")},
			wantErr: true,
		},
		{
			name: "nil joint index options",
			in1: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexedValue: "test"},
			},
			in2:     nil,
			wantErr: true,
		},
		{
			name:    "less than 1 index conditions",
			in1:     IndexConditions{},
			in2:     &JointIndexOptions{IndexType: SymmetricDifferenceIndexType},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := store.ByIndexes(tt.in1, tt.in2)
			if err != nil {
				if tt.wantErr {
					return
				}
				t.Errorf("ByIndexes() error = %v", err)
				return
			}
			if !hadAllSameElems(got, tt.out) {
				t.Errorf("ByIndexes() got = %v, want = %v", got, tt.out)
			}
		})
	}
}

func hadAllSameElems(a []interface{}, b []interface{}) (same bool) {
	if len(a) != len(b) {
		return false
	}
	if (a == nil) != (b == nil) {
		return false
	}

	// in our sense, elems in slice must be different
	set1 := sets.NewString()
	for _, v := range a {
		set1.Insert(v.(string))
	}
	for _, v := range b {
		if !set1.Has(v.(string)) {
			return false
		}
	}

	return true
}
