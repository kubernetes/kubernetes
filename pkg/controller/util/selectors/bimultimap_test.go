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

package selectors

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	pkglabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
)

func TestAssociations(t *testing.T) {
	cases := []struct {
		name                string
		ops                 []operation
		want                []expectation
		testAllPermutations bool
	}{{
		name: "single association",
		ops: []operation{
			putSelectingObject(key("hpa"), selector("a", "1")),
			putLabeledObject(key("pod"), labels("a", "1")),
		},
		want: []expectation{
			forwardSelect(key("hpa"), key("pod")),
			reverseSelect(key("pod"), key("hpa")),
		},
		testAllPermutations: true,
	}, {
		name: "multiple associations from a selecting object",
		ops: []operation{
			putSelectingObject(key("hpa"), selector("a", "1")),
			putLabeledObject(key("pod-1"), labels("a", "1")),
			putLabeledObject(key("pod-2"), labels("a", "1")),
		},
		want: []expectation{
			forwardSelect(key("hpa"), key("pod-1"), key("pod-2")),
			reverseSelect(key("pod-1"), key("hpa")),
			reverseSelect(key("pod-2"), key("hpa")),
		},
		testAllPermutations: true,
	}, {
		name: "multiple associations to a labeled object",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putSelectingObject(key("hpa-2"), selector("a", "1")),
			putLabeledObject(key("pod"), labels("a", "1")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod")),
			forwardSelect(key("hpa-2"), key("pod")),
			reverseSelect(key("pod"), key("hpa-1"), key("hpa-2")),
		},
		testAllPermutations: true,
	}, {
		name: "disjoint association sets",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putSelectingObject(key("hpa-2"), selector("a", "2")),
			putLabeledObject(key("pod-1"), labels("a", "1")),
			putLabeledObject(key("pod-2"), labels("a", "2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod-1")),
			forwardSelect(key("hpa-2"), key("pod-2")),
			reverseSelect(key("pod-1"), key("hpa-1")),
			reverseSelect(key("pod-2"), key("hpa-2")),
		},
		testAllPermutations: true,
	}, {
		name: "separate label cache paths",
		ops: []operation{
			putSelectingObject(key("hpa"), selector("a", "1")),
			putLabeledObject(key("pod-1"), labels("a", "1", "b", "2")),
			putLabeledObject(key("pod-2"), labels("a", "1", "b", "3")),
		},
		want: []expectation{
			forwardSelect(key("hpa"), key("pod-1"), key("pod-2")),
			reverseSelect(key("pod-1"), key("hpa")),
			reverseSelect(key("pod-2"), key("hpa")),
		},
		testAllPermutations: true,
	}, {
		name: "separate selector cache paths",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putSelectingObject(key("hpa-2"), selector("b", "2")),
			putLabeledObject(key("pod"), labels("a", "1", "b", "2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod")),
			forwardSelect(key("hpa-2"), key("pod")),
			reverseSelect(key("pod"), key("hpa-1"), key("hpa-2")),
		},
		testAllPermutations: true,
	}, {
		name: "selection in different namespaces",
		ops: []operation{
			putLabeledObject(key("pod-1", "namespace-1"), labels("a", "1")),
			putLabeledObject(key("pod-1", "namespace-2"), labels("a", "1")),
			putSelectingObject(key("hpa-1", "namespace-2"), selector("a", "1")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1", "namespace-1")), // selects nothing
			forwardSelect(key("hpa-1", "namespace-2"), key("pod-1", "namespace-2")),
			reverseSelect(key("pod-1", "namespace-1")), // selects nothing
			reverseSelect(key("pod-1", "namespace-2"), key("hpa-1", "namespace-2")),
		},
		testAllPermutations: true,
	}, {
		name: "update labeled objects",
		ops: []operation{
			putLabeledObject(key("pod-1"), labels("a", "1")),
			putSelectingObject(key("hpa-1"), selector("a", "2")),
			putLabeledObject(key("pod-1"), labels("a", "2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod-1")),
			reverseSelect(key("pod-1"), key("hpa-1")),
		},
	}, {
		name: "update selecting objects",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putLabeledObject(key("pod-1"), labels("a", "2")),
			putSelectingObject(key("hpa-1"), selector("a", "2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod-1")),
			reverseSelect(key("pod-1"), key("hpa-1")),
		},
	}, {
		name: "keep only labeled objects",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putLabeledObject(key("pod-1"), labels("a", "1")),
			putLabeledObject(key("pod-2"), labels("a", "1")),
			putLabeledObject(key("pod-3"), labels("a", "1")),
			keepOnly(key("pod-1"), key("pod-2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod-1"), key("pod-2")),
			reverseSelect(key("pod-1"), key("hpa-1")),
			reverseSelect(key("pod-2"), key("hpa-1")),
		},
	}, {
		name: "keep only selecting objects",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putSelectingObject(key("hpa-2"), selector("a", "1")),
			putSelectingObject(key("hpa-3"), selector("a", "1")),
			putLabeledObject(key("pod-1"), labels("a", "1")),
			keepOnlySelectors(key("hpa-1"), key("hpa-2")),
		},
		want: []expectation{
			forwardSelect(key("hpa-1"), key("pod-1")),
			forwardSelect(key("hpa-2"), key("pod-1")),
			reverseSelect(key("pod-1"), key("hpa-1"), key("hpa-2")),
		},
	}, {
		name: "put multiple associations and delete all",
		ops: []operation{
			putSelectingObject(key("hpa-1"), selector("a", "1")),
			putSelectingObject(key("hpa-2"), selector("a", "1")),
			putSelectingObject(key("hpa-3"), selector("a", "2")),
			putSelectingObject(key("hpa-4"), selector("b", "1")),
			putLabeledObject(key("pod-1"), labels("a", "1")),
			putLabeledObject(key("pod-2"), labels("a", "2")),
			putLabeledObject(key("pod-3"), labels("a", "1", "b", "1")),
			putLabeledObject(key("pod-4"), labels("a", "2", "b", "1")),
			putLabeledObject(key("pod-5"), labels("b", "1")),
			putLabeledObject(key("pod-6"), labels("b", "2")),
			deleteSelecting(key("hpa-1")),
			deleteSelecting(key("hpa-2")),
			deleteSelecting(key("hpa-3")),
			deleteSelecting(key("hpa-4")),
			deleteLabeled(key("pod-1")),
			deleteLabeled(key("pod-2")),
			deleteLabeled(key("pod-3")),
			deleteLabeled(key("pod-4")),
			deleteLabeled(key("pod-5")),
			deleteLabeled(key("pod-6")),
		},
		want: []expectation{
			emptyMap,
		},
	}, {
		name: "fuzz testing",
		ops: []operation{
			randomOperations(10000),
			deleteAll,
		},
		want: []expectation{
			emptyMap,
		},
	}}

	for _, tc := range cases {
		var permutations [][]int
		if tc.testAllPermutations {
			// Run test case with all permutations of operations.
			permutations = indexPermutations(len(tc.ops))
		} else {
			// Unless test is order dependent (e.g. includes
			// deletes) or just too big.
			var p []int
			for i := 0; i < len(tc.ops); i++ {
				p = append(p, i)
			}
			permutations = [][]int{p}
		}
		for _, permutation := range permutations {
			name := tc.name + fmt.Sprintf(" permutation %v", permutation)
			t.Run(name, func(t *testing.T) {
				multimap := NewBiMultimap()
				for i := range permutation {
					tc.ops[i](multimap)
					// Run consistency check after every operation.
					err := consistencyCheck(multimap)
					if err != nil {
						t.Fatal(err.Error())
					}
				}
				for _, expect := range tc.want {
					err := expect(multimap)
					if err != nil {
						t.Errorf("%v %v", tc.name, err)
					}
				}
			})
		}
	}
}

func TestEfficientAssociation(t *testing.T) {
	useOnceSelector := useOnce(selector("a", "1"))
	m := NewBiMultimap()
	m.PutSelector(key("hpa-1"), useOnceSelector)
	m.Put(key("pod-1"), labels("a", "1"))

	// Selector is used only during full scan. Second Put will use
	// cached association or explode.
	m.Put(key("pod-2"), labels("a", "1"))

	err := forwardSelect(key("hpa-1"), key("pod-1"), key("pod-2"))(m)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestUseOnceSelector(t *testing.T) {
	useOnceSelector := useOnce(selector("a", "1"))
	labels := pkglabels.Set(labels("a", "1"))

	// Use once.
	useOnceSelector.Matches(labels)
	// Use twice.
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic when using selector twice.")
		}
	}()
	useOnceSelector.Matches(labels)
}

func TestObjectsExist(t *testing.T) {
	m := NewBiMultimap()

	// Nothing exists in the empty map.
	assert.False(t, m.Exists(key("pod-1")))
	assert.False(t, m.SelectorExists(key("hpa-1")))

	// Adding entries.
	m.PutSelector(key("hpa-1"), useOnce(selector("a", "1")))
	m.Put(key("pod-1"), labels("a", "1"))

	// Entries exist.
	assert.True(t, m.Exists(key("pod-1")))
	assert.True(t, m.SelectorExists(key("hpa-1")))

	// Removing the entries.
	m.DeleteSelector(key("hpa-1"))
	m.Delete(key("pod-1"))

	// They don't exist anymore.
	assert.False(t, m.Exists(key("pod-1")))
	assert.False(t, m.SelectorExists(key("hpa-1")))
}

type useOnceSelector struct {
	used     bool
	selector pkglabels.Selector
}

func useOnce(s pkglabels.Selector) pkglabels.Selector {
	return &useOnceSelector{
		selector: s,
	}
}

func (u *useOnceSelector) Matches(l pkglabels.Labels) bool {
	if u.used {
		panic("useOnceSelector used more than once")
	}
	u.used = true
	return u.selector.Matches(l)
}

func (u *useOnceSelector) Empty() bool {
	return u.selector.Empty()
}

func (u *useOnceSelector) String() string {
	return u.selector.String()
}

func (u *useOnceSelector) Add(r ...pkglabels.Requirement) pkglabels.Selector {
	u.selector = u.selector.Add(r...)
	return u
}

func (u *useOnceSelector) Requirements() (pkglabels.Requirements, bool) {
	return u.selector.Requirements()
}

func (u *useOnceSelector) DeepCopySelector() pkglabels.Selector {
	u.selector = u.selector.DeepCopySelector()
	return u
}

func (u *useOnceSelector) RequiresExactMatch(label string) (value string, found bool) {
	v, f := u.selector.RequiresExactMatch(label)
	return v, f
}

func indexPermutations(size int) [][]int {
	var permute func([]int, []int) [][]int
	permute = func(placed, remaining []int) (permutations [][]int) {
		if len(remaining) == 0 {
			return [][]int{placed}
		}
		for i, v := range remaining {
			r := append([]int(nil), remaining...) // copy remaining
			r = append(r[:i], r[i+1:]...)         // delete placed index
			p := permute(append(placed, v), r)    // place index and permute
			permutations = append(permutations, p...)
		}
		return
	}
	var remaining []int
	for i := 0; i < size; i++ {
		remaining = append(remaining, i)
	}
	return permute(nil, remaining)
}

type operation func(*BiMultimap)

func putLabeledObject(key Key, labels map[string]string) operation {
	return func(m *BiMultimap) {
		m.Put(key, labels)
	}
}

func putSelectingObject(key Key, selector pkglabels.Selector) operation {
	return func(m *BiMultimap) {
		m.PutSelector(key, selector)
	}
}

func deleteLabeled(key Key) operation {
	return func(m *BiMultimap) {
		m.Delete(key)
	}
}

func deleteSelecting(key Key) operation {
	return func(m *BiMultimap) {
		m.DeleteSelector(key)
	}
}

func deleteAll(m *BiMultimap) {
	for key := range m.labeledObjects {
		m.Delete(key)
	}
	for key := range m.selectingObjects {
		m.DeleteSelector(key)
	}
}

func keepOnly(keys ...Key) operation {
	return func(m *BiMultimap) {
		m.KeepOnly(keys)
	}
}

func keepOnlySelectors(keys ...Key) operation {
	return func(m *BiMultimap) {
		m.KeepOnlySelectors(keys)
	}
}

func randomOperations(times int) operation {
	pods := []Key{
		key("pod-1"),
		key("pod-2"),
		key("pod-3"),
		key("pod-4"),
		key("pod-5"),
		key("pod-6"),
	}
	randomPod := func() Key {
		return pods[rand.Intn(len(pods))]
	}
	labels := []map[string]string{
		labels("a", "1"),
		labels("a", "2"),
		labels("b", "1"),
		labels("b", "2"),
		labels("a", "1", "b", "1"),
		labels("a", "2", "b", "2"),
		labels("a", "3"),
		labels("c", "1"),
	}
	randomLabels := func() map[string]string {
		return labels[rand.Intn(len(labels))]
	}
	hpas := []Key{
		key("hpa-1"),
		key("hpa-2"),
		key("hpa-3"),
	}
	randomHpa := func() Key {
		return hpas[rand.Intn(len(hpas))]
	}
	selectors := []pkglabels.Selector{
		selector("a", "1"),
		selector("b", "1"),
		selector("a", "1", "b", "1"),
		selector("c", "2"),
	}
	randomSelector := func() pkglabels.Selector {
		return selectors[rand.Intn(len(selectors))]
	}
	randomOp := func(m *BiMultimap) {
		switch rand.Intn(4) {
		case 0:
			m.Put(randomPod(), randomLabels())
		case 1:
			m.PutSelector(randomHpa(), randomSelector())
		case 2:
			m.Delete(randomPod())
		case 3:
			m.DeleteSelector(randomHpa())
		}
	}
	return func(m *BiMultimap) {
		for i := 0; i < times; i++ {
			randomOp(m)
		}
	}
}

type expectation func(*BiMultimap) error

func forwardSelect(key Key, want ...Key) expectation {
	return func(m *BiMultimap) error {
		got, _ := m.Select(key)
		if !unorderedEqual(want, got) {
			return fmt.Errorf("forward select %v wanted %v. got %v.", key, want, got)
		}
		return nil
	}
}

func reverseSelect(key Key, want ...Key) expectation {
	return func(m *BiMultimap) error {
		got, _ := m.ReverseSelect(key)
		if !unorderedEqual(want, got) {
			return fmt.Errorf("reverse select %v wanted %v. got %v.", key, want, got)
		}
		return nil
	}
}

func emptyMap(m *BiMultimap) error {
	if len(m.labeledObjects) != 0 {
		return fmt.Errorf("Found %v labeledObjects. Wanted none.", len(m.labeledObjects))
	}
	if len(m.selectingObjects) != 0 {
		return fmt.Errorf("Found %v selectingObjects. Wanted none.", len(m.selectingObjects))
	}
	if len(m.labeledBySelecting) != 0 {
		return fmt.Errorf("Found %v cached labeledBySelecting associations. Wanted none.", len(m.labeledBySelecting))
	}
	if len(m.selectingByLabeled) != 0 {
		return fmt.Errorf("Found %v cached selectingByLabeled associations. Wanted none.", len(m.selectingByLabeled))
	}
	return nil
}

func consistencyCheck(m *BiMultimap) error {
	emptyKey := Key{}
	emptyLabelsKey := labelsKey{}
	emptySelectorKey := selectorKey{}
	for k, v := range m.labeledObjects {
		if v == nil {
			return fmt.Errorf("Found nil labeled object for key %q", k)
		}
		if k == emptyKey {
			return fmt.Errorf("Found empty key for labeled object %+v", v)
		}
	}
	for k, v := range m.selectingObjects {
		if v == nil {
			return fmt.Errorf("Found nil selecting object for key %q", k)
		}
		if k == emptyKey {
			return fmt.Errorf("Found empty key for selecting object %+v", v)
		}
	}
	for k, v := range m.labeledBySelecting {
		if v == nil {
			return fmt.Errorf("Found nil labeledBySelecting entry for key %q", k)
		}
		if k == emptySelectorKey {
			return fmt.Errorf("Found empty key for labeledBySelecting object %+v", v)
		}
		for k2, v2 := range v.objects {
			if v2 == nil {
				return fmt.Errorf("Found nil object in labeledBySelecting under keys %q and %q", k, k2)
			}
			if k2 == emptyKey {
				return fmt.Errorf("Found empty key for object in labeledBySelecting under key %+v", k)
			}
		}
		if v.refCount < 1 {
			return fmt.Errorf("Found labeledBySelecting entry with no references (orphaned) under key %q", k)
		}
	}
	for k, v := range m.selectingByLabeled {
		if v == nil {
			return fmt.Errorf("Found nil selectingByLabeled entry for key %q", k)
		}
		if k == emptyLabelsKey {
			return fmt.Errorf("Found empty key for selectingByLabeled object %+v", v)
		}
		for k2, v2 := range v.objects {
			if v2 == nil {
				return fmt.Errorf("Found nil object in selectingByLabeled under keys %q and %q", k, k2)
			}
			if k2 == emptyKey {
				return fmt.Errorf("Found empty key for object in selectingByLabeled under key %+v", k)
			}
		}
		if v.refCount < 1 {
			return fmt.Errorf("Found selectingByLabeled entry with no references (orphaned) under key %q", k)
		}
	}
	return nil
}

func key(s string, ss ...string) Key {
	if len(ss) > 1 {
		panic("Key requires 1 or 2 parts.")
	}
	k := Key{
		Name: s,
	}
	if len(ss) >= 1 {
		k.Namespace = ss[0]
	}
	return k
}

func labels(ls ...string) map[string]string {
	if len(ls)%2 != 0 {
		panic("labels requires pairs of strings.")
	}
	ss := make(map[string]string)
	for i := 0; i < len(ls); i += 2 {
		ss[ls[i]] = ls[i+1]
	}
	return ss
}

func selector(ss ...string) pkglabels.Selector {
	if len(ss)%2 != 0 {
		panic("selector requires pairs of strings.")
	}
	s := pkglabels.NewSelector()
	for i := 0; i < len(ss); i += 2 {
		r, err := pkglabels.NewRequirement(ss[i], selection.Equals, []string{ss[i+1]})
		if err != nil {
			panic(err)
		}
		s = s.Add(*r)
	}
	return s
}

func unorderedEqual(as, bs []Key) bool {
	if len(as) != len(bs) {
		return false
	}
	aMap := make(map[Key]int)
	for _, a := range as {
		aMap[a]++
	}
	bMap := make(map[Key]int)
	for _, b := range bs {
		bMap[b]++
	}
	if len(aMap) != len(bMap) {
		return false
	}
	for a, count := range aMap {
		if bMap[a] != count {
			return false
		}
	}
	return true
}
