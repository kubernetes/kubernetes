// Copyright 2016 The rkt Authors
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

// TODO(sur) contribute this to the appc spec

package labelsort

import (
	"sort"

	"github.com/appc/spec/schema/types"
)

var ranks = map[types.ACIdentifier]int{
	"version": 0,
	"os":      1,
	"arch":    2,
}

// By is a functional type which compares two labels li, lj,
// returning true if li < lj, else false.
type By func(li, lj types.Label) bool

// RankedName compares the label names of li, lj lexically
// returning true if li.Name < lj.Name, else false.
// The names "version", "os", and "arch" always have lower ranks in that order,
// hence "version" < "os" < "arch" < [any other label name]
func RankedName(li, lj types.Label) bool {
	ri := rank(li.Name)
	rj := rank(lj.Name)

	if ri != rj {
		return ri < rj
	}

	return li.Name < lj.Name
}

func rank(name types.ACIdentifier) int {
	if i, ok := ranks[name]; ok {
		return i
	}

	return len(ranks) + 1
}

var _ sort.Interface = (*byLabelSorter)(nil)

type byLabelSorter struct {
	labels types.Labels
	by     By
}

func (by By) Sort(ls types.Labels) {
	s := byLabelSorter{
		labels: ls,
		by:     by,
	}

	sort.Sort(&s)
}

func (s *byLabelSorter) Len() int {
	return len(s.labels)
}

func (s *byLabelSorter) Less(i int, j int) bool {
	return s.by(s.labels[i], s.labels[j])
}

func (s *byLabelSorter) Swap(i int, j int) {
	s.labels[i], s.labels[j] = s.labels[j], s.labels[i]
}
