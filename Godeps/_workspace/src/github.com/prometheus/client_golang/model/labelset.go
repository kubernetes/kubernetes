// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"fmt"
	"sort"
	"strings"
)

// A LabelSet is a collection of LabelName and LabelValue pairs.  The LabelSet
// may be fully-qualified down to the point where it may resolve to a single
// Metric in the data store or not.  All operations that occur within the realm
// of a LabelSet can emit a vector of Metric entities to which the LabelSet may
// match.
type LabelSet map[LabelName]LabelValue

// Merge is a helper function to non-destructively merge two label sets.
func (l LabelSet) Merge(other LabelSet) LabelSet {
	result := make(LabelSet, len(l))

	for k, v := range l {
		result[k] = v
	}

	for k, v := range other {
		result[k] = v
	}

	return result
}

func (l LabelSet) String() string {
	labelStrings := make([]string, 0, len(l))
	for label, value := range l {
		labelStrings = append(labelStrings, fmt.Sprintf("%s=%q", label, value))
	}

	switch len(labelStrings) {
	case 0:
		return ""
	default:
		sort.Strings(labelStrings)
		return fmt.Sprintf("{%s}", strings.Join(labelStrings, ", "))
	}
}

// MergeFromMetric merges Metric into this LabelSet.
func (l LabelSet) MergeFromMetric(m Metric) {
	for k, v := range m {
		l[k] = v
	}
}
