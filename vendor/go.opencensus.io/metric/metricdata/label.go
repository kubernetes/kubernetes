// Copyright 2018, OpenCensus Authors
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

package metricdata

// LabelKey represents key of a label. It has optional
// description attribute.
type LabelKey struct {
	Key         string
	Description string
}

// LabelValue represents the value of a label.
// The zero value represents a missing label value, which may be treated
// differently to an empty string value by some back ends.
type LabelValue struct {
	Value   string // string value of the label
	Present bool   // flag that indicated whether a value is present or not
}

// NewLabelValue creates a new non-nil LabelValue that represents the given string.
func NewLabelValue(val string) LabelValue {
	return LabelValue{Value: val, Present: true}
}
