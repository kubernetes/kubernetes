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

package stackdriver

// Labels represents a set of Stackdriver Monitoring labels.
type Labels struct {
	m map[string]labelValue
}

type labelValue struct {
	val, desc string
}

// Set stores a label with the given key, value and description,
// overwriting any previous values with the given key.
func (labels *Labels) Set(key, value, description string) {
	if labels.m == nil {
		labels.m = make(map[string]labelValue)
	}
	labels.m[key] = labelValue{value, description}
}
