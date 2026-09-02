/*
Copyright The Kubernetes Authors.

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

package coverage

// SelectorMode classifies the label/field selector shape of a SelectionPredicate.
type SelectorMode string

const (
	SelectorNone  SelectorMode = "SelectorNone" // Everything
	SelectorLabel SelectorMode = "SelectorLabel"
	SelectorField SelectorMode = "SelectorField"
	SelectorBoth  SelectorMode = "SelectorBoth"
)

func classifySelector(hasLabel, hasField bool) SelectorMode {
	switch {
	case hasLabel && hasField:
		return SelectorBoth
	case hasLabel:
		return SelectorLabel
	case hasField:
		return SelectorField
	default:
		return SelectorNone
	}
}
