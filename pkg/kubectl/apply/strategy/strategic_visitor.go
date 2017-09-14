/*
Copyright 2017 The Kubernetes Authors.

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

package strategy

import "k8s.io/kubernetes/pkg/kubectl/apply"

// delegatingStrategy delegates merging fields to other visitor implementations
// based on the merge strategy preferred by the field.
type delegatingStrategy struct {
	options Options
	merge   mergeStrategy
	replace replaceStrategy
}

// createDelegatingStrategy returns a new delegatingStrategy
func createDelegatingStrategy(options Options) *delegatingStrategy {
	v := &delegatingStrategy{
		options: options,
	}
	v.replace = createReplaceStrategy(options, v)
	v.merge = createMergeStrategy(options, v)
	return v
}

// MergeList delegates visiting a list based on the field patch strategy.
// Defaults to "replace"
func (v delegatingStrategy) MergeList(diff apply.ListElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.MergeList(diff)
	case "replace":
		return v.replace.MergeList(diff)
	default:
		return v.replace.MergeList(diff)
	}
}

// MergeMap delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v delegatingStrategy) MergeMap(diff apply.MapElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.MergeMap(diff)
	case "replace":
		return v.replace.MergeMap(diff)
	default:
		return v.merge.MergeMap(diff)
	}
}

// MergeType delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v delegatingStrategy) MergeType(diff apply.TypeElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.MergeType(diff)
	case "replace":
		return v.replace.MergeType(diff)
	default:
		return v.merge.MergeType(diff)
	}
}

// MergePrimitive delegates visiting a primitive to the ReplaceVisitorSingleton.
func (v delegatingStrategy) MergePrimitive(diff apply.PrimitiveElement) (apply.Result, error) {
	// Always replace primitives
	return v.replace.MergePrimitive(diff)
}

// MergeEmpty
func (v delegatingStrategy) MergeEmpty(diff apply.EmptyElement) (apply.Result, error) {
	return v.merge.MergeEmpty(diff)
}

var _ apply.Strategy = &delegatingStrategy{}
