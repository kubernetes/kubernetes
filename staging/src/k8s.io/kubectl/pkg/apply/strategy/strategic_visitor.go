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

import (
	"k8s.io/kubectl/pkg/apply"
)

// delegatingStrategy delegates merging fields to other visitor implementations
// based on the merge strategy preferred by the field.
type delegatingStrategy struct {
	options    Options
	merge      mergeStrategy
	replace    replaceStrategy
	retainKeys retainKeysStrategy
}

// createDelegatingStrategy returns a new delegatingStrategy
func createDelegatingStrategy(options Options) *delegatingStrategy {
	v := &delegatingStrategy{
		options: options,
	}
	v.replace = createReplaceStrategy(options, v)
	v.merge = createMergeStrategy(options, v)
	v.retainKeys = createRetainKeysStrategy(options, v)
	return v
}

// MergeList delegates visiting a list based on the field patch strategy.
// Defaults to "replace"
func (v delegatingStrategy) MergeList(diff apply.ListElement) (apply.Result, error) {
	switch diff.GetFieldMergeType() {
	case apply.MergeStrategy:
		return v.merge.MergeList(diff)
	case apply.ReplaceStrategy:
		return v.replace.MergeList(diff)
	case apply.RetainKeysStrategy:
		return v.retainKeys.MergeList(diff)
	default:
		return v.replace.MergeList(diff)
	}
}

// MergeMap delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v delegatingStrategy) MergeMap(diff apply.MapElement) (apply.Result, error) {
	switch diff.GetFieldMergeType() {
	case apply.MergeStrategy:
		return v.merge.MergeMap(diff)
	case apply.ReplaceStrategy:
		return v.replace.MergeMap(diff)
	case apply.RetainKeysStrategy:
		return v.retainKeys.MergeMap(diff)
	default:
		return v.merge.MergeMap(diff)
	}
}

// MergeType delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v delegatingStrategy) MergeType(diff apply.TypeElement) (apply.Result, error) {
	switch diff.GetFieldMergeType() {
	case apply.MergeStrategy:
		return v.merge.MergeType(diff)
	case apply.ReplaceStrategy:
		return v.replace.MergeType(diff)
	case apply.RetainKeysStrategy:
		return v.retainKeys.MergeType(diff)
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

// doConflictDetect detects conflicts in element when option enabled, return error if conflict happened.
func (v delegatingStrategy) doConflictDetect(e apply.Element) error {
	if v.options.FailOnConflict {
		if e, ok := e.(apply.ConflictDetector); ok {
			return e.HasConflict()
		}
	}
	return nil
}

var _ apply.Strategy = &delegatingStrategy{}
