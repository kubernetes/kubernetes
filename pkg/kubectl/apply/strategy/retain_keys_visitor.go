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
	"fmt"
	"k8s.io/kubernetes/pkg/kubectl/apply"
)

func createRetainKeysStrategy(options Options, strategic *delegatingStrategy) retainKeysStrategy {
	return retainKeysStrategy{
		&mergeStrategy{strategic, options},
		strategic,
		options,
	}
}

// retainKeysStrategy merges the values in an Element into a single Result,
// dropping any fields omitted from the local copy. (but merging values when
// defined locally and remotely)
type retainKeysStrategy struct {
	merge     *mergeStrategy
	strategic *delegatingStrategy
	options   Options
}

// MergeMap merges the type instances in a TypeElement into a single Result
// keeping only the fields defined locally, but merging their values with
// the remote values.
func (v retainKeysStrategy) MergeType(e apply.TypeElement) (apply.Result, error) {
	// No merge logic if adding or deleting a field
	if result, done := v.merge.doAddOrDelete(&e); done {
		return result, nil
	}

	elem := map[string]apply.Element{}
	for key := range e.GetLocalMap() {
		elem[key] = e.GetValues()[key]
	}
	return v.merge.doMergeMap(elem)
}

// MergeMap returns an error.  Only TypeElements can have retainKeys.
func (v retainKeysStrategy) MergeMap(e apply.MapElement) (apply.Result, error) {
	return apply.Result{}, fmt.Errorf("Cannot use retainkeys with map element %v", e.Name)
}

// MergeList returns an error.  Only TypeElements can have retainKeys.
func (v retainKeysStrategy) MergeList(e apply.ListElement) (apply.Result, error) {
	return apply.Result{}, fmt.Errorf("Cannot use retainkeys with list element %v", e.Name)
}

// MergePrimitive returns an error.  Only TypeElements can have retainKeys.
func (v retainKeysStrategy) MergePrimitive(diff apply.PrimitiveElement) (apply.Result, error) {
	return apply.Result{}, fmt.Errorf("Cannot use retainkeys with primitive element %v", diff.Name)
}

// MergeEmpty returns an empty result
func (v retainKeysStrategy) MergeEmpty(diff apply.EmptyElement) (apply.Result, error) {
	return v.merge.MergeEmpty(diff)
}

var _ apply.Strategy = &retainKeysStrategy{}
