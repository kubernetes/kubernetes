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

// replaceVisitor creates a patch to replace a remote field value with a local field value
type replaceStrategy struct {
	strategic *delegatingStrategy
	options   Options
}

func createReplaceStrategy(options Options, strategic *delegatingStrategy) replaceStrategy {
	return replaceStrategy{
		strategic,
		options,
	}
}

// MergeList returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceStrategy) MergeList(e apply.ListElement) (apply.Result, error) {
	return v.doReplace(e)
}

// MergeMap returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceStrategy) MergeMap(e apply.MapElement) (apply.Result, error) {
	return v.doReplace(e)
}

// MergeType returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceStrategy) MergeType(e apply.TypeElement) (apply.Result, error) {
	return v.doReplace(e)
}

// MergePrimitive returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceStrategy) MergePrimitive(e apply.PrimitiveElement) (apply.Result, error) {
	return v.doReplace(e)
}

// MergeEmpty
func (v replaceStrategy) MergeEmpty(e apply.EmptyElement) (apply.Result, error) {
	return apply.Result{Operation: apply.SET}, nil
}

// replace returns the local value if specified, otherwise it returns the remote value
// this works regardless of the approach
func (v replaceStrategy) doReplace(e apply.Element) (apply.Result, error) {

	if result, done := v.doAddOrDelete(e); done {
		return result, nil
	}
	if err := v.doConflictDetect(e); err != nil {
		return apply.Result{}, err
	}
	if e.HasLocal() {
		// Specified locally, set the local value
		return apply.Result{Operation: apply.SET, MergedResult: e.GetLocal()}, nil
	} else if e.HasRemote() {
		// Not specified locally, set the remote value
		return apply.Result{Operation: apply.SET, MergedResult: e.GetRemote()}, nil
	} else {
		// Only specified in the recorded, drop the field.
		return apply.Result{Operation: apply.DROP, MergedResult: e.GetRemote()}, nil
	}
}

// doAddOrDelete will check if the field should be either added or deleted.  If either is true, it will
// true the operation and true.  Otherwise it will return false.
func (v replaceStrategy) doAddOrDelete(e apply.Element) (apply.Result, bool) {
	if apply.IsAdd(e) {
		return apply.Result{Operation: apply.SET, MergedResult: e.GetLocal()}, true
	}

	// Delete the List
	if apply.IsDrop(e) {
		return apply.Result{Operation: apply.DROP}, true
	}

	return apply.Result{}, false
}

// doConflictDetect returns error if element has conflict
func (v replaceStrategy) doConflictDetect(e apply.Element) error {
	return v.strategic.doConflictDetect(e)
}

var _ apply.Strategy = &replaceStrategy{}
