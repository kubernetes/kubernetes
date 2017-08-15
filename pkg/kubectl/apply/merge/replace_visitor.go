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

package merge

import (
	"k8s.io/kubernetes/pkg/kubectl/apply"
)

// replaceVisitor creates a patch to replace a remote field value with a local field value
type replaceVisitor struct {
	strategic *strategicVisitor
	options   Options
}

func createReplaceVisitor(options Options, strategic *strategicVisitor) replaceVisitor {
	return replaceVisitor{
		strategic,
		options,
	}
}

// VisitList returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceVisitor) VisitList(e apply.ListElement) (apply.Result, error) {
	return v.doReplace(e)
}

// VisitMap returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceVisitor) VisitMap(e apply.MapElement) (apply.Result, error) {
	return v.doReplace(e)
}

// VisitType returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceVisitor) VisitType(e apply.TypeElement) (apply.Result, error) {
	return v.doReplace(e)
}

// VisitPrimitive returns a result by merging the recorded, local and remote values
// - replacing the remote value with the local value
func (v replaceVisitor) VisitPrimitive(e apply.PrimitiveElement) (apply.Result, error) {
	return v.doReplace(e)
}

// VisitEmpty
func (v replaceVisitor) VisitEmpty(e apply.EmptyElement) (apply.Result, error) {
	return apply.Result{Operation: apply.SET}, nil
}

// replace returns the local value if specified, otherwise it returns the remote value
// this works regardless of the approach
func (v replaceVisitor) doReplace(e apply.Element) (apply.Result, error) {
	// TODO: Check for conflicts
	if result, done := v.doAddOrDelete(e); done {
		return result, nil
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
func (v replaceVisitor) doAddOrDelete(e apply.Element) (apply.Result, bool) {
	if apply.IsAdd(e) {
		return apply.Result{Operation: apply.SET, MergedResult: e.GetLocal()}, true
	}

	// Delete the List
	if apply.IsDrop(e) {
		return apply.Result{Operation: apply.DROP}, true
	}

	return apply.Result{}, false
}

var _ apply.Visitor = &replaceVisitor{}
