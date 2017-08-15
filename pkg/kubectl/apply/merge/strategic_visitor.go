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

import "k8s.io/kubernetes/pkg/kubectl/apply"

// strategicVisitor delegates merging fields to other visitor implementations
// based on the merge strategy preferred by the field.
type strategicVisitor struct {
	options Options
	merge   mergeVisitor
	replace replaceVisitor
}

// createStrategicVisitor returns a new strategicVisitor
func createStrategicVisitor(options Options) *strategicVisitor {
	v := &strategicVisitor{
		options: options,
	}
	v.replace = createReplaceVisitor(options, v)
	v.merge = createMergeVisitor(options, v)
	return v
}

// VisitList delegates visiting a list based on the field patch strategy.
// Defaults to "replace"
func (v strategicVisitor) VisitList(diff apply.ListElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.VisitList(diff)
	case "replace":
		return v.replace.VisitList(diff)
	default:
		return v.replace.VisitList(diff)
	}
}

// VisitMap delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v strategicVisitor) VisitMap(diff apply.MapElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.VisitMap(diff)
	case "replace":
		return v.replace.VisitMap(diff)
	default:
		return v.merge.VisitMap(diff)
	}
}

// VisitType delegates visiting a map based on the field patch strategy.
// Defaults to "merge"
func (v strategicVisitor) VisitType(diff apply.TypeElement) (apply.Result, error) {
	// TODO: Support retainkeys
	switch diff.GetFieldMergeType() {
	case "merge":
		return v.merge.VisitType(diff)
	case "replace":
		return v.replace.VisitType(diff)
	default:
		return v.merge.VisitType(diff)
	}
}

// VisitPrimitive delegates visiting a primitive to the ReplaceVisitorSingleton.
func (v strategicVisitor) VisitPrimitive(diff apply.PrimitiveElement) (apply.Result, error) {
	// Always replace primitives
	return v.replace.VisitPrimitive(diff)
}

// VisitEmpty
func (v strategicVisitor) VisitEmpty(diff apply.EmptyElement) (apply.Result, error) {
	return v.merge.VisitEmpty(diff)
}

var _ apply.Visitor = &strategicVisitor{}
