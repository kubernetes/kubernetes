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

package apply

// Strategy implements a strategy for merging recorded, local and remote values contained
// in an element and returns the merged result.
// Follows the visitor pattern
type Strategy interface {
	// MergeList is invoked by ListElements when Merge is called
	MergeList(ListElement) (Result, error)

	// MergeMap is invoked by MapElements when Merge is called
	MergeMap(MapElement) (Result, error)

	// MergeType is invoked by TypeElements when Merge is called
	MergeType(TypeElement) (Result, error)

	// MergePrimitive is invoked by PrimitiveElements when Merge is called
	MergePrimitive(PrimitiveElement) (Result, error)

	// MergeEmpty is invoked by EmptyElements when Merge is called
	MergeEmpty(EmptyElement) (Result, error)
}

// Operation records whether a field should be set or dropped
type Operation int

const (
	// ERROR is an error during merge
	ERROR Operation = iota
	// SET sets the field on an object
	SET
	// DROP drops the field from an object
	DROP
)

// Result is the result of merging fields
type Result struct {
	// Operation is the operation that should be performed for the merged field
	Operation Operation
	// MergedResult is the new merged value
	MergedResult interface{}
}

const (
	// MergeStrategy is the strategy to merge the local and remote values
	MergeStrategy = "merge"

	// RetainKeysStrategy is the strategy to merge the local and remote values, but drop any fields not defined locally
	RetainKeysStrategy = "retainKeys"

	// ReplaceStrategy is the strategy to replace the remote value with the local value
	ReplaceStrategy = "replace"
)
