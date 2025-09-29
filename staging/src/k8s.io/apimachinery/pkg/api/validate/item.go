/*
Copyright 2024 The Kubernetes Authors.

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

package validate

import (
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// MatchItemFn takes a pointer to an item and returns true if it matches the criteria.
type MatchItemFn[T any] func(*T) bool

// SliceItem finds the first item in newList that satisfies the match function,
// and if found, also looks for a matching item in oldList. If the value of the
// item is the same as the previous value, as per the equiv function, then no
// validation is performed. Otherwise, it invokes 'itemValidator' on these items.
//
// This function processes only the *first* matching item found in newList. It
// assumes that the match functions targets a unique identifier (primary key)
// and will match at most one element per list. If this assumption is violated,
// changes in list order can lead this function to have inconsistent behavior.
//
// The fldPath passed to itemValidator is indexed to the matched item's
// position in newList.
//
// This function does not validate items that were removed (present in oldList
// but not in newList).
func SliceItem[TList ~[]TItem, TItem any](
	ctx context.Context, op operation.Operation, fldPath *field.Path,
	newList, oldList TList,
	matches MatchItemFn[TItem],
	equiv MatchFunc[TItem],
	itemValidator func(ctx context.Context, op operation.Operation, fldPath *field.Path, newObj, oldObj *TItem) field.ErrorList,
) field.ErrorList {
	var matchedNew, matchedOld *TItem
	var newIndex int

	for i := range newList {
		if matches(&newList[i]) {
			matchedNew = &newList[i]
			newIndex = i
			break
		}
	}
	if matchedNew == nil {
		return nil
	}

	for i := range oldList {
		if matches(&oldList[i]) {
			matchedOld = &oldList[i]
			break
		}
	}

	if op.Type == operation.Update && matchedOld != nil && equiv(*matchedNew, *matchedOld) {
		return nil
	}

	return itemValidator(ctx, op, fldPath.Index(newIndex), matchedNew, matchedOld)
}
