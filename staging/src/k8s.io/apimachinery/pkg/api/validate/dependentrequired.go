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

package validate

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// DependentRequired verifies that when triggerIsSet(obj) is true, dependentIsSet(obj)
// is also true; otherwise reports an error at fldPath.Child(dependentName).
// On Update, the check is skipped if neither side's set-ness changed from oldObj,
// so unrelated updates can proceed past a pre-existing violation.
func DependentRequired[T any](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *T,
	triggerName string, triggerIsSet ExtractorFn[*T, bool],
	dependentName string, dependentIsSet ExtractorFn[*T, bool],
) field.ErrorList {
	if obj == nil {
		return nil
	}
	if op.Type == operation.Update && oldObj != nil {
		if triggerIsSet(obj) == triggerIsSet(oldObj) && dependentIsSet(obj) == dependentIsSet(oldObj) {
			return nil
		}
	}
	if !triggerIsSet(obj) {
		return nil
	}
	if dependentIsSet(obj) {
		return nil
	}
	return field.ErrorList{
		field.Required(fldPath.Child(dependentName),
			fmt.Sprintf("must be set when %s is set", triggerName)).
			WithOrigin("dependentRequired"),
	}
}
