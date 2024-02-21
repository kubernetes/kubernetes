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

package patch

import (
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/managedfields"
)

func ApplySMD(
	typeConverter managedfields.TypeConverter,
	originalObject runtime.Object,
	patch *unstructured.Unstructured,
) (runtime.Object, error) {
	if patch.GroupVersionKind() != originalObject.GetObjectKind().GroupVersionKind() {
		return nil, fmt.Errorf("patch and original object are not of the same gvk")
	} else if typeConverter == nil {
		return nil, fmt.Errorf("type converter must not be nil")
	}

	liveObjTyped, err := typeConverter.ObjectToTyped(originalObject)
	if err != nil {
		return nil, fmt.Errorf("failed to convert original object to typed object: %w", err)
	}

	patchObjTyped, err := typeConverter.ObjectToTyped(patch)
	if err != nil {
		return nil, fmt.Errorf("failed to convert patch object to typed object: %w", err)
	}

	newObjTyped, err := liveObjTyped.Merge(patchObjTyped)
	if err != nil {
		return nil, fmt.Errorf("failed to merge patch: %w", err)
	}

	// Our mutating admission policy sets the fields but does not track ownership.
	// Newly introduced fields in the patch won't be tracked by a field manager
	// (so if the original object is updated again but the mutating policy is
	// not active, the fields will be dropped).
	//
	// This necessarily means that changes to an object by a mutating policy
	// are only preserved if the policy was active at the time of the change.
	// (If the policy is not active, the changes may be dropped.)

	newObj, err := typeConverter.TypedToObject(newObjTyped)
	if err != nil {
		return nil, fmt.Errorf("failed to convert typed object to object: %w", err)
	}

	return newObj, nil
}
