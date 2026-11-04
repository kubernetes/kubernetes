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

package v1alpha1

import (
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/internal/storagecompression"
)

// The +k8s:customValidation tags in types_compression.go call these. The checks
// themselves live in storagecompression, so a later version can reuse them.

// ValidateCustom_StorageCompressionConfiguration rejects selectors that an
// earlier entry repeats or covers.
func ValidateCustom_StorageCompressionConfiguration(
	ctx context.Context, op operation.Operation, fldPath *field.Path,
	obj, oldObj *StorageCompressionConfiguration,
) field.ErrorList {
	entries := make([][]string, 0, len(obj.CompressionResources))
	for _, entry := range obj.CompressionResources {
		entries = append(entries, entry.Resources)
	}
	return storagecompression.ValidateMasking(entries, fldPath.Child("compressionResources"))
}

// ValidateCustom_CompressionResourceConfiguration_Resources checks one entry's
// selectors: their names, and repeats or overlaps within the entry.
func ValidateCustom_CompressionResourceConfiguration_Resources(
	ctx context.Context, op operation.Operation, fldPath *field.Path,
	obj, oldObj []string,
) field.ErrorList {
	return storagecompression.ValidateSelectors(obj, fldPath)
}
