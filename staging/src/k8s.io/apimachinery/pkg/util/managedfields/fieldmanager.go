/*
Copyright 2018 The Kubernetes Authors.

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

package managedfields

import (
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
)

// FieldManager updates the managed fields and merges applied
// configurations.
type FieldManager = internal.FieldManager

// NewDefaultFieldManager creates a new FieldManager that merges apply requests
// and update managed fields for other types of requests.
func NewDefaultFieldManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion, subresource string, resetFields map[fieldpath.APIVersion]fieldpath.Filter) (*FieldManager, error) {
	f, err := internal.NewStructuredMergeManager(typeConverter, objectConverter, objectDefaulter, kind.GroupVersion(), hub, resetFields)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return internal.NewDefaultFieldManager(f, typeConverter, objectConverter, objectCreater, kind, subresource), nil
}

// NewDefaultCRDFieldManager creates a new FieldManager specifically for
// CRDs. This allows for the possibility of fields which are not defined
// in models, as well as having no models defined at all.
func NewDefaultCRDFieldManager(typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, objectDefaulter runtime.ObjectDefaulter, objectCreater runtime.ObjectCreater, kind schema.GroupVersionKind, hub schema.GroupVersion, subresource string, resetFields map[fieldpath.APIVersion]fieldpath.Filter) (_ *FieldManager, err error) {
	f, err := internal.NewCRDStructuredMergeManager(typeConverter, objectConverter, objectDefaulter, kind.GroupVersion(), hub, resetFields)
	if err != nil {
		return nil, fmt.Errorf("failed to create field manager: %v", err)
	}
	return internal.NewDefaultFieldManager(f, typeConverter, objectConverter, objectCreater, kind, subresource), nil
}

func ValidateManagedFields(encodedManagedFields []metav1.ManagedFieldsEntry) error {
	_, err := internal.DecodeManagedFields(encodedManagedFields)
	return err
}
