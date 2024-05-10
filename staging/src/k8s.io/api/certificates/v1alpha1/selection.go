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

package v1alpha1

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

func init() {
	SchemeBuilder.Register(addSelectorFuncs)
}

// addSelectorFuncs adds versioned selector funcs for resources to the scheme.
func addSelectorFuncs(scheme *runtime.Scheme) error {
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("ClusterTrustBundle"), ClusterTrustBundleSelectorFunc)
	return nil
}

// ClusterTrustBundleSelectorFunc returns true if the object matches the label and field selectors.
func ClusterTrustBundleSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return ClusterTrustBundleMatcher(selector).Matches(obj)
}

// ClusterTrustBundleMatcher returns a generic matcher for a given label and field selector.
func ClusterTrustBundleMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  ClusterTrustBundleGetAttrs,
	}
}

// ClusterTrustBundleGetAttrs returns labels and fields of a given object for filtering purposes.
func ClusterTrustBundleGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	ctr, ok := obj.(*ClusterTrustBundle)
	if !ok {
		return nil, nil, fmt.Errorf("%T is not a ClusterTrustBundle", obj)
	}
	return labels.Set(ctr.ObjectMeta.Labels), ClusterTrustBundleToSelectableFields(ctr), nil
}

// ClusterTrustBundleToSelectableFields returns a field set that represents the object.
func ClusterTrustBundleToSelectableFields(ctr *ClusterTrustBundle) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(ctr)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	specificFieldsSet["spec.signerName"] = ctr.Spec.SignerName
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}
