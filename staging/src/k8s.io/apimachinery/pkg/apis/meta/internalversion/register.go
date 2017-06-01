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

package internalversion

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// Scheme is the registry for any type that adheres to the meta API spec.
var scheme = runtime.NewScheme()

// Codecs provides access to encoding and decoding for the scheme.
var Codecs = serializer.NewCodecFactory(scheme)

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(scheme)

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// addToGroupVersion registers common meta types into schemas.
func addToGroupVersion(scheme *runtime.Scheme, groupVersion schema.GroupVersion) error {
	if err := scheme.AddIgnoredConversionType(&metav1.TypeMeta{}, &metav1.TypeMeta{}); err != nil {
		return err
	}
	scheme.AddConversionFuncs(
		metav1.Convert_string_To_labels_Selector,
		metav1.Convert_labels_Selector_To_string,

		metav1.Convert_string_To_fields_Selector,
		metav1.Convert_fields_Selector_To_string,

		Convert_map_to_v1_LabelSelector,
		Convert_v1_LabelSelector_to_map,

		Convert_internalversion_ListOptions_To_v1_ListOptions,
		Convert_v1_ListOptions_To_internalversion_ListOptions,
	)
	// ListOptions is the only options struct which needs conversion (it exposes labels and fields
	// as selectors for convenience). The other types have only a single representation today.
	scheme.AddKnownTypes(SchemeGroupVersion,
		&ListOptions{},
		&metav1.GetOptions{},
		&metav1.ExportOptions{},
		&metav1.DeleteOptions{},
	)
	// Allow delete options to be decoded across all version in this scheme (we may want to be more clever than this)
	scheme.AddUnversionedTypes(SchemeGroupVersion, &metav1.DeleteOptions{})
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	return nil
}

// Unlike other API groups, meta internal knows about all meta external versions, but keeps
// the logic for conversion private.
func init() {
	if err := addToGroupVersion(scheme, SchemeGroupVersion); err != nil {
		panic(err)
	}
}
