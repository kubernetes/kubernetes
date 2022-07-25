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

package v1beta1

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1beta1"}

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// AddMetaToScheme registers base meta types into schemas.
func AddMetaToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	return nil
}

// RegisterConversions adds conversion functions to the given scheme.
func RegisterConversions(s *runtime.Scheme) error {
	if err := s.AddGeneratedConversionFunc((*PartialObjectMetadataList)(nil), (*v1.PartialObjectMetadataList)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_v1beta1_PartialObjectMetadataList_To_v1_PartialObjectMetadataList(a.(*PartialObjectMetadataList), b.(*v1.PartialObjectMetadataList), scope)
	}); err != nil {
		return err
	}
	if err := s.AddGeneratedConversionFunc((*v1.PartialObjectMetadataList)(nil), (*PartialObjectMetadataList)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_v1_PartialObjectMetadataList_To_v1beta1_PartialObjectMetadataList(a.(*v1.PartialObjectMetadataList), b.(*PartialObjectMetadataList), scope)
	}); err != nil {
		return err
	}
	return nil
}
