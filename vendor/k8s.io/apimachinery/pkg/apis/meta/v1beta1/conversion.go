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
	"unsafe"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
)

// Convert_Slice_string_To_v1beta1_IncludeObjectPolicy allows converting a URL query parameter value
func Convert_Slice_string_To_v1beta1_IncludeObjectPolicy(in *[]string, out *IncludeObjectPolicy, s conversion.Scope) error {
	if len(*in) > 0 {
		*out = IncludeObjectPolicy((*in)[0])
	}
	return nil
}

// Convert_v1beta1_PartialObjectMetadataList_To_v1_PartialObjectMetadataList allows converting PartialObjectMetadataList between versions
func Convert_v1beta1_PartialObjectMetadataList_To_v1_PartialObjectMetadataList(in *PartialObjectMetadataList, out *v1.PartialObjectMetadataList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = *(*[]v1.PartialObjectMetadata)(unsafe.Pointer(&in.Items))
	return nil
}

// Convert_v1_PartialObjectMetadataList_To_v1beta1_PartialObjectMetadataList allows converting PartialObjectMetadataList between versions
func Convert_v1_PartialObjectMetadataList_To_v1beta1_PartialObjectMetadataList(in *v1.PartialObjectMetadataList, out *PartialObjectMetadataList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = *(*[]v1.PartialObjectMetadata)(unsafe.Pointer(&in.Items))
	return nil
}
