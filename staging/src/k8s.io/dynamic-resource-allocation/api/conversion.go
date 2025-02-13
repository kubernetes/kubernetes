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

package api

import (
	"errors"
	"unique"

	v1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	localSchemeBuilder runtime.SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

func Convert_api_UniqueString_To_string(in *UniqueString, out *string, s conversion.Scope) error {
	if *in == NullUniqueString {
		*out = ""
		return nil
	}
	*out = in.String()
	return nil
}

func Convert_string_To_api_UniqueString(in *string, out *UniqueString, s conversion.Scope) error {
	if *in == "" {
		*out = NullUniqueString
		return nil
	}
	*out = UniqueString(unique.Make(*in))
	return nil
}

func Convert_v1beta1_Device_To_api_Device(in *v1beta1.Device, out *Device, s conversion.Scope) error {
	return nil
}

func Convert_api_Device_To_v1beta1_Device(out *Device, in *v1beta1.Device, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.Device not supported")
}

func Convert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(in *v1beta1.ResourceSliceSpec, out *ResourceSliceSpec, s conversion.Scope) error {
	return nil
}

func Convert_api_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(out *ResourceSliceSpec, in *v1beta1.ResourceSliceSpec, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.ResourceSliceSpec not supported")
}
