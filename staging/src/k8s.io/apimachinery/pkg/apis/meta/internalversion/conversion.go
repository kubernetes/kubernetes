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

package internalversion

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
)

func Convert_v1_ListOptions_To_internalversion_ListOptions(in *v1.ListOptions, out *ListOptions, s conversion.Scope) error {
	return autoConvert_v1_ListOptions_To_internalversion_ListOptions(in, out, s)
}

func Convert_internalversion_ListOptions_To_v1_ListOptions(in *ListOptions, out *v1.ListOptions, s conversion.Scope) error {
	return autoConvert_internalversion_ListOptions_To_v1_ListOptions(in, out, s)
}
