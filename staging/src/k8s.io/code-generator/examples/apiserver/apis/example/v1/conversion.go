/*
Copyright 2025 The Kubernetes Authors.

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

package v1

import (
	conversion "k8s.io/apimachinery/pkg/conversion"
	example "k8s.io/code-generator/examples/apiserver/apis/example"
)

// manually created final conversion function required because the object contains private fields that cannot be auto-converted
func Convert_v1_ConversionPrivate_To_example_ConversionPrivate(in *ConversionPrivate, out *example.ConversionPrivate, scope conversion.Scope) error {
	return autoConvert_v1_ConversionPrivate_To_example_ConversionPrivate(in, out, scope)
}

// manually created final conversion function required because the object contains private fields that cannot be auto-converted
func Convert_example_ConversionPrivate_To_v1_ConversionPrivate(in *example.ConversionPrivate, out *ConversionPrivate, scope conversion.Scope) error {
	return autoConvert_example_ConversionPrivate_To_v1_ConversionPrivate(in, out, scope)
}

// custom conversion function to exercise use of custom functions in slice/map/pointer fields
func Convert_v1_ConversionCustom_To_example_ConversionCustom(in *ConversionCustom, out *example.ConversionCustom, scope conversion.Scope) error {
	return autoConvert_v1_ConversionCustom_To_example_ConversionCustom(in, out, scope)
}

// custom conversion function to exercise use of custom functions in slice/map/pointer fields
func Convert_example_ConversionCustom_To_v1_ConversionCustom(in *example.ConversionCustom, out *ConversionCustom, scope conversion.Scope) error {
	return autoConvert_example_ConversionCustom_To_v1_ConversionCustom(in, out, scope)
}
