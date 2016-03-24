/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func addConversionFuncs(scheme *runtime.Scheme) {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_api_PodSpec_To_v1_PodSpec,
		Convert_v1_PodSpec_To_api_PodSpec,
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}

	err = api.Scheme.AddFieldLabelConversionFunc("batch/v1", "Job",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "metadata.namespace", "status.successful":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}

// The following two PodSpec conversions functions where copied from pkg/api/conversion.go
// for the generated functions to work properly.
// This should be fixed: https://github.com/kubernetes/kubernetes/issues/12977
func Convert_api_PodSpec_To_v1_PodSpec(in *api.PodSpec, out *v1.PodSpec, s conversion.Scope) error {
	return v1.Convert_api_PodSpec_To_v1_PodSpec(in, out, s)
}

func Convert_v1_PodSpec_To_api_PodSpec(in *v1.PodSpec, out *api.PodSpec, s conversion.Scope) error {
	return v1.Convert_v1_PodSpec_To_api_PodSpec(in, out, s)
}
