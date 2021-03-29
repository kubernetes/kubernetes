/*
Copyright 2015 The Kubernetes Authors.

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

	v1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/batch"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Job"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "metadata.namespace", "status.successful":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label %q not supported for Job", label)
			}
		})
}

// The following functions don't do anything special, but they need to be added
// here due to the dependency of v1beta1 on v1.

func Convert_batch_JobSpec_To_v1_JobSpec(in *batch.JobSpec, out *v1.JobSpec, s conversion.Scope) error {
	return autoConvert_batch_JobSpec_To_v1_JobSpec(in, out, s)
}

func Convert_v1_JobSpec_To_batch_JobSpec(in *v1.JobSpec, out *batch.JobSpec, s conversion.Scope) error {
	return autoConvert_v1_JobSpec_To_batch_JobSpec(in, out, s)
}
