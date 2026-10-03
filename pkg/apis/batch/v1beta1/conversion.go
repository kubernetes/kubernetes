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
	"fmt"

	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/batch"
)

// Convert_batch_CronJobStatus_To_v1beta1_CronJobStatus converts the internal
// CronJobStatus to the v1beta1 version. The nextScheduleTime field is only
// present in the internal and batch/v1 types (it was added alongside the
// CronJobsNextScheduleTime feature); it does not exist in the deprecated
// batch/v1beta1 CronJob and is intentionally dropped here.
func Convert_batch_CronJobStatus_To_v1beta1_CronJobStatus(in *batch.CronJobStatus, out *batchv1beta1.CronJobStatus, s conversion.Scope) error {
	return autoConvert_batch_CronJobStatus_To_v1beta1_CronJobStatus(in, out, s)
}

func addConversionFuncs(scheme *runtime.Scheme) error {
	var err error
	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	for _, kind := range []string{"Job", "JobTemplate", "CronJob"} { // don't close over range variables
		err = scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind(kind),
			func(label, value string) (string, string, error) {
				switch label {
				case "metadata.name", "metadata.namespace", "status.successful":
					return label, value, nil
				default:
					return "", "", fmt.Errorf("field label %q not supported for %q", label, kind)
				}
			})
		if err != nil {
			return err
		}
	}
	return nil
}
