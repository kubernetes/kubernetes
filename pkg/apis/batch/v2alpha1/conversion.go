/*
Copyright 2016 The Kubernetes Authors.

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

package v2alpha1

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_batch_JobSpec_To_v2alpha1_JobSpec,
		Convert_v2alpha1_JobSpec_To_batch_JobSpec,
	)
	if err != nil {
		return err
	}

	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	for _, kind := range []string{"Job", "JobTemplate", "CronJob"} {
		err = api.Scheme.AddFieldLabelConversionFunc("batch/v2alpha1", kind,
			func(label, value string) (string, string, error) {
				switch label {
				case "metadata.name", "metadata.namespace", "status.successful":
					return label, value, nil
				default:
					return "", "", fmt.Errorf("field label not supported: %s", label)
				}
			})
		if err != nil {
			return err
		}
	}
	return nil
}

func Convert_batch_JobSpec_To_v2alpha1_JobSpec(in *batch.JobSpec, out *JobSpec, s conversion.Scope) error {
	out.Parallelism = in.Parallelism
	out.Completions = in.Completions
	out.ActiveDeadlineSeconds = in.ActiveDeadlineSeconds
	out.Selector = in.Selector
	if in.ManualSelector != nil {
		out.ManualSelector = new(bool)
		*out.ManualSelector = *in.ManualSelector
	} else {
		out.ManualSelector = nil
	}

	if err := v1.Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func Convert_v2alpha1_JobSpec_To_batch_JobSpec(in *JobSpec, out *batch.JobSpec, s conversion.Scope) error {
	out.Parallelism = in.Parallelism
	out.Completions = in.Completions
	out.ActiveDeadlineSeconds = in.ActiveDeadlineSeconds
	out.Selector = in.Selector
	if in.ManualSelector != nil {
		out.ManualSelector = new(bool)
		*out.ManualSelector = *in.ManualSelector
	} else {
		out.ManualSelector = nil
	}

	if err := v1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}
