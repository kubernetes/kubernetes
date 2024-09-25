/*
Copyright 2023 The Kubernetes Authors.

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

package job

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

// Constants for limits on completions and parallelism in Indexed Jobs
const (
	completionsSoftLimit                        = 100_000  // Soft limit for the number of completions in an Indexed Job
	parallelismSoftLimitForUnlimitedCompletions = 10_000   // Soft limit for parallelism when completions are unlimited
)

// WarningsForJobSpec generates warnings for the fields within a JobSpec.
// It checks the job configuration for potential issues, such as exceeding
// safe thresholds for completions and parallelism in Indexed jobs.
// Additionally, it delegates warnings for the pod template to pod.GetWarningsForPodTemplate.
func WarningsForJobSpec(ctx context.Context, path *field.Path, spec, oldSpec *batch.JobSpec) []string {
	var warnings []string  // Slice to collect warning messages

	// Check if the job uses IndexedCompletion mode
	if spec.CompletionMode != nil && *spec.CompletionMode == batch.IndexedCompletion {
		// Dereference completion and parallelism values, defaulting to 0 if not set
		completions := ptr.Deref(spec.Completions, 0)
		parallelism := ptr.Deref(spec.Parallelism, 0)

		// If completions and parallelism exceed recommended limits, generate a warning
		if completions > completionsSoftLimit && parallelism > parallelismSoftLimitForUnlimitedCompletions {
			msg := "In Indexed Jobs with a number of completions higher than 10^5 and a parallelism higher than 10^4, Kubernetes might not be able to track completedIndexes when a big number of indexes fail"
			warnings = append(warnings, fmt.Sprintf("%s: %s", path, msg))  // Append formatted warning message to the list
		}
	}

	// If an oldSpec is provided, set its pod template to compare with the new spec
	var oldPodTemplate *core.PodTemplateSpec
	if oldSpec != nil {
		oldPodTemplate = &oldSpec.Template
	}

	// Delegate warnings related to the pod template to pod.GetWarningsForPodTemplate
	warnings = append(warnings, pod.GetWarningsForPodTemplate(ctx, path.Child("template"), &spec.Template, oldPodTemplate)...)

	// Return the accumulated list of warnings
	return warnings
}
