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

package v1

import (
	"math"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_Job(obj *batchv1.Job) {
	// For a non-parallel job, you can leave both `.spec.completions` and
	// `.spec.parallelism` unset.  When both are unset, both are defaulted to 1.
	if obj.Spec.Completions == nil && obj.Spec.Parallelism == nil {
		obj.Spec.Completions = ptr.To[int32](1)
		obj.Spec.Parallelism = ptr.To[int32](1)
	}
	if obj.Spec.Parallelism == nil {
		obj.Spec.Parallelism = ptr.To[int32](1)
	}
	if obj.Spec.BackoffLimit == nil {
		if obj.Spec.BackoffLimitPerIndex != nil {
			obj.Spec.BackoffLimit = ptr.To[int32](math.MaxInt32)
		} else {
			obj.Spec.BackoffLimit = ptr.To[int32](6)
		}
	}
	labels := obj.Spec.Template.Labels
	if labels != nil && len(obj.Labels) == 0 {
		obj.Labels = labels
	}
	if obj.Spec.CompletionMode == nil {
		mode := batchv1.NonIndexedCompletion
		obj.Spec.CompletionMode = &mode
	}
	if obj.Spec.Suspend == nil {
		obj.Spec.Suspend = ptr.To(false)
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.JobPodReplacementPolicy) {
		if obj.Spec.PodReplacementPolicy == nil {
			if obj.Spec.PodFailurePolicy != nil {
				obj.Spec.PodReplacementPolicy = ptr.To(batchv1.Failed)
			} else {
				obj.Spec.PodReplacementPolicy = ptr.To(batchv1.TerminatingOrFailed)
			}
		}
	}
	if obj.Spec.ManualSelector == nil {
		obj.Spec.ManualSelector = ptr.To(false)
	}
}

func SetDefaults_CronJob(obj *batchv1.CronJob) {
	if obj.Spec.ConcurrencyPolicy == "" {
		obj.Spec.ConcurrencyPolicy = batchv1.AllowConcurrent
	}
	if obj.Spec.Suspend == nil {
		obj.Spec.Suspend = ptr.To(false)
	}
	if obj.Spec.SuccessfulJobsHistoryLimit == nil {
		obj.Spec.SuccessfulJobsHistoryLimit = ptr.To[int32](3)
	}
	if obj.Spec.FailedJobsHistoryLimit == nil {
		obj.Spec.FailedJobsHistoryLimit = ptr.To[int32](1)
	}
}

func SetDefaults_PodFailurePolicyOnPodConditionsPattern(obj *batchv1.PodFailurePolicyOnPodConditionsPattern) {
	if obj.Status == "" {
		obj.Status = corev1.ConditionTrue
	}
}
