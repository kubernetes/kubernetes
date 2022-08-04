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
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilpointer "k8s.io/utils/pointer"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_Job(obj *batchv1.Job) {
	// For a non-parallel job, you can leave both `.spec.completions` and
	// `.spec.parallelism` unset.  When both are unset, both are defaulted to 1.
	if obj.Spec.Completions == nil && obj.Spec.Parallelism == nil {
		obj.Spec.Completions = utilpointer.Int32Ptr(1)
		obj.Spec.Parallelism = utilpointer.Int32Ptr(1)
	}
	if obj.Spec.Parallelism == nil {
		obj.Spec.Parallelism = utilpointer.Int32Ptr(1)
	}
	if obj.Spec.BackoffLimit == nil {
		obj.Spec.BackoffLimit = utilpointer.Int32Ptr(6)
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
		obj.Spec.Suspend = utilpointer.BoolPtr(false)
	}
	if obj.Spec.PodFailurePolicy != nil {
		for _, rule := range obj.Spec.PodFailurePolicy.Rules {
			if rule.OnPodConditions != nil {
				for i, pattern := range rule.OnPodConditions {
					if pattern.Status == "" {
						rule.OnPodConditions[i].Status = corev1.ConditionTrue
					}
				}
			}
		}
	}
}

func SetDefaults_CronJob(obj *batchv1.CronJob) {
	if obj.Spec.ConcurrencyPolicy == "" {
		obj.Spec.ConcurrencyPolicy = batchv1.AllowConcurrent
	}
	if obj.Spec.Suspend == nil {
		obj.Spec.Suspend = utilpointer.BoolPtr(false)
	}
	if obj.Spec.SuccessfulJobsHistoryLimit == nil {
		obj.Spec.SuccessfulJobsHistoryLimit = utilpointer.Int32Ptr(3)
	}
	if obj.Spec.FailedJobsHistoryLimit == nil {
		obj.Spec.FailedJobsHistoryLimit = utilpointer.Int32Ptr(1)
	}
}
