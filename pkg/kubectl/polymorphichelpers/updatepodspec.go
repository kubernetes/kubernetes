/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func updatePodSpecForObject(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error) {
	switch t := obj.(type) {
	case *v1.Pod:
		return true, fn(&t.Spec)
		// ReplicationController
	case *v1.ReplicationController:
		if t.Spec.Template == nil {
			t.Spec.Template = &v1.PodTemplateSpec{}
		}
		return true, fn(&t.Spec.Template.Spec)

		// Deployment
	case *extensionsv1beta1.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta1.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.Deployment:
		return true, fn(&t.Spec.Template.Spec)

		// DaemonSet
	case *extensionsv1beta1.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)

		// ReplicaSet
	case *extensionsv1beta1.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)

		// StatefulSet
	case *appsv1beta1.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1beta2.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)
	case *appsv1.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)

		// Job
	case *batchv1.Job:
		return true, fn(&t.Spec.Template.Spec)

		// CronJob
	case *batchv1beta1.CronJob:
		return true, fn(&t.Spec.JobTemplate.Spec.Template.Spec)
	case *batchv2alpha1.CronJob:
		return true, fn(&t.Spec.JobTemplate.Spec.Template.Spec)

	default:
		return false, fmt.Errorf("the object is not a pod or does not have a pod template: %T", t)
	}
}
