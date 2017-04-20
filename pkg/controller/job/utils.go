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

package job

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
)

func IsJobFinished(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func newControllerRef(j *batch.Job) *metav1.OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return &metav1.OwnerReference{
		APIVersion:         controllerKind.GroupVersion().String(),
		Kind:               controllerKind.Kind,
		Name:               j.Name,
		UID:                j.UID,
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
}
