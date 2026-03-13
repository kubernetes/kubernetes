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

package webhook

import (
	"k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

// alwaysDeny all requests made to this function.
func alwaysDeny(ar v1.AdmissionReview) *v1.AdmissionResponse {
	klog.V(2).Info("calling always-deny")
	reviewResponse := v1.AdmissionResponse{}
	reviewResponse.Allowed = false
	reviewResponse.Result = &metav1.Status{Message: "this webhook denies all requests"}
	return &reviewResponse
}
