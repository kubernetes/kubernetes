/*
Copyright 2019 The Kubernetes Authors.

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
	v1 "k8s.io/api/admission/v1"
	"k8s.io/api/admission/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func convertAdmissionRequestToV1(r *v1beta1.AdmissionRequest) *v1.AdmissionRequest {
	return &v1.AdmissionRequest{
		Kind:               r.Kind,
		Namespace:          r.Namespace,
		Name:               r.Name,
		Object:             r.Object,
		Resource:           r.Resource,
		Operation:          v1.Operation(r.Operation),
		UID:                r.UID,
		DryRun:             r.DryRun,
		OldObject:          r.OldObject,
		Options:            r.Options,
		RequestKind:        r.RequestKind,
		RequestResource:    r.RequestResource,
		RequestSubResource: r.RequestSubResource,
		SubResource:        r.SubResource,
		UserInfo:           r.UserInfo,
	}
}

func convertAdmissionRequestToV1beta1(r *v1.AdmissionRequest) *v1beta1.AdmissionRequest {
	return &v1beta1.AdmissionRequest{
		Kind:               r.Kind,
		Namespace:          r.Namespace,
		Name:               r.Name,
		Object:             r.Object,
		Resource:           r.Resource,
		Operation:          v1beta1.Operation(r.Operation),
		UID:                r.UID,
		DryRun:             r.DryRun,
		OldObject:          r.OldObject,
		Options:            r.Options,
		RequestKind:        r.RequestKind,
		RequestResource:    r.RequestResource,
		RequestSubResource: r.RequestSubResource,
		SubResource:        r.SubResource,
		UserInfo:           r.UserInfo,
	}
}

func convertAdmissionResponseToV1(r *v1beta1.AdmissionResponse) *v1.AdmissionResponse {
	var pt *v1.PatchType
	if r.PatchType != nil {
		t := v1.PatchType(*r.PatchType)
		pt = &t
	}
	return &v1.AdmissionResponse{
		UID:              r.UID,
		Allowed:          r.Allowed,
		AuditAnnotations: r.AuditAnnotations,
		Patch:            r.Patch,
		PatchType:        pt,
		Result:           r.Result,
	}
}

func convertAdmissionResponseToV1beta1(r *v1.AdmissionResponse) *v1beta1.AdmissionResponse {
	var pt *v1beta1.PatchType
	if r.PatchType != nil {
		t := v1beta1.PatchType(*r.PatchType)
		pt = &t
	}
	return &v1beta1.AdmissionResponse{
		UID:              r.UID,
		Allowed:          r.Allowed,
		AuditAnnotations: r.AuditAnnotations,
		Patch:            r.Patch,
		PatchType:        pt,
		Result:           r.Result,
	}
}

func toV1AdmissionResponse(err error) *v1.AdmissionResponse {
	return &v1.AdmissionResponse{
		Result: &metav1.Status{
			Message: err.Error(),
		},
	}
}
