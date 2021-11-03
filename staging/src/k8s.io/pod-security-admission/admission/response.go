/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"fmt"

	admissionv1 "k8s.io/api/admission/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

var (
	sharedAllowedResponse                        = allowedResponse()
	sharedAllowedPrivilegedResponse              = allowedResponse()
	sharedAllowedByUserExemptionResponse         = allowedResponse()
	sharedAllowedByNamespaceExemptionResponse    = allowedResponse()
	sharedAllowedByRuntimeClassExemptionResponse = allowedResponse()
)

func init() {
	sharedAllowedPrivilegedResponse.AuditAnnotations = map[string]string{
		api.EnforcedPolicyAnnotationKey: api.LevelVersion{Level: api.LevelPrivileged, Version: api.LatestVersion()}.String(),
	}
	sharedAllowedByUserExemptionResponse.AuditAnnotations = map[string]string{api.ExemptionReasonAnnotationKey: "user"}
	sharedAllowedByNamespaceExemptionResponse.AuditAnnotations = map[string]string{api.ExemptionReasonAnnotationKey: "namespace"}
	sharedAllowedByRuntimeClassExemptionResponse.AuditAnnotations = map[string]string{api.ExemptionReasonAnnotationKey: "runtimeClass"}
}

// allowedResponse is the response used when the admission decision is allow.
func allowedResponse() *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{Allowed: true}
}

// forbiddenResponse is the response used when the admission decision is deny for policy violations.
func forbiddenResponse(attrs api.Attributes, err error) *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: false,
		Result:  &apierrors.NewForbidden(attrs.GetResource().GroupResource(), attrs.GetName(), err).ErrStatus,
	}
}

// invalidResponse is the response used for namespace requests when namespace labels are invalid.
func invalidResponse(attrs api.Attributes, fieldErrors field.ErrorList) *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: false,
		Result:  &apierrors.NewInvalid(attrs.GetKind().GroupKind(), attrs.GetName(), fieldErrors).ErrStatus,
	}
}

// errorResponse is the response used to capture generic errors.
func errorResponse(err error, status *metav1.Status) *admissionv1.AdmissionResponse {
	var errDetail string
	if err != nil {
		errDetail = fmt.Sprintf("%s: %v", status.Message, err)
	} else {
		errDetail = status.Message
	}
	return &admissionv1.AdmissionResponse{
		Allowed:          false,
		Result:           status,
		AuditAnnotations: map[string]string{"error": errDetail},
	}
}
