/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"net/http"

	"k8s.io/api/admissionregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type policyDecisionKind string

const (
	admit policyDecisionKind = "admit"
	deny  policyDecisionKind = "deny"
)

type policyDecision struct {
	kind    policyDecisionKind
	message string
	reason  metav1.StatusReason
}

type policyDecisionWithMetadata struct {
	policyDecision
	definition *v1alpha1.ValidatingAdmissionPolicy
	binding    *v1alpha1.ValidatingAdmissionPolicyBinding
}

type policyError struct {
	deniedDecisions []policyDecisionWithMetadata
}

func (p *policyError) Status() metav1.Status {
	var deniedDecision policyDecisionWithMetadata
	if len(p.deniedDecisions) == 0 {
		return metav1.Status{
			Status: metav1.StatusFailure,
			Reason: metav1.StatusReasonInternalError,
			Code:   500,
		}
	} else {
		deniedDecision = p.deniedDecisions[0]
	}
	if len(deniedDecision.reason) == 0 {
		deniedDecision.reason = metav1.StatusReasonInvalid
	}
	var bindingName, message string
	if deniedDecision.binding != nil {
		bindingName = deniedDecision.binding.Name
		message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' with binding '%s' denied request: %s", deniedDecision.definition.Name, bindingName, deniedDecision.message)
	} else {
		message = fmt.Sprintf("ValidatingAdmissionPolicy '%s' '%s' denied request: %s", deniedDecision.definition.Name, bindingName, deniedDecision.message)
	}

	return metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    reasonToCode(deniedDecision.reason),
		Reason:  deniedDecision.reason,
		Message: message,
	}
}

func reasonToCode(r metav1.StatusReason) int32 {
	switch r {
	case metav1.StatusReasonForbidden:
		return http.StatusForbidden
	case metav1.StatusReasonUnauthorized:
		return http.StatusUnauthorized
	case metav1.StatusReasonRequestEntityTooLarge:
		return http.StatusRequestEntityTooLarge
	case metav1.StatusReasonInvalid:
		return http.StatusUnprocessableEntity
	default:
		// It should not reach here since we only allow above reason to be set from API level
		return http.StatusUnprocessableEntity
	}
}

func (p *policyError) Error() string {
	return p.Status().Message
}
