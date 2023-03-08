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

package validatingadmissionpolicy

import (
	"net/http"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type PolicyDecisionAction string

const (
	ActionAdmit PolicyDecisionAction = "admit"
	ActionDeny  PolicyDecisionAction = "deny"
)

type PolicyDecisionEvaluation string

const (
	EvalAdmit PolicyDecisionEvaluation = "admit"
	EvalError PolicyDecisionEvaluation = "error"
	EvalDeny  PolicyDecisionEvaluation = "deny"
)

// PolicyDecision contains the action determined from a cel evaluation along with metadata such as message, reason and duration
type PolicyDecision struct {
	Action     PolicyDecisionAction
	Evaluation PolicyDecisionEvaluation
	Message    string
	Reason     metav1.StatusReason
	Elapsed    time.Duration
}

type PolicyAuditAnnotationAction string

const (
	// AuditAnnotationActionPublish indicates that the audit annotation should be
	// published with the audit event.
	AuditAnnotationActionPublish PolicyAuditAnnotationAction = "publish"
	// AuditAnnotationActionError indicates that the valueExpression resulted
	// in an error.
	AuditAnnotationActionError PolicyAuditAnnotationAction = "error"
	// AuditAnnotationActionExclude indicates that the audit annotation should be excluded
	// because the valueExpression evaluated to null, or because FailurePolicy is Ignore
	// and the expression failed with a parse error, type check error, or runtime error.
	AuditAnnotationActionExclude PolicyAuditAnnotationAction = "exclude"
)

type PolicyAuditAnnotation struct {
	Key     string
	Value   string
	Elapsed time.Duration
	Action  PolicyAuditAnnotationAction
	Error   string
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
