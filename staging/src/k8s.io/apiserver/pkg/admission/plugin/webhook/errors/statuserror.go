/*
Copyright 2017 The Kubernetes Authors.

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

package errors

import (
	"fmt"
	"net/http"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
)

// ToStatusErr returns a StatusError with information about the webhook plugin
func ToStatusErr(attr admission.Attributes, webhookName string, result *metav1.Status) *apierrors.StatusError {
	if result == nil {
		result = &metav1.Status{}
	} else {
		// deepcopy before mutating it
		result = result.DeepCopy()
	}

	gr := attr.GetResource().GroupResource()
	name := attr.GetName()
	if len(name) == 0 {
		name = "Unknown"
	}

	// sanitize status
	if result.Code == 0 {
		result.Code = http.StatusForbidden
	}
	if len(result.Status) == 0 {
		result.Status = metav1.StatusFailure
	}

	deniedBy := fmt.Sprintf("admission webhook %q denied the request", webhookName)
	switch {
	case len(result.Message) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Message)
	case len(result.Reason) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Reason)
	default:
		result.Message = fmt.Sprintf("%s without explanation", deniedBy)
	}

	if result.Details == nil {
		if result.Code == http.StatusInternalServerError {
			result.Details = &metav1.StatusDetails{
				Causes: []metav1.StatusCause{{Message: result.Message}},
			}
		} else {
			result.Details = &metav1.StatusDetails{
				Group: attr.GetResource().Group,
				Kind:  attr.GetResource().Resource, // yes, this is odd. But we replicate apierrors.NewForbidden here which does the same.
				Name:  name,
			}
		}
	}

	prefix := fmt.Sprintf("%s %q is forbidden", gr.String(), name)
	if result.Code == http.StatusInternalServerError {
		prefix = "Internal error occurred"
	}
	result.Message = fmt.Sprintf("%s: %s", prefix, result.Message)

	if len(result.Reason) == 0 {
		if result.Code == http.StatusInternalServerError {
			result.Reason = metav1.StatusReasonInternalError
		} else {
			result.Reason = metav1.StatusReasonForbidden
		}
	}

	return &apierrors.StatusError{
		ErrStatus: *result,
	}
}

// NewDryRunUnsupportedErr returns a StatusError with information about the webhook plugin
func NewDryRunUnsupportedErr(webhookName string) *apierrors.StatusError {
	reason := fmt.Sprintf("admission webhook %q does not support dry run", webhookName)
	return apierrors.NewBadRequest(reason)
}
