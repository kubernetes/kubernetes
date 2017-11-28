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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ToStatusErr returns a StatusError with information about the webhook plugin
func ToStatusErr(webhookName string, result *metav1.Status) *apierrors.StatusError {
	deniedBy := fmt.Sprintf("admission webhook %q denied the request", webhookName)
	const noExp = "without explanation"

	if result == nil {
		result = &metav1.Status{Status: metav1.StatusFailure}
	}

	switch {
	case len(result.Message) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Message)
	case len(result.Reason) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Reason)
	default:
		result.Message = fmt.Sprintf("%s %s", deniedBy, noExp)
	}

	return &apierrors.StatusError{
		ErrStatus: *result,
	}
}
