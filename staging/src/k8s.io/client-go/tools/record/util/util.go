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

package util

import (
	"net/http"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
)

// ValidateEventType checks that eventtype is an expected type of event
func ValidateEventType(eventtype string) bool {
	switch eventtype {
	case v1.EventTypeNormal, v1.EventTypeWarning:
		return true
	}
	return false
}

// IsKeyNotFoundError is utility function that checks if an error is not found error
func IsKeyNotFoundError(err error) bool {
	statusErr, _ := err.(*errors.StatusError)

	if statusErr != nil && statusErr.Status().Code == http.StatusNotFound {
		return true
	}

	return false
}
