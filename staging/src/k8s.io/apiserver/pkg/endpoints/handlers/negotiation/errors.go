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

package negotiation

import (
	"fmt"
	"net/http"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// errNotAcceptable indicates Accept negotiation has failed
type errNotAcceptable struct {
	accepted []string
}

func NewNotAcceptableError(accepted []string) error {
	return errNotAcceptable{accepted}
}

func (e errNotAcceptable) Error() string {
	return fmt.Sprintf("only the following media types are accepted: %v", strings.Join(e.accepted, ", "))
}

func (e errNotAcceptable) Status() metav1.Status {
	return metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusNotAcceptable,
		Reason:  metav1.StatusReasonNotAcceptable,
		Message: e.Error(),
	}
}

// errUnsupportedMediaType indicates Content-Type is not recognized
type errUnsupportedMediaType struct {
	accepted []string
}

func NewUnsupportedMediaTypeError(accepted []string) error {
	return errUnsupportedMediaType{accepted}
}

func (e errUnsupportedMediaType) Error() string {
	return fmt.Sprintf("the body of the request was in an unknown format - accepted media types include: %v", strings.Join(e.accepted, ", "))
}

func (e errUnsupportedMediaType) Status() metav1.Status {
	return metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusUnsupportedMediaType,
		Reason:  metav1.StatusReasonUnsupportedMediaType,
		Message: e.Error(),
	}
}
