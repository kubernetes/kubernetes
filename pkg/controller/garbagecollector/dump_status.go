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

package garbagecollector

import (
	"encoding/json"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

type ErrorType string

const (
	SyncError            ErrorType = "Sync"
	AttemptToDeleteError ErrorType = "AttemptToDelete"
	AttemptToOrphanError ErrorType = "AttemptToOrphan"
)

// errorResult represents an error and its metadata that are to be marshalled into JSON.
type errorResult struct {
	ErrorTime      metav1.Time      `json:"errorTime"`
	Error          string           `json:"error"`
	ErrorType      ErrorType        `json:"errorType"`
	InvolvedObject *objectReference `json:"involvedObject,omitempty"`
}

// statusResult represents a status that is to be marshalled into JSON.
type statusResult struct {
	Errors []errorResult `json:"errors"`
}

// newErrorResult creates a new errorResult.
func newErrorResult(errorType ErrorType, record *errorRecord, identity *objectReference) errorResult {
	result := errorResult{
		InvolvedObject: identity,
		ErrorType:      errorType,
	}

	if record != nil {
		result.ErrorTime = metav1.NewTime(record.ErrorTime)
		if record.Error != nil {
			result.Error = record.Error.Error()
		}
	}
	return result
}

// serveHTTPStatus responds to an HTTP request on /status endpoint.
func (h *debugHTTPHandler) serveHTTPStatus(w http.ResponseWriter, req *http.Request) {
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.GCDebugStatus) {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	errors := h.controller.errorRecorder.DumpErrors()
	if errors == nil {
		errors = make([]errorResult, 0)
	}

	status := statusResult{
		Errors: errors,
	}

	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Write(data)
	w.WriteHeader(http.StatusOK)
}
