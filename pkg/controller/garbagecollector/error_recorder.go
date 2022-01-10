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
	"sync"
	"time"
)

// errorRecord represents an error with additional metadata
type errorRecord struct {
	ErrorTime time.Time
	Error     error
}

func NewErrorRecord(err error) *errorRecord {
	return &errorRecord{
		ErrorTime: time.Now(),
		Error:     err,
	}
}

type errorRecorder interface {
	SetAttemptToDeleteError(node *node, err error)
	SetAttemptToOrphanError(node *node, err error)
	SetSyncError(err error)
	ClearAttemptToDeleteError(node *node)
	ClearAttemptToOrphanError(node *node)
	ClearSyncError()
	DumpErrors() []errorResult
}

var _ errorRecorder = &errRecorder{}

// errRecorder stores various types of errors for later debugging. Its methods are thread-safe.
type errRecorder struct {
	attemptToDeleteErrorsLock sync.RWMutex
	attemptToOrphanErrorsLock sync.RWMutex
	syncErrorLock             sync.RWMutex
	attemptToDeleteErrors     map[*node]*errorRecord
	attemptToOrphanErrors     map[*node]*errorRecord
	syncError                 *errorRecord
}

// newErrorRecorder creates a new errorRecorder.
func newErrorRecorder() errorRecorder {
	return &errRecorder{
		attemptToDeleteErrors: make(map[*node]*errorRecord),
		attemptToOrphanErrors: make(map[*node]*errorRecord),
	}
}

func (r *errRecorder) SetAttemptToDeleteError(node *node, err error) {
	r.attemptToDeleteErrorsLock.Lock()
	defer r.attemptToDeleteErrorsLock.Unlock()
	r.attemptToDeleteErrors[node] = NewErrorRecord(err)
}

func (r *errRecorder) SetAttemptToOrphanError(node *node, err error) {
	r.attemptToOrphanErrorsLock.Lock()
	defer r.attemptToOrphanErrorsLock.Unlock()
	r.attemptToOrphanErrors[node] = NewErrorRecord(err)
}

func (r *errRecorder) SetSyncError(err error) {
	r.syncErrorLock.Lock()
	defer r.syncErrorLock.Unlock()
	r.syncError = NewErrorRecord(err)
}

func (r *errRecorder) ClearAttemptToDeleteError(node *node) {
	r.attemptToDeleteErrorsLock.Lock()
	defer r.attemptToDeleteErrorsLock.Unlock()
	delete(r.attemptToDeleteErrors, node)
}

func (r *errRecorder) ClearAttemptToOrphanError(node *node) {
	r.attemptToOrphanErrorsLock.Lock()
	defer r.attemptToOrphanErrorsLock.Unlock()
	delete(r.attemptToOrphanErrors, node)
}

func (r *errRecorder) ClearSyncError() {
	r.syncErrorLock.Lock()
	defer r.syncErrorLock.Unlock()
	r.syncError = nil
}

func (r *errRecorder) DumpErrors() []errorResult {
	var results []errorResult

	func() {
		r.syncErrorLock.Lock()
		defer r.syncErrorLock.Unlock()

		if syncError := r.syncError; syncError != nil {
			results = append(results, newErrorResult(SyncError, syncError, nil))
		}
	}()

	func() {
		r.attemptToDeleteErrorsLock.Lock()
		defer r.attemptToDeleteErrorsLock.Unlock()

		for node, errRecord := range r.attemptToDeleteErrors {
			identity := node.identity
			results = append(results, newErrorResult(AttemptToDeleteError, errRecord, &identity))
		}
	}()

	func() {
		r.attemptToOrphanErrorsLock.Lock()
		defer r.attemptToOrphanErrorsLock.Unlock()

		for node, errRecord := range r.attemptToOrphanErrors {
			identity := node.identity
			results = append(results, newErrorResult(AttemptToOrphanError, errRecord, &identity))
		}
	}()

	return results
}

// noopErrRecorder
type noopErrRecorder struct {
}

// newNoopErrorRecorder creates a new noopErrRecorder.
func newNoopErrorRecorder() errorRecorder {
	return &noopErrRecorder{}
}

var _ errorRecorder = &noopErrRecorder{}

func (n noopErrRecorder) SetAttemptToDeleteError(node *node, err error) {
}

func (n noopErrRecorder) SetAttemptToOrphanError(node *node, err error) {
}

func (n noopErrRecorder) SetSyncError(err error) {
}

func (n noopErrRecorder) ClearAttemptToDeleteError(node *node) {
}

func (n noopErrRecorder) ClearAttemptToOrphanError(node *node) {
}

func (n noopErrRecorder) ClearSyncError() {
}

func (n noopErrRecorder) DumpErrors() []errorResult {
	return nil
}
