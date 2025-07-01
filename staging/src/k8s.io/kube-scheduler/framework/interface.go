/*
Copyright 2025 The Kubernetes Authors.

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

// This file defines the scheduling framework plugin interfaces.

package framework

import (
	"errors"
	"strings"

	"github.com/google/go-cmp/cmp"         //nolint:depguard
	"github.com/google/go-cmp/cmp/cmpopts" //nolint:depguard
)

// Code is the Status code/type which is returned from plugins.
type Code int

// These are predefined codes used in a Status.
// Note: when you add a new status, you have to add it in `codes` slice below.
const (
	// Success means that plugin ran correctly and found pod schedulable.
	// NOTE: A nil status is also considered as "Success".
	Success Code = iota
	// Error is one of the failures, used for internal plugin errors, unexpected input, etc.
	// Plugin shouldn't return this code for expected failures, like Unschedulable.
	// Since it's the unexpected failure, the scheduling queue registers the pod without unschedulable plugins.
	// Meaning, the Pod will be requeued to activeQ/backoffQ soon.
	Error
	// Unschedulable is one of the failures, used when a plugin finds a pod unschedulable.
	// If it's returned from PreFilter or Filter, the scheduler might attempt to
	// run other postFilter plugins like preemption to get this pod scheduled.
	// Use UnschedulableAndUnresolvable to make the scheduler skipping other postFilter plugins.
	// The accompanying status message should explain why the pod is unschedulable.
	//
	// We regard the backoff as a penalty of wasting the scheduling cycle.
	// When the scheduling queue requeues Pods, which was rejected with Unschedulable in the last scheduling,
	// the Pod goes through backoff.
	Unschedulable
	// UnschedulableAndUnresolvable is used when a plugin finds a pod unschedulable and
	// other postFilter plugins like preemption would not change anything.
	// See the comment on PostFilter interface for more details about how PostFilter should handle this status.
	// Plugins should return Unschedulable if it is possible that the pod can get scheduled
	// after running other postFilter plugins.
	// The accompanying status message should explain why the pod is unschedulable.
	//
	// We regard the backoff as a penalty of wasting the scheduling cycle.
	// When the scheduling queue requeues Pods, which was rejected with UnschedulableAndUnresolvable in the last scheduling,
	// the Pod goes through backoff.
	UnschedulableAndUnresolvable
	// Wait is used when a Permit plugin finds a pod scheduling should wait.
	Wait
	// Skip is used in the following scenarios:
	// - when a Bind plugin chooses to skip binding.
	// - when a PreFilter plugin returns Skip so that coupled Filter plugin/PreFilterExtensions() will be skipped.
	// - when a PreScore plugin returns Skip so that coupled Score plugin will be skipped.
	Skip
	// Pending means that the scheduling process is finished successfully,
	// but the plugin wants to stop the scheduling cycle/binding cycle here.
	//
	// For example, if your plugin has to notify the scheduling result to an external component,
	// and wait for it to complete something **before** binding.
	// It's different from when to return Unschedulable/UnschedulableAndUnresolvable,
	// because in this case, the scheduler decides where the Pod can go successfully,
	// but we need to wait for the external component to do something based on that scheduling result.
	//
	// We regard the backoff as a penalty of wasting the scheduling cycle.
	// In the case of returning Pending, we cannot say the scheduling cycle is wasted
	// because the scheduling result is used to proceed the Pod's scheduling forward,
	// that particular scheduling cycle is failed though.
	// So, Pods rejected by such reasons don't need to suffer a penalty (backoff).
	// When the scheduling queue requeues Pods, which was rejected with Pending in the last scheduling,
	// the Pod goes to activeQ directly ignoring backoff.
	Pending
)

// This list should be exactly the same as the codes iota defined above in the same order.
var codes = []string{"Success", "Error", "Unschedulable", "UnschedulableAndUnresolvable", "Wait", "Skip", "Pending"}

func (c Code) String() string {
	return codes[c]
}

// Status indicates the result of running a plugin. It consists of a code, a
// message, (optionally) an error, and a plugin name it fails by.
// When the status code is not Success, the reasons should explain why.
// And, when code is Success, all the other fields should be empty.
// NOTE: A nil Status is also considered as Success.
type Status struct {
	code    Code
	reasons []string
	err     error
	// plugin is an optional field that records the plugin name causes this status.
	// It's set by the framework when code is Unschedulable, UnschedulableAndUnresolvable or Pending.
	plugin string
}

func (s *Status) WithError(err error) *Status {
	s.err = err
	return s
}

// Code returns code of the Status.
func (s *Status) Code() Code {
	if s == nil {
		return Success
	}
	return s.code
}

// Message returns a concatenated message on reasons of the Status.
func (s *Status) Message() string {
	if s == nil {
		return ""
	}
	return strings.Join(s.Reasons(), ", ")
}

// SetPlugin sets the given plugin name to s.plugin.
func (s *Status) SetPlugin(plugin string) {
	s.plugin = plugin
}

// WithPlugin sets the given plugin name to s.plugin,
// and returns the given status object.
func (s *Status) WithPlugin(plugin string) *Status {
	s.SetPlugin(plugin)
	return s
}

// Plugin returns the plugin name which caused this status.
func (s *Status) Plugin() string {
	return s.plugin
}

// Reasons returns reasons of the Status.
func (s *Status) Reasons() []string {
	if s.err != nil {
		return append([]string{s.err.Error()}, s.reasons...)
	}
	return s.reasons
}

// AppendReason appends given reason to the Status.
func (s *Status) AppendReason(reason string) {
	s.reasons = append(s.reasons, reason)
}

// IsSuccess returns true if and only if "Status" is nil or Code is "Success".
func (s *Status) IsSuccess() bool {
	return s.Code() == Success
}

// IsWait returns true if and only if "Status" is non-nil and its Code is "Wait".
func (s *Status) IsWait() bool {
	return s.Code() == Wait
}

// IsSkip returns true if and only if "Status" is non-nil and its Code is "Skip".
func (s *Status) IsSkip() bool {
	return s.Code() == Skip
}

// IsRejected returns true if "Status" is Unschedulable (Unschedulable, UnschedulableAndUnresolvable, or Pending).
func (s *Status) IsRejected() bool {
	code := s.Code()
	return code == Unschedulable || code == UnschedulableAndUnresolvable || code == Pending
}

// AsError returns nil if the status is a success, a wait or a skip; otherwise returns an "error" object
// with a concatenated message on reasons of the Status.
func (s *Status) AsError() error {
	if s.IsSuccess() || s.IsWait() || s.IsSkip() {
		return nil
	}
	if s.err != nil {
		return s.err
	}
	return errors.New(s.Message())
}

// Equal checks equality of two statuses. This is useful for testing with
// cmp.Equal.
func (s *Status) Equal(x *Status) bool {
	if s == nil || x == nil {
		return s.IsSuccess() && x.IsSuccess()
	}
	if s.code != x.code {
		return false
	}
	if !cmp.Equal(s.err, x.err, cmpopts.EquateErrors()) {
		return false
	}
	if !cmp.Equal(s.reasons, x.reasons) {
		return false
	}
	return cmp.Equal(s.plugin, x.plugin)
}

func (s *Status) String() string {
	return s.Message()
}

// NewStatus makes a Status out of the given arguments and returns its pointer.
func NewStatus(code Code, reasons ...string) *Status {
	s := &Status{
		code:    code,
		reasons: reasons,
	}
	return s
}

// AsStatus wraps an error in a Status.
func AsStatus(err error) *Status {
	if err == nil {
		return nil
	}
	return &Status{
		code: Error,
		err:  err,
	}
}
