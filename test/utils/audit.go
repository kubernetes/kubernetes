/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"bufio"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/mutating"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

// AuditEvent is a simplified representation of an audit event for testing purposes
type AuditEvent struct {
	ID                 types.UID
	Level              auditinternal.Level
	Stage              auditinternal.Stage
	RequestURI         string
	Verb               string
	Code               int32
	User               string
	ImpersonatedUser   string
	ImpersonatedGroups string
	Resource           string
	Namespace          string
	RequestObject      bool
	ResponseObject     bool
	AuthorizeDecision  string

	// The Check functions in this package takes ownerships of these maps. You should
	// not reference these maps after calling the Check functions.
	AdmissionWebhookMutationAnnotations map[string]string
	AdmissionWebhookPatchAnnotations    map[string]string

	// Only populated when a filter is provided to testEventFromInternalFiltered
	CustomAuditAnnotations map[string]string
}

type AuditAnnotationsFilter func(key, val string) bool

// MissingEventsReport provides an analysis if any events are missing
type MissingEventsReport struct {
	FirstEventChecked *auditinternal.Event
	LastEventChecked  *auditinternal.Event
	NumEventsChecked  int
	MissingEvents     []AuditEvent
}

// String returns a human readable string representation of the report
func (m *MissingEventsReport) String() string {
	return fmt.Sprintf(`missing %d events

- first event checked: %#v

- last event checked: %#v

- number of events checked: %d

- missing events: %#v`, len(m.MissingEvents), m.FirstEventChecked, m.LastEventChecked, m.NumEventsChecked, m.MissingEvents)
}

// CheckAuditLines searches the audit log for the expected audit lines.
func CheckAuditLines(stream io.Reader, expected []AuditEvent, version schema.GroupVersion) (missingReport *MissingEventsReport, err error) {
	return CheckAuditLinesFiltered(stream, expected, version, nil)
}

// CheckAuditLinesFiltered searches the audit log for the expected audit lines, customAnnotationsFilter
// controls which audit annotations are added to AuditEvent.CustomAuditAnnotations.
// If the customAnnotationsFilter is nil, AuditEvent.CustomAuditAnnotations will be empty.
func CheckAuditLinesFiltered(stream io.Reader, expected []AuditEvent, version schema.GroupVersion, customAnnotationsFilter AuditAnnotationsFilter) (missingReport *MissingEventsReport, err error) {
	expectations := newAuditEventTracker(expected)

	scanner := bufio.NewScanner(stream)

	missingReport = &MissingEventsReport{
		MissingEvents: expected,
	}

	var i int
	for i = 0; scanner.Scan(); i++ {
		line := scanner.Text()

		e := &auditinternal.Event{}
		decoder := audit.Codecs.UniversalDecoder(version)
		if err := runtime.DecodeInto(decoder, []byte(line), e); err != nil {
			return missingReport, fmt.Errorf("failed decoding buf: %s, apiVersion: %s", line, version)
		}
		if i == 0 {
			missingReport.FirstEventChecked = e
		}
		missingReport.LastEventChecked = e

		event, err := testEventFromInternalFiltered(e, customAnnotationsFilter)
		if err != nil {
			return missingReport, err
		}

		expectations.Mark(event)
	}
	if err := scanner.Err(); err != nil {
		return missingReport, err
	}

	missingReport.MissingEvents = expectations.Missing()
	missingReport.NumEventsChecked = i
	return missingReport, nil
}

// testEventFromInternalFiltered takes an internal audit event and returns a test event, customAnnotationsFilter
// controls which audit annotations are added to AuditEvent.CustomAuditAnnotations.
// If the customAnnotationsFilter is nil, AuditEvent.CustomAuditAnnotations will be empty.
func testEventFromInternalFiltered(e *auditinternal.Event, customAnnotationsFilter AuditAnnotationsFilter) (AuditEvent, error) {
	event := AuditEvent{
		Level:      e.Level,
		Stage:      e.Stage,
		RequestURI: e.RequestURI,
		Verb:       e.Verb,
		User:       e.User.Username,
	}
	if e.ObjectRef != nil {
		event.Namespace = e.ObjectRef.Namespace
		event.Resource = e.ObjectRef.Resource
	}
	if e.ResponseStatus != nil {
		event.Code = e.ResponseStatus.Code
	}
	if e.ResponseObject != nil {
		event.ResponseObject = true
	}
	if e.RequestObject != nil {
		event.RequestObject = true
	}
	if e.ImpersonatedUser != nil {
		event.ImpersonatedUser = e.ImpersonatedUser.Username
		sort.Strings(e.ImpersonatedUser.Groups)
		event.ImpersonatedGroups = strings.Join(e.ImpersonatedUser.Groups, ",")
	}
	event.AuthorizeDecision = e.Annotations["authorization.k8s.io/decision"]
	for k, v := range e.Annotations {
		if strings.HasPrefix(k, mutating.PatchAuditAnnotationPrefix) {
			if event.AdmissionWebhookPatchAnnotations == nil {
				event.AdmissionWebhookPatchAnnotations = map[string]string{}
			}
			event.AdmissionWebhookPatchAnnotations[k] = v
		} else if strings.HasPrefix(k, mutating.MutationAuditAnnotationPrefix) {
			if event.AdmissionWebhookMutationAnnotations == nil {
				event.AdmissionWebhookMutationAnnotations = map[string]string{}
			}
			event.AdmissionWebhookMutationAnnotations[k] = v
		} else if customAnnotationsFilter != nil && customAnnotationsFilter(k, v) {
			if event.CustomAuditAnnotations == nil {
				event.CustomAuditAnnotations = map[string]string{}
			}
			event.CustomAuditAnnotations[k] = v
		}
	}
	return event, nil
}

// auditEvent is a private wrapper on top of AuditEvent used by auditEventTracker
type auditEvent struct {
	event AuditEvent
	found bool
}

// auditEventTracker keeps track of AuditEvent expectations and marks matching events as found
type auditEventTracker struct {
	events []*auditEvent
}

// newAuditEventTracker creates a tracker that tracks whether expect events are found
func newAuditEventTracker(expected []AuditEvent) *auditEventTracker {
	expectations := &auditEventTracker{events: []*auditEvent{}}
	for _, event := range expected {
		// we copy the references to the maps in event
		expectations.events = append(expectations.events, &auditEvent{event: event, found: false})
	}
	return expectations
}

// Mark marks the given event as found if it's expected
func (t *auditEventTracker) Mark(event AuditEvent) {
	for _, e := range t.events {
		if reflect.DeepEqual(e.event, event) {
			e.found = true
		}
	}
}

// Missing reports events that are expected but not found
func (t *auditEventTracker) Missing() []AuditEvent {
	var missing []AuditEvent
	for _, e := range t.events {
		if !e.found {
			missing = append(missing, e.event)
		}
	}
	return missing
}
