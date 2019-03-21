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
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
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
}

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
	expectations := buildEventExpectations(expected)

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

		event, err := testEventFromInternal(e)
		if err != nil {
			return missingReport, err
		}

		// If the event was expected, mark it as found.
		if _, found := expectations[event]; found {
			expectations[event] = true
		}
	}
	if err := scanner.Err(); err != nil {
		return missingReport, err
	}

	missingEvents := findMissing(expectations)
	missingReport.MissingEvents = missingEvents
	missingReport.NumEventsChecked = i
	return missingReport, nil
}

// CheckAuditList searches an audit event list for the expected audit events.
func CheckAuditList(el auditinternal.EventList, expected []AuditEvent) (missing []AuditEvent, err error) {
	expectations := buildEventExpectations(expected)

	for _, e := range el.Items {
		event, err := testEventFromInternal(&e)
		if err != nil {
			return expected, err
		}

		// If the event was expected, mark it as found.
		if _, found := expectations[event]; found {
			expectations[event] = true
		}
	}

	missing = findMissing(expectations)
	return missing, nil
}

// CheckForDuplicates checks a list for duplicate events
func CheckForDuplicates(el auditinternal.EventList) (auditinternal.EventList, error) {
	// eventMap holds a map of audit events with just a nil value
	eventMap := map[AuditEvent]*bool{}
	duplicates := auditinternal.EventList{}
	var err error
	for _, e := range el.Items {
		event, err := testEventFromInternal(&e)
		if err != nil {
			return duplicates, err
		}
		event.ID = e.AuditID
		if _, ok := eventMap[event]; ok {
			duplicates.Items = append(duplicates.Items, e)
			err = fmt.Errorf("failed duplicate check")
			continue
		}
		eventMap[event] = nil
	}
	return duplicates, err
}

// buildEventExpectations creates a bool map out of a list of audit events
func buildEventExpectations(expected []AuditEvent) map[AuditEvent]bool {
	expectations := map[AuditEvent]bool{}
	for _, event := range expected {
		expectations[event] = false
	}
	return expectations
}

// testEventFromInternal takes an internal audit event and returns a test event
func testEventFromInternal(e *auditinternal.Event) (AuditEvent, error) {
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
	return event, nil
}

// findMissing checks for false values in the expectations map and returns them as a list
func findMissing(expectations map[AuditEvent]bool) []AuditEvent {
	var missing []AuditEvent
	for event, found := range expectations {
		if !found {
			missing = append(missing, event)
		}
	}
	return missing
}
