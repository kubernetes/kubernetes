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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type AuditEvent struct {
	Level             auditinternal.Level
	Stage             auditinternal.Stage
	RequestURI        string
	Verb              string
	Code              int32
	User              string
	Resource          string
	Namespace         string
	RequestObject     bool
	ResponseObject    bool
	AuthorizeDecision string
}

// Search the audit log for the expected audit lines.
func CheckAuditLines(stream io.Reader, expected []AuditEvent, version schema.GroupVersion) (missing []AuditEvent, err error) {
	expectations := map[AuditEvent]bool{}
	for _, event := range expected {
		expectations[event] = false
	}

	scanner := bufio.NewScanner(stream)
	for scanner.Scan() {
		line := scanner.Text()
		event, err := parseAuditLine(line, version)
		if err != nil {
			return expected, err
		}

		// If the event was expected, mark it as found.
		if _, found := expectations[event]; found {
			expectations[event] = true
		}
	}
	if err := scanner.Err(); err != nil {
		return expected, err
	}

	missing = make([]AuditEvent, 0)
	for event, found := range expectations {
		if !found {
			missing = append(missing, event)
		}
	}
	return missing, nil
}

func parseAuditLine(line string, version schema.GroupVersion) (AuditEvent, error) {
	e := &auditinternal.Event{}
	decoder := audit.Codecs.UniversalDecoder(version)
	if err := runtime.DecodeInto(decoder, []byte(line), e); err != nil {
		return AuditEvent{}, fmt.Errorf("failed decoding buf: %s, apiVersion: %s", line, version)
	}

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
	event.AuthorizeDecision = e.Annotations["authorization.k8s.io/decision"]
	return event, nil
}
