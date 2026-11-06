/*
Copyright The Kubernetes Authors.

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

package coverage

import (
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/admission"
)

// AuditAnnotationKeyPrefix prefixes the coverage audit annotation of each plugin.
const AuditAnnotationKeyPrefix = "coverage.admission.k8s.io/"

// AuditAnnotationKey returns the key of the coverage audit annotation written by
// the plugin that dispatches hooks of the given kind, e.g.
// "coverage.admission.k8s.io/validating-webhook".
func AuditAnnotationKey(kind HookKind) string {
	return AuditAnnotationKeyPrefix + string(kind)
}

// AuditAnnotationEntry is one element of the JSON list stored in a coverage
// audit annotation.
type AuditAnnotationEntry struct {
	// Name is the webhook or policy name.
	Name string `json:"name"`
	// Configuration is the webhook configuration name; set for webhooks only.
	Configuration string `json:"configuration,omitempty"`
	// Binding is the policy binding name; set for policies only.
	Binding string `json:"binding,omitempty"`
	// Source is empty if coverage couldn't be determined.
	Source string `json:"source,omitempty"`
	// Target is the request shape the hook didn't intercept, e.g. "UPDATE pods/dynamic".
	Target string `json:"target"`
}

// AddAuditAnnotation records entries as the coverage audit annotation of the
// plugin dispatching hooks of the given kind. It writes nothing for no entries,
// so requests without violations carry no annotation.
//
// Admission annotations can't be overwritten with a different value, so each
// plugin must call it at most once per request, with every entry it collected.
func AddAuditAnnotation(attr admission.Attributes, kind HookKind, entries []AuditAnnotationEntry) error {
	if len(entries) == 0 {
		return nil
	}
	value, err := utiljson.Marshal(entries)
	if err != nil {
		return err
	}
	return attr.AddAnnotation(AuditAnnotationKey(kind), string(value))
}
