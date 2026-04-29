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

// Package guardrails defines the JSON schema (Rule, Report) emitted by
// validation-gen --report-rules: the declarative-validation rules declared
// by DV tags, indexed by (Group, Version, Kind, field path). The schema
// feeds external "guardrail" tools (e.g. test-coverage verifiers) that
// reason about the declared rule set.
package guardrails

// Rule is one declared field-validation error.
type Rule struct {
	ErrorType string `json:"errorType"`
	Origin    string `json:"origin,omitempty"`
}

// Report is the marshaled output for one (Group, Version): every Kind's
// field path → declared rules. Group is empty for the core API group;
// Version is empty for non-API packages (in which case Group falls back
// to the package import path).
type Report struct {
	Group   string                       `json:"group"`
	Version string                       `json:"version"`
	Kinds   map[string]map[string][]Rule `json:"kinds"`
}
