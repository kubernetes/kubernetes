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

// Package rfc contains constants defined by IETF RFCs that are used
// throughout Kubernetes for validation and naming constraints.
package rfc

// DNS naming constraints from RFC 1123 and RFC 1035.
// See: https://www.rfc-editor.org/rfc/rfc1123
// See: https://www.rfc-editor.org/rfc/rfc1035
const (
	// DNS1123LabelMaxLength is the maximum length of a single label in a
	// DNS-1123 compliant name. A DNS-1123 label must consist of lower case
	// alphanumeric characters or '-', start with an alphabetic character,
	// and end with an alphanumeric character.
	//
	// This applies to: container names, volume names, port names, and other
	// DNS label-constrained fields.
	DNS1123LabelMaxLength int = 63

	// DNS1123SubdomainMaxLength is the maximum length of a DNS-1123 subdomain.
	// A DNS-1123 subdomain must consist of lower case alphanumeric characters,
	// '-' or '.', and must start and end with an alphanumeric character.
	//
	// This applies to: node names, namespace names, service names, pod names,
	// and other DNS subdomain-constrained fields.
	DNS1123SubdomainMaxLength int = 253

	// DNS1035LabelMaxLength is the maximum length of a single label in a
	// DNS-1035 compliant name. A DNS-1035 label must consist of lower case
	// alphanumeric characters or '-', start with an alphabetic character,
	// and end with an alphanumeric character.
	//
	// This is used for service names when they need to be valid DNS names
	// (as opposed to DNS subdomains).
	DNS1035LabelMaxLength int = 63
)
