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

// Package rfc contains well-known length constants for the
// Kubernetes-specific naming formats that are historically named after the
// DNS RFCs (1123 and 1035). The Kubernetes formats do not strictly match the
// RFC definitions — they are subsets tuned for Kubernetes object names — but
// the identifiers below are kept as-is because they are widely used across
// the codebase and the ecosystem. New code that needs a name limit should
// prefer these constants over hard-coding the integer.
//
// See also KEP-5311 (relaxed validation for Service names) for the
// in-progress effort to decouple Service name validation from the
// RFC-1035 framing.
package rfc

// Length constants for Kubernetes name-like fields. The names retain their
// "DNS1123"/"DNS1035" suffixes for source compatibility with the rest of
// the ecosystem, but the formats they gate are Kubernetes-specific and are
// not identical to the corresponding RFC grammars.
const (
	// DNS1123LabelMaxLength is the maximum length of a single "DNS-1123 label"
	// as used by Kubernetes for fields such as container names, volume names,
	// and port names.
	//
	// Kubernetes requires such values to consist of lower case alphanumeric
	// characters or '-', start with an alphabetic character, and end with an
	// alphanumeric character.
	DNS1123LabelMaxLength int = 63

	// DNS1123SubdomainMaxLength is the maximum length of a "DNS-1123 subdomain"
	// as used by Kubernetes for most object names (namespace names, node
	// names, service names, pod names, etc.).
	//
	// Kubernetes requires such values to consist of lower case alphanumeric
	// characters, '-' or '.', and to start and end with an alphanumeric
	// character.
	DNS1123SubdomainMaxLength int = 253

	// DNS1035LabelMaxLength is the maximum length of a "DNS-1035 label" as
	// historically used by Kubernetes Service names.
	//
	// Note: KEP-5311 is relaxing Service name validation away from the
	// RFC-1035 framing. New code SHOULD NOT introduce new uses of this
	// constant; it is retained here for in-tree consumers only.
	DNS1035LabelMaxLength int = 63
)
