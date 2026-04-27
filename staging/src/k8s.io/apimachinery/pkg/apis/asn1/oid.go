/*
Copyright 2024 The Kubernetes Authors.

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

package asn1

import "encoding/asn1"

// These constants store suffixes for use with the CNCF Private Enterprise Number allocated to Kubernetes:
// https://www.iana.org/assignments/enterprise-numbers.txt
//
// Root: 1.3.6.1.4.1.57683
//
// Cloud Native Computing Foundation
const (
	// single-value, string value
	x509UIDSuffix = 2
)

func makeOID(suffix int) asn1.ObjectIdentifier {
	return asn1.ObjectIdentifier{1, 3, 6, 1, 4, 1, 57683, suffix}
}

// X509UID returns an OID (1.3.6.1.4.1.57683.2) for an element of an x509 distinguished name representing a user UID.
// The UID is a unique value for a particular user that will change if the user is removed from the system
// and another user is added with the same username.
//
// This element must not appear more than once in a distinguished name, and the value must be a string
func X509UID() asn1.ObjectIdentifier {
	return makeOID(x509UIDSuffix)
}
