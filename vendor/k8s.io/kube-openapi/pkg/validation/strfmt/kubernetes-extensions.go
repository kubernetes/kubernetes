// Copyright 2024 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"encoding/json"
	"regexp"
)

const k8sPrefix = "k8s-"

func init() {
	// register formats in the KubernetesExtensions registry:
	//   - k8s-short-name
	//   - k8s-long-name
	shortName := ShortName("")
	Default.Add(k8sPrefix+"short-name", &shortName, IsShortName)

	longName := LongName("")
	Default.Add(k8sPrefix+"long-name", &longName, IsLongName)
}

// ShortName is a name, up to 63 characters long, composed of alphanumeric
// characters and dashes, which cannot begin or end with a dash.
//
// ShortName almost conforms to the definition of a label in DNS (RFC 1123),
// except that uppercase letters are not allowed.
//
// xref: https://github.com/kubernetes/kubernetes/issues/71140
//
// swagger:strfmt k8s-short-name
type ShortName string

func (r ShortName) MarshalText() ([]byte, error) {
	return []byte(string(r)), nil
}

func (r *ShortName) UnmarshalText(data []byte) error { // validation is performed later on
	*r = ShortName(data)
	return nil
}

func (r ShortName) String() string {
	return string(r)
}

func (r ShortName) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(r))
}

func (r *ShortName) UnmarshalJSON(data []byte) error {
	return unmarshalJSON(r, data)
}

func (r *ShortName) DeepCopyInto(out *ShortName) {
	*out = *r
}

func (r *ShortName) DeepCopy() *ShortName {
	return deepCopy(r)
}

const shortNameFmt string = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"

// ShortNameMaxLength is a label's max length in DNS (RFC 1123)
const ShortNameMaxLength int = 63

var shortNameRegexp = regexp.MustCompile("^" + shortNameFmt + "$")

// IsShortName checks if a string is a valid ShortName.
func IsShortName(value string) bool {
	return len(value) <= ShortNameMaxLength &&
		shortNameRegexp.MatchString(value)
}

// LongName is a name, up to 253 characters long, composed of dot-separated
// segments; each segment uses only alphanumerics and dashes (no
// leading/trailing).
//
// LongName almost conforms to the definition of a subdomain in DNS (RFC 1123),
// except that uppercase letters are not allowed, and there is no max length
// limit of 63 for each of the dot-separated DNS Labels that make up the
// subdomain.
//
// xref: https://github.com/kubernetes/kubernetes/issues/71140
// xref: https://github.com/kubernetes/kubernetes/issues/79351
//
// swagger:strfmt k8s-long-name
type LongName string

func (r LongName) MarshalText() ([]byte, error) {
	return []byte(string(r)), nil
}

func (r *LongName) UnmarshalText(data []byte) error { // validation is performed later on
	*r = LongName(data)
	return nil
}

func (r LongName) String() string {
	return string(r)
}

func (r LongName) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(r))
}

func (r *LongName) UnmarshalJSON(data []byte) error {
	return unmarshalJSON(r, data)
}

func (r *LongName) DeepCopyInto(out *LongName) {
	*out = *r
}

func (r *LongName) DeepCopy() *LongName {
	return deepCopy(r)
}

const longNameFmt string = shortNameFmt + "(\\." + shortNameFmt + ")*"

// LongNameMaxLength is a subdomain's max length in DNS (RFC 1123)
const LongNameMaxLength int = 253

var longNameRegexp = regexp.MustCompile("^" + longNameFmt + "$")

// IsLongName checks if a string is a valid LongName.
func IsLongName(value string) bool {
	return len(value) <= LongNameMaxLength &&
		longNameRegexp.MatchString(value)
}
