/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import celconfig "k8s.io/apiserver/pkg/apis/cel"

const (
	// DefaultMaxRequestSizeBytes is the size of the largest request that will be accepted
	DefaultMaxRequestSizeBytes = celconfig.MaxRequestSizeBytes

	// MaxDurationSizeJSON
	// OpenAPI duration strings follow RFC 3339, section 5.6 - see the comment on maxDatetimeSizeJSON
	MaxDurationSizeJSON = 32
	// MaxDatetimeSizeJSON
	// OpenAPI datetime strings follow RFC 3339, section 5.6, and the longest possible
	// such string is 9999-12-31T23:59:59.999999999Z, which has length 30 - we add 2
	// to allow for quotation marks
	MaxDatetimeSizeJSON = 32
	// MinDurationSizeJSON
	// Golang allows a string of 0 to be parsed as a duration, so that plus 2 to account for
	// quotation marks makes 3
	MinDurationSizeJSON = 3
	// JSONDateSize is the size of a date serialized as part of a JSON object
	// RFC 3339 dates require YYYY-MM-DD, and then we add 2 to allow for quotation marks
	JSONDateSize = 12
	// MinDatetimeSizeJSON is the minimal length of a datetime formatted as RFC 3339
	// RFC 3339 datetimes require a full date (YYYY-MM-DD) and full time (HH:MM:SS), and we add 3 for
	// quotation marks like always in addition to the capital T that separates the date and time
	MinDatetimeSizeJSON = 21
	// MinStringSize is the size of literal ""
	MinStringSize = 2
	// MinBoolSize is the length of literal true
	MinBoolSize = 4
	// MinNumberSize is the length of literal 0
	MinNumberSize = 1

	MaxNameFormatRegexSize = 128
)
