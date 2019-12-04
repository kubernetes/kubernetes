// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tag

import "errors"

const (
	maxKeyLength = 255

	// valid are restricted to US-ASCII subset (range 0x20 (' ') to 0x7e ('~')).
	validKeyValueMin = 32
	validKeyValueMax = 126
)

var (
	errInvalidKeyName = errors.New("invalid key name: only ASCII characters accepted; max length must be 255 characters")
	errInvalidValue   = errors.New("invalid value: only ASCII characters accepted; max length must be 255 characters")
)

func checkKeyName(name string) bool {
	if len(name) == 0 {
		return false
	}
	if len(name) > maxKeyLength {
		return false
	}
	return isASCII(name)
}

func isASCII(s string) bool {
	for _, c := range s {
		if (c < validKeyValueMin) || (c > validKeyValueMax) {
			return false
		}
	}
	return true
}

func checkValue(v string) bool {
	if len(v) > maxKeyLength {
		return false
	}
	return isASCII(v)
}
