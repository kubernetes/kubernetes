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

package resourceversion

import (
	"fmt"
	"strings"
)

type InvalidResourceVersion struct {
	rv string
}

func (i InvalidResourceVersion) Error() string {
	return fmt.Sprintf("resource version is not well formed: %s", i.rv)
}

// CompareResourceVersion runs a comparison between two ResourceVersions. This
// only has semantic meaning when the comparison is done on two objects of the
// same resource. The return values are:
//
//	-1: If RV a < RV b
//	 0: If RV a == RV b
//	+1: If RV a > RV b
//
// The function will return an error if the resource version is not a properly
// formatted positive integer, but has no restriction on length. A properly
// formatted integer will not contain leading zeros or non integer characters.
// Zero is also considered an invalid value as it is used as a special value in
// list/watch events and will never be a live resource version.
func CompareResourceVersion(a, b string) (int, error) {
	if !isWellFormed(a) {
		return 0, InvalidResourceVersion{rv: a}
	}
	if !isWellFormed(b) {
		return 0, InvalidResourceVersion{rv: b}
	}
	// both are well-formed integer strings with no leading zeros
	aLen := len(a)
	bLen := len(b)
	switch {
	case aLen < bLen:
		// shorter is less
		return -1, nil
	case aLen > bLen:
		// longer is greater
		return 1, nil
	default:
		// equal-length compares lexically
		return strings.Compare(a, b), nil
	}
}

func isWellFormed(s string) bool {
	if len(s) == 0 {
		return false
	}
	if s[0] == '0' {
		return false
	}
	for i := range s {
		if !isDigit(s[i]) {
			return false
		}
	}
	return true
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}
