/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"strings"
)

func IsPullAlways(p PullPolicy) bool {
	// Default to pull always
	if len(p) == 0 {
		return true
	}
	return pullPoliciesEqual(p, PullAlways)
}

func IsPullNever(p PullPolicy) bool {
	return pullPoliciesEqual(p, PullNever)
}

func IsPullIfNotPresent(p PullPolicy) bool {
	return pullPoliciesEqual(p, PullIfNotPresent)
}

func pullPoliciesEqual(p1, p2 PullPolicy) bool {
	return strings.ToLower(string(p1)) == strings.ToLower(string(p2))
}
