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
	"reflect"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"

	"github.com/davecgh/go-spew/spew"
)

// Conversion error conveniently packages up errors in conversions.
type ConversionError struct {
	In, Out interface{}
	Message string
}

// Return a helpful string about the error
func (c *ConversionError) Error() string {
	return spew.Sprintf(
		"Conversion error: %s. (in: %v(%+v) out: %v)",
		c.Message, reflect.TypeOf(c.In), c.In, reflect.TypeOf(c.Out),
	)
}

// Semantic can do semantic deep equality checks for api objects.
// Example: api.Semantic.DeepEqual(aPod, aPodWithNonNilButEmptyMaps) == true
var Semantic = conversion.EqualitiesOrDie(
	func(a, b resource.Quantity) bool {
		// Ignore formatting, only care that numeric value stayed the same.
		// TODO: if we decide it's important, after we drop v1beta1/2, we
		// could start comparing format.
		//
		// Uninitialized quantities are equivilent to 0 quantities.
		if a.Amount == nil && b.MilliValue() == 0 {
			return true
		}
		if b.Amount == nil && a.MilliValue() == 0 {
			return true
		}
		if a.Amount == nil || b.Amount == nil {
			return false
		}
		return a.Amount.Cmp(b.Amount) == 0
	},
	pullPoliciesEqual,
)

// TODO: Address these per #1502

func IsPullAlways(p PullPolicy) bool {
	return pullPoliciesEqual(p, PullAlways)
}

func IsPullNever(p PullPolicy) bool {
	return pullPoliciesEqual(p, PullNever)
}

func IsPullIfNotPresent(p PullPolicy) bool {
	// Default to pull if not present
	if len(p) == 0 {
		return true
	}
	return pullPoliciesEqual(p, PullIfNotPresent)
}

func pullPoliciesEqual(p1, p2 PullPolicy) bool {
	return strings.ToLower(string(p1)) == strings.ToLower(string(p2))
}
