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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	core "k8s.io/kubernetes/pkg/apis/core"
)

// The integer-resource check used value.MilliValue()%1000, and MilliValue()
// multiplies by 1000, so it wrapped for a whole number past the int64 range and
// reported the value as fractional. 10^16 and 10^18 are integers but were
// rejected. AsScale(0) reports integer-ness exactly.
func TestValidateResourceQuantityValueIntegerOverflow(t *testing.T) {
	intResource := core.ResourceName("example.com/device") // an integer resource

	cases := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"small-integer", "4", false},
		{"integer-in-milli-form", "1000m", false},
		{"whole-number-past-the-milli-range", "10000000000000000", false},    // 10^16
		{"larger-whole-number-past-the-range", "1000000000000000000", false}, // 10^18
		{"whole-number-past-the-int64-range", "18446744073709551616", false}, // 2^64
		{"fractional", "1500m", true},
		{"fractional-milli", "2500m", true},
		// Fractional values past the milli-overflow point: MilliValue()%1000 is
		// unreliable there, so this pins that AsScale still rejects them.
		{"huge-fractional-past-the-int64-range", "18446744073709551616500m", true}, // 2^64 + 0.5
		{"fractional-past-the-milli-range", "1000000000000000500m", true},          // 10^15 + 0.5
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q := resource.MustParse(tc.value)
			errs := ValidateResourceQuantityValue(intResource, q, field.NewPath("x"))
			if gotErr := len(errs) > 0; gotErr != tc.wantErr {
				t.Errorf("value %s: got errors %v, wantErr=%v", tc.value, errs, tc.wantErr)
			}
		})
	}
}
