/*
Copyright The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	core "k8s.io/kubernetes/pkg/apis/core"
)

// An integer resource accepts any whole number and rejects a fraction, at every
// magnitude. A sub-milli fraction just below an integer stays accepted.
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
		{"just-above-an-integer", "1.0001", true},
		// still fractions, still rejected
		{"huge-fractional-past-the-int64-range", "18446744073709551616500m", true}, // 2^64 + 0.5
		{"fractional-past-the-milli-range", "1000000000000000500m", true},          // 10^15 + 0.5
		// the milli projection rounds these onto a whole number, so they pass
		{"within-one-milli-below-an-integer", "1.9999", false},
		{"within-one-milli-below-one", "0.9999", false},
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

// The same accept/reject boundary must hold on the int64 and the promoted
// inf.Dec forms.
func TestValidateResourceQuantityValueIntegerBoundary(t *testing.T) {
	intResource := core.ResourceName("example.com/device")

	testCases := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"a milli below an integer", "0.999", true},
		{"just inside a milli of an integer", "0.999000001", false},
		{"the integer itself", "1", false},
		{"just above an integer", "1.000000001", true},
		{"a milli below the next integer", "1.999", true},
		{"just inside a milli of the next integer", "1.999000001", false},
		{"the largest integer whose milli value fits", "9223372036854775", false},
		{"the first integer whose milli value overflows", "9223372036854776", false},
		{"a fraction whose milli value is the rail", "9223372036854775807m", true},
		{"a fraction whose milli value overflows past it", "9223372036854775808m", true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, form := range []string{"parsed", "promoted to inf.Dec"} {
				q := resource.MustParse(tc.value)
				if form != "parsed" {
					q.ToDec()
				}
				errs := ValidateResourceQuantityValue(intResource, q, field.NewPath("x"))
				if gotErr := len(errs) > 0; gotErr != tc.wantErr {
					t.Errorf("%s %s: got errors %v, wantErr=%v", tc.value, form, errs, tc.wantErr)
				}
			}
		})
	}
}

// A value the check has always accepted stays accepted through the container path.
func TestValidateContainerResourceRequirementsKeepsRoundedValues(t *testing.T) {
	requirements := &core.ResourceRequirements{
		Limits: core.ResourceList{core.ResourceName("example.com/device"): resource.MustParse("1.9999")},
	}
	if errs := ValidateContainerResourceRequirements(requirements, sets.New[string](), field.NewPath("resources"), PodValidationOptions{}); len(errs) > 0 {
		t.Errorf("a value the check has always accepted was rejected: %v", errs)
	}
}

// ValidateResourceQuotaUpdate revalidates spec.hard without consulting the
// stored value, so a quota already holding a whole number past the milli range
// could not be updated at all, not even to change a label.
func TestValidateResourceQuotaUpdateRevalidatesStoredQuantities(t *testing.T) {
	quota := func(hard, label string) *core.ResourceQuota {
		return &core.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name: "quota", Namespace: "ns", ResourceVersion: "1",
				Labels: map[string]string{"team": label},
			},
			Spec: core.ResourceQuotaSpec{
				Hard: core.ResourceList{core.ResourcePods: resource.MustParse(hard)},
			},
		}
	}
	stored := quota("10000000000000000", "before")
	if errs := ValidateResourceQuotaUpdate(quota("10000000000000000", "after"), stored); len(errs) != 0 {
		t.Errorf("a label-only update to a quota holding 10^16 pods should pass, got %v", errs)
	}

	fractional := quota("18446744073709551616m", "before")
	if errs := ValidateResourceQuotaUpdate(quota("18446744073709551616m", "after"), fractional); len(errs) == 0 {
		t.Errorf("18446744073709551616m is not a whole number of pods and should still be rejected")
	}
}

// The answer belongs to the number, not to how it was written, and it has to be
// the right answer: these are whole numbers.
func TestValidateResourceQuantityValueIgnoresSpelling(t *testing.T) {
	intResource := core.ResourceName("example.com/device")
	spellings := [][]string{
		{"1e16", "10P", "10000000000000000"},
		{"1e18", "1E", "1000000000000000000"},
		{"1e19", "10000000000000000000"},
		{"2e9", "2G", "2000000000"},
	}
	for _, group := range spellings {
		first := len(ValidateResourceQuantityValue(intResource, resource.MustParse(group[0]), field.NewPath("x")))
		for _, value := range group[1:] {
			got := len(ValidateResourceQuantityValue(intResource, resource.MustParse(value), field.NewPath("x")))
			if got != first {
				t.Errorf("%s and %s are the same quantity but got %d and %d errors", group[0], value, first, got)
			}
		}
		if first != 0 {
			t.Errorf("%s is a whole number and should be accepted, got %d errors", group[0], first)
		}
	}
}
