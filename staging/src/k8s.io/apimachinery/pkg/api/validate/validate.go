/*
Copyright 2017 The Kubernetes Authors.

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

package validate

import (
	"net"
	"time"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NameValidator validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names.  If the name is
// not valid for any reason, this returns a list of descriptions of individual
// characteristics of the value that were not valid.  Otherwise this returns an
// empty list or nil.
type NameValidator = func(name string) []string

// Name calls a provided NameValidator and turns the results into a field.ErrorList.
func Name(fn NameValidator, name string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range fn(name) {
		allErrs = append(allErrs, field.Invalid(fldPath, name, msg))
	}
	return allErrs
}

const isNegativeErrorMsg string = `must be greater than or equal to 0`

// NonNegative validates that given int64 is not negative.
func NonNegative(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNegativeErrorMsg))
	}
	return allErrs
}

// NonNegativeQuantity validates that a given Quantity is not negative.
func NonNegativeQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNegativeErrorMsg))
	}
	return allErrs
}

// NonNegativeDuration validates that given Duration is not negative.
func NonNegativeDuration(value time.Duration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if int64(value) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, `must be greater than or equal to 0`))
	}
	return allErrs
}

const isNotPositiveErrorMsg string = `must be greater than 0`

// Positive validates that given int64 is positive.
func Positive(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNotPositiveErrorMsg))
	}
	return allErrs
}

// PositiveQuantity validates that a given Quantity is positive.
func PositiveQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNotPositiveErrorMsg))
	}
	return allErrs
}

// InRange validates that a given int64 is within an inclusive range.
func InRange(value, lo, hi int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsInRange(value, lo, hi) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// DNS1123Label validates that a name is a proper DNS label.
func DNS1123Label(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Label(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// DNS1123Subdomain validates that a name is a proper DNS subdomain.
func DNS1123Subdomain(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Subdomain(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// Immutablevalidates that a value has not changed.
func Immutable(after, before interface{}, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !equality.Semantic.DeepEqual(before, after) {
		allErrs = append(allErrs, field.Invalid(fldPath, after, `field is immutable`))
	}
	return allErrs
}

// IPString validates that a string is a valid IP address. Checks is an
// open-ended list of functions that can check some fact and return additional
// errors.
func IPString(value string, fldPath *field.Path, fns ...func(ip net.IP) []string) field.ErrorList {
	ip, allErrs := parseIP(value, fldPath)
	if ip == nil {
		return allErrs
	}
	for _, fn := range fns {
		for _, msg := range fn(ip) {
			allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
		}
	}
	return allErrs
}

// Parse an IP xor return an error.
func parseIP(value string, fldPath *field.Path) (net.IP, field.ErrorList) {
	ip := net.ParseIP(value)
	if ip == nil {
		return nil, field.ErrorList{
			field.Invalid(fldPath, value, "must be a valid IP address"),
		}
	}
	return ip, nil
}

// Returns true iff the IP is a v6 address.
func isIPV6(ip net.IP) bool {
	// The Go way to check this is to try to convert to 4-byte form.
	return ip.To4() == nil
}

// IPNotUnspecified can be passed to IPString as an additional check to ensure
// the IP is not "unspecified" (0.0.0.0 or ::).
func IPNotUnspecified(ip net.IP) []string {
	if ip.IsUnspecified() {
		return []string{"may not be unspecified"}
	}
	return nil
}

// IPNotLoopback can be passed to IPString as an additional check to ensure
// the IP is not in the loopback range.
func IPNotLoopback(ip net.IP) []string {
	if ip.IsLoopback() {
		msg4 := "may not be in the loopback range (127.0.0.0/8)"
		msg6 := "may not be loopback"
		msg := msg4
		if isIPV6(ip) {
			msg = msg6
		}
		return []string{msg}
	}
	return nil
}

// IPNotLinkLocal can be passed to IPString as an additional check to ensure
// the IP is not in the link-local ranges.
func IPNotLinkLocal(ip net.IP) []string {
	var msgs []string // nil
	if ip.IsLinkLocalUnicast() {
		msg4 := "may not be in the link-local range (169.254.0.0/16)"
		msg6 := "may not be in the link-local range (fe80::/10)"
		msg := msg4
		if isIPV6(ip) {
			msg = msg6
		}
		msgs = append(msgs, msg)
	}
	if ip.IsLinkLocalMulticast() {
		msg4 := "may not be in the link-local multicast range (224.0.0.0/24)"
		msg6 := "may not be in the link-local multicast range (ff02::/16)"
		msg := msg4
		if isIPV6(ip) {
			msg = msg6
		}
		msgs = append(msgs, msg)
	}
	return msgs
}
