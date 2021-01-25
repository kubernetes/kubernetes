/*
Copyright 2019 The Kubernetes Authors.

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
	"net"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/allocation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

var ValidateServiceIPRangeName = apimachineryvalidation.NameIsDNSSubdomain

func ValidateServiceIPRange(ipRange *allocation.ServiceIPRange) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ipRange.ObjectMeta, false, ValidateServiceIPRangeName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateRange(ipRange.Spec.Range)...)
	return allErrs
}

func validateRange(cidr string) field.ErrorList {
	allErrs := field.ErrorList{}
	ip, subnet, err := net.ParseCIDR(cidr)
	if err != nil || ip == nil || subnet == nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("range"), cidr, "not valid IP subnet, i.e. 10.96.0.0/16,2001:db2::/64"))
	} else if !ip.Equal(subnet.IP) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("range"), ip, "IP does not match subnet address"))
	}
	return allErrs
}

var ValidateServiceIPName = NameIsServiceIP

// NameIsServiceIP validates that the name is the decimal representation of an IP address
func NameIsServiceIP(name string, prefix bool) []string {
	var errs []string
	if prefix {
		errs = append(errs, "prefix not allowed")
	}
	ip := net.ParseIP(name)
	if ip == nil {
		errs = append(errs, "not a valid ip address")
	}
	return errs
}

func ValidateServiceIP(ipAddress *allocation.ServiceIP) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ipAddress.ObjectMeta, false, ValidateServiceIPName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateServiceIPRangeRef(ipAddress.Spec.ServiceIPRangeRef)...)
	return allErrs

}

func validateServiceIPRangeRef(rangeRegf allocation.ServiceIPRangeRef) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO validate reference to the ServiceIPRange allocator
	return allErrs
}
