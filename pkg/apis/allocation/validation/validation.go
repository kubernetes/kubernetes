/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/allocation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

func ValidateIPRequest(ipRequest *allocation.IPRequest) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ipRequest.ObjectMeta, false, ValidateIPRequestName, field.NewPath("metadata"))
	return allErrs
}

// ValidateIPAddressName validates that the name is the decimal representation of an IP address
func ValidateIPRequestName(name string, prefix bool) []string {
	var errs []string
	if prefix {
		errs = append(errs, "prefix not allowed")
	}
	ip := net.ParseIP(name)
	if ip == nil {
		errs = append(errs, "not a valid ip address")
	} else if ip.String() != name {
		errs = append(errs, "name is not an ip canonical name")
	}
	return errs
}
