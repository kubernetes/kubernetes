/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func ValidateMasterConfiguration(c *kubeadm.MasterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	return allErrs
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	return allErrs
}

func ValidateDiscovery(c *kubeadm.Discovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	var count int
	if c.Token != nil {
		allErrs = append(allErrs, ValidateTokenDiscovery(c.Token, fldPath)...)
		count++
	}
	if c.File != nil {
		allErrs = append(allErrs, ValidateFileDiscovery(c.File, fldPath)...)
		count++
	}
	if c.HTTPS != nil {
		allErrs = append(allErrs, ValidateHTTPSDiscovery(c.HTTPS, fldPath)...)
		count++
	}
	if count != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "exactly one discovery strategy can be provided"))
	}
	return allErrs
}

func ValidateFileDiscovery(c *kubeadm.FileDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func ValidateHTTPSDiscovery(c *kubeadm.HTTPSDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func ValidateTokenDiscovery(c *kubeadm.TokenDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}
