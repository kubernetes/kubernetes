/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation"
)

const IsNegativeErrorMsg string = `must be greater than or equal to 0`

//FIXME: call the func signature "validate.ValidatorFunc" or something more generic
//FIXME: get rid of all this, just use "content.IsDNS..." ?
// NameIsDNSSubdomain is a validate.NameValidator for names that must be a DNS subdomain.
func NameIsDNSSubdomain(name string) []string {
	return validation.IsDNS1123Subdomain(name)
}

// NameIsDNSLabel is a validate.NameValidator for names that must be a DNS 1123 label.
func NameIsDNSLabel(name string) []string {
	return validation.IsDNS1123Label(name)
}

// NameIsDNS1035Label is a validate.NameValidator for names that must be a DNS 952 label.
func NameIsDNS1035Label(name string) []string {
	return validation.IsDNS1035Label(name)
}

// ValidateNamespaceName can be used to check whether the given namespace name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateNamespaceName = NameIsDNSLabel

// ValidateServiceAccountName can be used to check whether the given service account name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateServiceAccountName = NameIsDNSSubdomain
