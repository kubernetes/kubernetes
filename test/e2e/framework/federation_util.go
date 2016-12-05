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

package framework

import (
	"fmt"
	"regexp"

	"k8s.io/kubernetes/pkg/api/validation"
	validationutil "k8s.io/kubernetes/pkg/util/validation"
)

// GetValidDNSSubdomainName massages the given name to be a valid dns subdomain name.
// Most resources (such as secrets, clusters) require the names to be valid dns subdomain.
// This is a generic function (not specific to federation). Should be moved to a more generic location if others want to use it.
func GetValidDNSSubdomainName(name string) (string, error) {
	// "_" are not allowed. Replace them by "-".
	name = regexp.MustCompile("_").ReplaceAllLiteralString(name, "-")
	maxLength := validationutil.DNS1123SubdomainMaxLength
	if len(name) > maxLength {
		name = name[0 : maxLength-1]
	}
	// Verify that name now passes the validation.
	if errors := validation.NameIsDNSSubdomain(name, false); len(errors) != 0 {
		return "", fmt.Errorf("errors in converting name to a valid DNS subdomain %s", errors)
	}
	return name, nil
}
