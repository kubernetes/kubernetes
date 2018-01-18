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

package util

import (
	"fmt"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/bootstrap/token/api"
	"regexp"
	"strings"
)

var bootstrapGroupRegexp = regexp.MustCompile(`\A` + api.BootstrapGroupPattern + `\z`)

// ValidateBootstrapGroupName checks if the provided group name is a valid
// bootstrap group name. Returns nil if valid or a validation error if invalid.
// TODO(mattmoyer): this validation should migrate out to client-go (see https://github.com/kubernetes/client-go/issues/114)
func ValidateBootstrapGroupName(name string) error {
	if bootstrapGroupRegexp.Match([]byte(name)) {
		return nil
	}
	return fmt.Errorf("bootstrap group %q is invalid (must match %s)", name, api.BootstrapGroupPattern)
}

// ValidateUsages validates that the passed in string are valid usage strings for bootstrap tokens.
func ValidateUsages(usages []string) error {
	validUsages := sets.NewString(api.KnownTokenUsages...)
	invalidUsages := sets.NewString()
	for _, usage := range usages {
		if !validUsages.Has(usage) {
			invalidUsages.Insert(usage)
		}
	}
	if len(invalidUsages) > 0 {
		return fmt.Errorf("invalide bootstrap token usage string: %s, valid usage options: %s", strings.Join(invalidUsages.List(), ","), strings.Join(api.KnownTokenUsages, ","))
	}
	return nil
}
