/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

// ValidatePolicy checks for errors in the Config
// It does not return early so that it can find as many errors as possible
func ValidatePolicy(policy schedulerapi.Policy) error {
	var validationErrors []error

	for _, priority := range policy.Priorities {
		if priority.Weight <= 0 || priority.Weight >= schedulerapi.MaxWeight {
			validationErrors = append(validationErrors, fmt.Errorf("Priority %s should have a positive weight applied to it or it has overflown", priority.Name))
		}
	}

	binders := 0
	for _, extender := range policy.ExtenderConfigs {
		if extender.Weight <= 0 {
			validationErrors = append(validationErrors, fmt.Errorf("Priority for extender %s should have a positive weight applied to it", extender.URLPrefix))
		}
		if extender.BindVerb != "" {
			binders++
		}
	}
	if binders > 1 {
		validationErrors = append(validationErrors, fmt.Errorf("Only one extender can implement bind, found %v", binders))
	}
	return utilerrors.NewAggregate(validationErrors)
}
