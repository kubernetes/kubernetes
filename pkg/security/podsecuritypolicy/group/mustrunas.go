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

package group

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

// mustRunAs implements the GroupStrategy interface
type mustRunAs struct {
	ranges []extensions.IDRange
	field  string
}

var _ GroupStrategy = &mustRunAs{}

// NewMustRunAs provides a new MustRunAs strategy based on ranges.
func NewMustRunAs(ranges []extensions.IDRange, field string) (GroupStrategy, error) {
	if len(ranges) == 0 {
		return nil, fmt.Errorf("ranges must be supplied for MustRunAs")
	}
	return &mustRunAs{
		ranges: ranges,
		field:  field,
	}, nil
}

// Generate creates the group based on policy rules.  By default this returns the first group of the
// first range (min val).
func (s *mustRunAs) Generate(pod *api.Pod) ([]int64, error) {
	return []int64{s.ranges[0].Min}, nil
}

// Generate a single value to be applied.  This is used for FSGroup.  This strategy will return
// the first group of the first range (min val).
func (s *mustRunAs) GenerateSingle(pod *api.Pod) (*int64, error) {
	single := new(int64)
	*single = s.ranges[0].Min
	return single, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
// Groups are passed in here to allow this strategy to support multiple group fields (fsgroup and
// supplemental groups).
func (s *mustRunAs) Validate(pod *api.Pod, groups []int64) field.ErrorList {
	allErrs := field.ErrorList{}

	if pod.Spec.SecurityContext == nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("securityContext"), pod.Spec.SecurityContext, "unable to validate nil security context"))
		return allErrs
	}

	if len(groups) == 0 && len(s.ranges) > 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath(s.field), groups, "unable to validate empty groups against required ranges"))
	}

	for _, group := range groups {
		if !s.isGroupValid(group) {
			detail := fmt.Sprintf("%d is not an allowed group", group)
			allErrs = append(allErrs, field.Invalid(field.NewPath(s.field), groups, detail))
		}
	}

	return allErrs
}

func (s *mustRunAs) isGroupValid(group int64) bool {
	for _, rng := range s.ranges {
		if psputil.FallsInRange(group, rng) {
			return true
		}
	}
	return false
}
