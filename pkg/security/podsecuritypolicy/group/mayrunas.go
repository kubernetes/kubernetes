/*
Copyright 2018 The Kubernetes Authors.

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

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// mayRunAs implements the GroupStrategy interface.
type mayRunAs struct {
	ranges []policy.IDRange
}

var _ GroupStrategy = &mayRunAs{}

// NewMayRunAs provides a new MayRunAs strategy.
func NewMayRunAs(ranges []policy.IDRange) (GroupStrategy, error) {
	if len(ranges) == 0 {
		return nil, fmt.Errorf("ranges must be supplied for MayRunAs")
	}
	return &mayRunAs{
		ranges: ranges,
	}, nil
}

// Generate creates the group based on policy rules.  This strategy returns an empty slice.
func (s *mayRunAs) Generate(_ *api.Pod) ([]int64, error) {
	return nil, nil
}

// Generate a single value to be applied.  This is used for FSGroup.  This strategy returns nil.
func (s *mayRunAs) GenerateSingle(_ *api.Pod) (*int64, error) {
	return nil, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
// Groups are passed in here to allow this strategy to support multiple group fields (fsgroup and
// supplemental groups).
func (s *mayRunAs) Validate(fldPath *field.Path, _ *api.Pod, groups []int64) field.ErrorList {
	return ValidateGroupsInRanges(fldPath, s.ranges, groups)
}
