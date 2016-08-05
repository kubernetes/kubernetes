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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// GroupStrategy defines the interface for all group constraint strategies.
type GroupStrategy interface {
	// Generate creates the group based on policy rules.  The underlying implementation can
	// decide whether it will return a full range of values or a subset of values from the
	// configured ranges.
	Generate(pod *api.Pod) ([]int64, error)
	// Generate a single value to be applied.  The underlying implementation decides which
	// value to return if configured with multiple ranges.  This is used for FSGroup.
	GenerateSingle(pod *api.Pod) (*int64, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(pod *api.Pod, groups []int64) field.ErrorList
}
