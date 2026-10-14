/*
Copyright The Kubernetes Authors.

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

package pod

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
)

// PodSpecTestcase holds a declarative validation testcase for podspec validation
type PodSpecTestcase struct {
	// ID of the testcase, so callers can filter or augment specific tests
	ID string
	// Tweaks to apply to the pod spec to make it invalid
	Tweaks []podtest.Tweak
	// ExpectedErrs are the expected declarative validation errors
	ExpectedErrs field.ErrorList
}

func InvalidPodSpecTestcases(fldPath *field.Path) []PodSpecTestcase {
	return []PodSpecTestcase{
		{
			ID:     "containers_required",
			Tweaks: []podtest.Tweak{podtest.SetContainers()},
			ExpectedErrs: field.ErrorList{
				field.Required(fldPath.Child("containers"), "").MarkAlpha(),
			},
		},
	}
}
