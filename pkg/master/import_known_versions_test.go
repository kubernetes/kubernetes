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

package master

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestGroupVersions(t *testing.T) {
	// legacyUnsuffixedGroups contains the groups released prior to deciding that kubernetes API groups should be dns-suffixed
	// new groups should be suffixed with ".k8s.io" (https://github.com/kubernetes/kubernetes/pull/31887#issuecomment-244462396)
	legacyUnsuffixedGroups := sets.NewString(
		"",
		"apps",
		"autoscaling",
		"batch",
		"componentconfig",
		"extensions",
		"federation",
		"policy",
	)

	// No new groups should be added to the legacyUnsuffixedGroups exclusion list
	if len(legacyUnsuffixedGroups) != 8 {
		t.Errorf("No additional unnamespaced groups should be created")
	}

	for _, gv := range registered.RegisteredGroupVersions() {
		if !strings.HasSuffix(gv.Group, ".k8s.io") && !legacyUnsuffixedGroups.Has(gv.Group) {
			t.Errorf("Group %s does not have the standard kubernetes API group suffix of .k8s.io", gv.Group)
		}
	}
}
