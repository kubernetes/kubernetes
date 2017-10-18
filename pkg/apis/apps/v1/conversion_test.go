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

package v1_test

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/extensions"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

func TestV1RollingUpdateDaemonSetConversion(t *testing.T) {
	intorstr := intstr.FromInt(1)
	testcases := map[string]struct {
		rollingUpdateDs1 *extensions.RollingUpdateDaemonSet
		rollingUpdateDs2 *appsv1.RollingUpdateDaemonSet
	}{
		"RollingUpdateDaemonSet Conversion 2": {
			rollingUpdateDs1: &extensions.RollingUpdateDaemonSet{MaxUnavailable: intorstr},
			rollingUpdateDs2: &appsv1.RollingUpdateDaemonSet{MaxUnavailable: &intorstr},
		},
	}

	for k, tc := range testcases {
		// extensions -> v1
		internal1 := &appsv1.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDs2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1", tc.rollingUpdateDs2, internal1)
		}

		// v1 -> extensions
		internal2 := &extensions.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDs1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1 to extensions", tc.rollingUpdateDs1, internal2)
		}
	}
}
