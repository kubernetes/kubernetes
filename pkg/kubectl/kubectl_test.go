/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/extensions"
)

func TestGroupExpander(t *testing.T) {
	tests := []struct {
		resource      string
		kind, version string
		errFn         func(err error) bool
	}{
		{
			resource: "",
			errFn:    func(err error) bool { return err != nil && strings.Contains(err.Error(), "no resource") },
		},
		{
			resource: "pod",
			kind:     "Pod",
			version:  "v1",
		},
		{
			resource: "pod.",
			kind:     "Pod",
			version:  "v1",
		},
		{
			resource: "job",
			kind:     "Job",
			version:  "extensions/v1beta1",
		},
		{
			resource: "job.extensions",
			kind:     "Job",
			version:  "extensions/v1beta1",
		},
		{
			resource: "job.ext",
			errFn: func(err error) bool {
				return err != nil && strings.Contains(err.Error(), `resource "job" is part of group "extensions", not "ext"`)
			},
		},
	}
	for i, test := range tests {
		expander := GroupExpander{api.RESTMapper}
		version, kind, err := expander.VersionAndKindForResource(test.resource)
		if test.errFn != nil {
			if !test.errFn(err) {
				t.Errorf("%d: unexpected error: %v", i, err)
				continue
			}
		} else if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if err != nil {
			continue
		}
		if test.kind != kind || test.version != version {
			t.Errorf("%d: unexpected kind/version %q %q for %s", i, kind, version, test.resource)
		}
	}
}
