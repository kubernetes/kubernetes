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

package kubectl

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/printers"
)

func TestResourceFilter(t *testing.T) {
	tests := []struct {
		name   string
		hide   bool
		object runtime.Object
	}{
		{"v1.Pod pending", false, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodPending}}},
		{"v1.Pod running", false, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodRunning}}},
		{"v1.Pod succeeded", true, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodSucceeded}}},
		{"v1.Pod failed", true, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodFailed}}},
		{"v1.Pod evicted", true, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodFailed, Reason: "Evicted"}}},
		{"v1.Pod unknown", false, &v1.Pod{Status: v1.PodStatus{Phase: v1.PodUnknown}}},

		{"api.Pod pending", false, &api.Pod{Status: api.PodStatus{Phase: api.PodPending}}},
		{"api.Pod running", false, &api.Pod{Status: api.PodStatus{Phase: api.PodRunning}}},
		{"api.Pod succeeded", true, &api.Pod{Status: api.PodStatus{Phase: api.PodSucceeded}}},
		{"api.Pod failed", true, &api.Pod{Status: api.PodStatus{Phase: api.PodFailed}}},
		{"api.Pod evicted", true, &api.Pod{Status: api.PodStatus{Phase: api.PodFailed, Reason: "Evicted"}}},
		{"api.Pod unknown", false, &api.Pod{Status: api.PodStatus{Phase: api.PodUnknown}}},
	}

	filters := NewResourceFilter()

	options := &printers.PrintOptions{
		ShowAll: false,
	}
	for _, test := range tests {
		got, err := filters.Filter(test.object, options)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", test.name, err)
			continue
		}
		if want := test.hide; got != want {
			t.Errorf("%v: got %v, want %v", test.name, got, want)
		}
	}

	options.ShowAll = true
	for _, test := range tests {
		got, err := filters.Filter(test.object, options)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", test.name, err)
			continue
		}
		if want := false; got != want {
			t.Errorf("%v (ShowAll): got %v, want %v", test.name, got, want)
		}
	}
}
