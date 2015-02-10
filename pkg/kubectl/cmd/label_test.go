/*
Copyright 2014 Google Inc. All rights reserved.

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

package cmd

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestValidateLabels(t *testing.T) {
	tests := []struct {
		meta      *api.ObjectMeta
		labels    map[string]string
		expectErr bool
		test      string
	}{
		{
			meta: &api.ObjectMeta{
				Labels: map[string]string{
					"a": "b",
					"c": "d",
				},
			},
			labels: map[string]string{
				"a": "c",
				"d": "b",
			},
			test:      "one shared",
			expectErr: true,
		},
		{
			meta: &api.ObjectMeta{
				Labels: map[string]string{
					"a": "b",
					"c": "d",
				},
			},
			labels: map[string]string{
				"b": "d",
				"c": "a",
			},
			test:      "second shared",
			expectErr: true,
		},
		{
			meta: &api.ObjectMeta{
				Labels: map[string]string{
					"a": "b",
					"c": "d",
				},
			},
			labels: map[string]string{
				"b": "a",
				"d": "c",
			},
			test: "no overlap",
		},
		{
			meta: &api.ObjectMeta{},
			labels: map[string]string{
				"b": "a",
				"d": "c",
			},
			test: "no labels",
		},
	}
	for _, test := range tests {
		err := validateNoOverwrites(test.meta, test.labels)
		if test.expectErr && err == nil {
			t.Errorf("%s: unexpected non-error", test.test)
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
	}
}

func TestParseLabels(t *testing.T) {
	tests := []struct {
		labels         []string
		expected       map[string]string
		expectedRemove []string
		expectErr      bool
	}{
		{
			labels:   []string{"a=b", "c=d"},
			expected: map[string]string{"a": "b", "c": "d"},
		},
		{
			labels:   []string{},
			expected: map[string]string{},
		},
		{
			labels:         []string{"a=b", "c=d", "e-"},
			expected:       map[string]string{"a": "b", "c": "d"},
			expectedRemove: []string{"e"},
		},
		{
			labels:    []string{"ab", "c=d"},
			expectErr: true,
		},
		{
			labels:    []string{"a=b", "c=d", "a-"},
			expectErr: true,
		},
	}
	for _, test := range tests {
		labels, remove, err := parseLabels(test.labels)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error: %v", test)
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v %v", err, test)
		}
		if !reflect.DeepEqual(labels, test.expected) {
			t.Errorf("expected: %v, got %v", test.expected, labels)
		}
		if !reflect.DeepEqual(remove, test.expectedRemove) {
			t.Errorf("expected: %v, got %v", test.expectedRemove, remove)
		}
	}
}

func TestLabelFunc(t *testing.T) {
	tests := []struct {
		obj       runtime.Object
		overwrite bool
		version   string
		labels    map[string]string
		remove    []string
		expected  runtime.Object
		expectErr bool
	}{
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
			},
			labels:    map[string]string{"a": "b"},
			expectErr: true,
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
			},
			labels:    map[string]string{"a": "c"},
			overwrite: true,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "c"},
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
			},
			labels: map[string]string{"c": "d"},
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b", "c": "d"},
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
			},
			labels:  map[string]string{"c": "d"},
			version: "2",
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels:          map[string]string{"a": "b", "c": "d"},
					ResourceVersion: "2",
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
			},
			labels: map[string]string{},
			remove: []string{"a"},
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{},
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b", "c": "d"},
				},
			},
			labels: map[string]string{"e": "f"},
			remove: []string{"a"},
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"c": "d",
						"e": "f",
					},
				},
			},
		},
	}
	for _, test := range tests {
		out, err := labelFunc(test.obj, test.overwrite, test.version, test.labels, test.remove)
		if test.expectErr {
			if err == nil {
				t.Errorf("unexpected non-error: %v", test)
			}
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v %v", err, test)
		}
		if !reflect.DeepEqual(out, test.expected) {
			t.Errorf("expected: %v, got %v", test.expected, out)
		}
	}
}
