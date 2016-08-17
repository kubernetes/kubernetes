/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"testing"
)

func TestValidateArgs(t *testing.T) {
	f, _, _, _ := NewAPIFactory()

	tests := []struct {
		flags     map[string]string
		filenames []string
		args      []string
		expectErr bool
		testName  string
	}{
		{
			expectErr: true,
			testName:  "nothing",
		},
		{
			flags:     map[string]string{},
			args:      []string{"foo"},
			expectErr: true,
			testName:  "no file, no image",
		},
		{
			filenames: []string{"bar.yaml"},
			args:      []string{"foo"},
			testName:  "valid file example",
		},
		{
			flags: map[string]string{
				"image": "foo:v2",
			},
			args:     []string{"foo"},
			testName: "missing second image name",
		},
		{
			flags: map[string]string{
				"image": "foo:v2",
			},
			args:     []string{"foo", "foo-v2"},
			testName: "valid image example",
		},
		{
			flags: map[string]string{
				"image": "foo:v2",
			},
			filenames: []string{"bar.yaml"},
			args:      []string{"foo", "foo-v2"},
			expectErr: true,
			testName:  "both filename and image example",
		},
	}
	for _, test := range tests {
		out := &bytes.Buffer{}
		cmd := NewCmdRollingUpdate(f, out)

		if test.flags != nil {
			for key, val := range test.flags {
				cmd.Flags().Set(key, val)
			}
		}
		err := validateArguments(cmd, test.filenames, test.args)
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v (%s)", err, test.testName)
		}
		if err == nil && test.expectErr {
			t.Errorf("unexpected non-error (%s)", test.testName)
		}
	}
}
