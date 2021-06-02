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

package cmd

import (
	"bytes"
	"encoding/json"
	"github.com/pkg/errors"
	"sigs.k8s.io/yaml"
	"testing"
)

func TestNewCmdVersion(t *testing.T) {
	var buf bytes.Buffer
	cmd := newCmdVersion(&buf)
	if err := cmd.Execute(); err != nil {
		t.Errorf("Cannot execute version command: %v", err)
	}
}

func TestRunVersion(t *testing.T) {
	var buf bytes.Buffer
	iface := make(map[string]interface{})
	flagNameOutput := "output"
	cmd := newCmdVersion(&buf)

	testCases := []struct {
		name              string
		flag              string
		expectedError     bool
		shouldBeValidYAML bool
		shouldBeValidJSON bool
	}{
		{
			name: "valid: run without flags",
		},
		{
			name: "valid: run with flag 'short'",
			flag: "short",
		},
		{
			name:              "valid: run with flag 'yaml'",
			flag:              "yaml",
			shouldBeValidYAML: true,
		},
		{
			name:              "valid: run with flag 'json'",
			flag:              "json",
			shouldBeValidJSON: true,
		},
		{
			name:          "invalid: run with unsupported flag",
			flag:          "unsupported-flag",
			expectedError: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			if len(tc.flag) > 0 {
				if err = cmd.Flags().Set(flagNameOutput, tc.flag); err != nil {
					goto error
				}
			}
			buf.Reset()
			if err = RunVersion(&buf, cmd); err != nil {
				goto error
			}
			if buf.String() == "" {
				err = errors.New("empty output")
				goto error
			}
			if tc.shouldBeValidYAML {
				err = yaml.Unmarshal(buf.Bytes(), &iface)
			} else if tc.shouldBeValidJSON {
				err = json.Unmarshal(buf.Bytes(), &iface)
			}
		error:
			if (err != nil) != tc.expectedError {
				t.Errorf("Test case %q: RunVersion expected error: %v, saw: %v; %v", tc.name, tc.expectedError, err != nil, err)
			}
		})
	}
}
