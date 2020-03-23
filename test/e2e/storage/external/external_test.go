/*
Copyright 2019 The Kubernetes Authors.

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

package external

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
)

func TestDriverParameter(t *testing.T) {
	expected := &driverDefinition{
		DriverInfo: testsuites.DriverInfo{
			Name: "foo.example.com",
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
		},
		SupportedSizeRange: volume.SizeRange{
			Min: "5Gi",
		},
	}
	testcases := []struct {
		name     string
		filename string
		err      string
		expected *driverDefinition
	}{
		{
			name:     "no such file",
			filename: "no-such-file.yaml",
			err:      "open no-such-file.yaml: no such file or directory",
		},
		{
			name: "empty file name",
			err:  "missing file name",
		},
		{
			name:     "yaml",
			filename: "testdata/driver.yaml",
			expected: expected,
		},
		{
			name:     "json",
			filename: "testdata/driver.json",
			expected: expected,
		},
	}

	for _, testcase := range testcases {
		actual, err := loadDriverDefinition(testcase.filename)
		if testcase.err == "" {
			assert.NoError(t, err, testcase.name)
		} else {
			if assert.Error(t, err, testcase.name) {
				assert.Equal(t, testcase.err, err.Error())
			}
		}
		if err == nil {
			assert.Equal(t, testcase.expected, actual)
		}
	}
}
