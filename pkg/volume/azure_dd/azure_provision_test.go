// +build !providerless

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

package azure_dd

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
)

func TestParseZoned(t *testing.T) {
	tests := []struct {
		msg         string
		zoneString  string
		diskKind    v1.AzureDataDiskKind
		expected    bool
		expectError bool
	}{
		{
			msg:      "managed disk should default to zoned",
			diskKind: v1.AzureManagedDisk,
			expected: true,
		},
		{
			msg:      "shared blob disk should default to un-zoned",
			diskKind: v1.AzureSharedBlobDisk,
			expected: false,
		},
		{
			msg:      "shared dedicated disk should default to un-zoned",
			diskKind: v1.AzureDedicatedBlobDisk,
			expected: false,
		},
		{
			msg:        "managed disk should support zoned=true",
			diskKind:   v1.AzureManagedDisk,
			zoneString: "true",
			expected:   true,
		},
		{
			msg:        "managed disk should support zoned=false",
			diskKind:   v1.AzureManagedDisk,
			zoneString: "false",
			expected:   false,
		},
		{
			msg:        "shared blob disk should support zoned=false",
			diskKind:   v1.AzureSharedBlobDisk,
			zoneString: "false",
			expected:   false,
		},
		{
			msg:         "shared blob disk shouldn't support zoned=true",
			diskKind:    v1.AzureSharedBlobDisk,
			zoneString:  "true",
			expectError: true,
		},
		{
			msg:        "shared dedicated disk should support zoned=false",
			diskKind:   v1.AzureDedicatedBlobDisk,
			zoneString: "false",
			expected:   false,
		},
		{
			msg:         "dedicated blob disk shouldn't support zoned=true",
			diskKind:    v1.AzureDedicatedBlobDisk,
			zoneString:  "true",
			expectError: true,
		},
	}

	for i, test := range tests {
		real, err := parseZoned(test.zoneString, test.diskKind)
		if test.expectError {
			assert.Error(t, err, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		} else {
			assert.Equal(t, test.expected, real, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		}
	}
}

func TestConvertTagsToMap(t *testing.T) {
	testCases := []struct {
		desc           string
		tags           string
		expectedOutput map[string]string
		expectedError  bool
	}{
		{
			desc:           "should return empty map when tag is empty",
			tags:           "",
			expectedOutput: map[string]string{},
			expectedError:  false,
		},
		{
			desc: "sing valid tag should be converted",
			tags: "key=value",
			expectedOutput: map[string]string{
				"key": "value",
			},
			expectedError: false,
		},
		{
			desc: "multiple valid tags should be converted",
			tags: "key1=value1,key2=value2",
			expectedOutput: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
			expectedError: false,
		},
		{
			desc: "whitespaces should be trimmed",
			tags: "key1=value1, key2=value2",
			expectedOutput: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
			expectedError: false,
		},
		{
			desc:           "should return error for invalid format",
			tags:           "foo,bar",
			expectedOutput: nil,
			expectedError:  true,
		},
		{
			desc:           "should return error for when key is missed",
			tags:           "key1=value1,=bar",
			expectedOutput: nil,
			expectedError:  true,
		},
	}

	for i, c := range testCases {
		m, err := ConvertTagsToMap(c.tags)
		if c.expectedError {
			assert.NotNil(t, err, "TestCase[%d]: %s", i, c.desc)
		} else {
			assert.Nil(t, err, "TestCase[%d]: %s", i, c.desc)
			if !reflect.DeepEqual(m, c.expectedOutput) {
				t.Errorf("got: %v, expected: %v, desc: %v", m, c.expectedOutput, c.desc)
			}
		}
	}
}
