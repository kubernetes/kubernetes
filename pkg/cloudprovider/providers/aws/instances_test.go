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

package aws

import (
	"testing"
)

func TestParseInstance(t *testing.T) {
	tests := []struct {
		Kubernetes  kubernetesInstanceID
		Aws         awsInstanceID
		ExpectError bool
	}{
		{
			Kubernetes: "aws:///us-east-1a/i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "aws:////i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "aws:///us-east-1a/i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes: "aws:////i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes: "i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes:  "vol-123456789",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws:///us-east-1a/vol-12345678abcdef01",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws://accountid/us-east-1a/vol-12345678abcdef01",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws:///us-east-1a/vol-12345678abcdef01/suffix",
			ExpectError: true,
		},
		{
			Kubernetes:  "",
			ExpectError: true,
		},
	}

	for _, test := range tests {
		awsID, err := test.Kubernetes.mapToAWSInstanceID()
		if err != nil {
			if !test.ExpectError {
				t.Errorf("unexpected error parsing %s: %v", test.Kubernetes, err)
			}
		} else {
			if test.ExpectError {
				t.Errorf("expected error parsing %s", test.Kubernetes)
			} else if test.Aws != awsID {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsID)
			}
		}
	}
}
