/*
Copyright 2021 The Kubernetes Authors.

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

package version

import (
	"testing"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestPrintVersionSkewWarning(t *testing.T) {
	testCases := []struct {
		name              string
		clientVersion     apimachineryversion.Info
		serverVersion     apimachineryversion.Info
		isWarningExpected bool
	}{
		{
			name:              "Should not warn if server and client versions are same",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			isWarningExpected: false,
		},
		{
			name:              "Should not warn if server and client versions are same and server is alpha",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.19.7-alpha"},
			isWarningExpected: false,
		},
		{
			name:              "Should not warn if server and client versions are same and server is beta",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.19.7-beta"},
			isWarningExpected: false,
		},
		{
			name:              "Should not warn if server is 1 minor version ahead of client",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.18.5"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			isWarningExpected: false,
		},
		{
			name:              "Should not warn if server is 1 minor version behind client",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.18.5"},
			isWarningExpected: false,
		},
		{
			name:              "Should warn if server is 2 minor versions ahead of client",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.17.7"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			isWarningExpected: true,
		},
		{
			name:              "Should warn if server is 2 minor versions behind client",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v1.17.7"},
			isWarningExpected: true,
		},
		{
			name:              "Should warn if major versions are not equal",
			clientVersion:     apimachineryversion.Info{GitVersion: "v1.19.1"},
			serverVersion:     apimachineryversion.Info{GitVersion: "v2.19.1"},
			isWarningExpected: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			warningMessage, err := getVersionSkewWarning(tc.clientVersion, tc.serverVersion)
			if err != nil {
				t.Errorf("error: %s", err)
			}

			if tc.isWarningExpected && warningMessage == "" {
				t.Error("warning was expected")
			} else if !tc.isWarningExpected && warningMessage != "" {
				t.Errorf("warning was not expected. but got %s", warningMessage)
			}
		})
	}
}
