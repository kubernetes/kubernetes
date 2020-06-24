/*
Copyright 2020 The Kubernetes Authors.

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

package util

import (
	"bytes"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	fakediscovery "k8s.io/client-go/discovery/fake"
	coretesting "k8s.io/client-go/testing"
	"os"
	"testing"
)

type fakeCachedDiscoveryClient struct {
	fakediscovery.FakeDiscovery
}

func (svp *fakeCachedDiscoveryClient) Fresh() bool {
	return true
}

func (svp *fakeCachedDiscoveryClient) Invalidate() {
}

func TestVersionSkewWarning(t *testing.T) {
	discoveryClient := &fakeCachedDiscoveryClient{}
	discoveryClient.Fake = &coretesting.Fake{}
	output := &bytes.Buffer{}
	mvf := &MatchVersionFlags{
		Delegate:                 genericclioptions.NewTestConfigFlags().WithDiscoveryClient(discoveryClient),
		versionSkewWarningWriter: output,
	}

	testCases := []struct {
		name               string
		clientMajorVersion string
		clientMinorVersion string
		serverMajorVersion string
		serverMinorVersion string
		isWarningExpected  bool
	}{
		{
			name:               "Should not warn if server and client versions are same",
			clientMajorVersion: "1",
			clientMinorVersion: "19",
			serverMajorVersion: "1",
			serverMinorVersion: "19",
			isWarningExpected:  false,
		},
		{
			name:               "Should not warn if server and client versions are same except server minor has a plus sign",
			clientMajorVersion: "1",
			clientMinorVersion: "19",
			serverMajorVersion: "1",
			serverMinorVersion: "19+",
			isWarningExpected:  false,
		},
		{
			name:               "Should not warn if server is 1 minor version ahead of client",
			clientMajorVersion: "1",
			clientMinorVersion: "18",
			serverMajorVersion: "1",
			serverMinorVersion: "19",
			isWarningExpected:  false,
		},
		{
			name:               "Should not warn if server is 1 minor version behind client",
			clientMajorVersion: "1",
			clientMinorVersion: "19",
			serverMajorVersion: "1",
			serverMinorVersion: "18",
			isWarningExpected:  false,
		},
		{
			name:               "Should warn if server is 2 minor versions ahead of client",
			clientMajorVersion: "1",
			clientMinorVersion: "17",
			serverMajorVersion: "1",
			serverMinorVersion: "19",
			isWarningExpected:  true,
		},
		{
			name:               "Should warn if server is 2 minor versions behind client",
			clientMajorVersion: "1",
			clientMinorVersion: "19",
			serverMajorVersion: "1",
			serverMinorVersion: "17",
			isWarningExpected:  true,
		},
		{
			name:               "Should warn if major versions are not equal",
			clientMajorVersion: "1",
			clientMinorVersion: "0",
			serverMajorVersion: "2",
			serverMinorVersion: "0",
			isWarningExpected:  true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output.Reset()

			mvf.clientVersion = apimachineryversion.Info{Major: tc.clientMajorVersion, Minor: tc.clientMinorVersion}
			discoveryClient.FakedServerVersion = &apimachineryversion.Info{Major: tc.serverMajorVersion, Minor: tc.serverMinorVersion}

			mvf.warnIfUnsupportedVersionSkew()

			if tc.isWarningExpected && output.Len() == 0 {
				t.Error("warning was expected, but not written to the output")
			} else if !tc.isWarningExpected && output.Len() > 0 {
				t.Errorf("warning was not expected, but was written to the output: %s", output.String())
			}
		})
	}
}

func TestVersionSkewWarningSuppression(t *testing.T) {
	discoveryClient := &fakeCachedDiscoveryClient{}
	discoveryClient.Fake = &coretesting.Fake{}
	output := &bytes.Buffer{}
	mvf := &MatchVersionFlags{
		Delegate:                 genericclioptions.NewTestConfigFlags().WithDiscoveryClient(discoveryClient),
		versionSkewWarningWriter: output,
	}

	if err := os.Setenv(suppressVersionSkewWarningEnvironmentVariable, "true"); err != nil {
		t.Fatalf(err.Error())
	}
	defer os.Unsetenv(suppressVersionSkewWarningEnvironmentVariable)

	mvf.clientVersion = apimachineryversion.Info{Major: "1", Minor: "19"}
	discoveryClient.FakedServerVersion = &apimachineryversion.Info{Major: "1", Minor: "17"}

	mvf.warnIfUnsupportedVersionSkew()
	if output.Len() > 0 {
		t.Errorf("warning was expected to be suppressed, but was written to the output: %s", output.String())
	}
}
