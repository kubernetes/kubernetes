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

package metrics

import (
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"testing"
)

func TestVersionParsing(t *testing.T) {
	var tests = []struct {
		desc            string
		versionString   string
		expectedVersion string
	}{
		{
			"v1.15.0-alpha-1.12345",
			"v1.15.0-alpha-1.12345",
			"1.15.0",
		},
		{
			"Parse out defaulted string",
			"v0.0.0-master",
			"0.0.0",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			version := apimachineryversion.Info{
				GitVersion: test.versionString,
			}
			parsedV := parseVersion(version)
			if test.expectedVersion != parsedV.String() {
				t.Errorf("Got %v, wanted %v", parsedV.String(), test.expectedVersion)
			}
		})
	}
}
