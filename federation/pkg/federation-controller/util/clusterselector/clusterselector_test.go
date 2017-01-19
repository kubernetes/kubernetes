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

package clusterselector

import (
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"testing"
)

type testData struct {
	clusterLabels     map[string]string
	objectAnnotations map[string]string
	expectedResult    bool
	expectErr         bool
}

func TestSendToCluster(t *testing.T) {

	tests := make([]testData, 4)

	tests[0] = testData{
		clusterLabels: map[string]string{
			"location":    "europe",
			"environment": "prod",
		},
		objectAnnotations: map[string]string{
			federationapi.FederationClusterSelector: "[{\"key\": \"location\", \"operator\": \"in\", \"values\": [\"europe\"]}, {\"key\": \"environment\", \"operator\": \"==\", \"values\": [\"prod\"]}]",
		},
		expectedResult: true,
		expectErr:      false,
	}

	tests[1] = testData{
		clusterLabels: map[string]string{
			"location":    "europe",
			"environment": "test",
		},
		objectAnnotations: map[string]string{
			federationapi.FederationClusterSelector: "[{\"key\": \"location\", \"operator\": \"in\", \"values\": [\"europe\"]}, {\"key\": \"environment\", \"operator\": \"==\", \"values\": [\"prod\"]}]",
		},
		expectedResult: false,
		expectErr:      false,
	}

	tests[2] = testData{
		clusterLabels: map[string]string{
			"location":    "europe",
			"environment": "test",
		},
		objectAnnotations: map[string]string{
			federationapi.FederationClusterSelector: "[{\"key\": \"location\", \"operator\": \"!=\", \"values\": [\"usa\"]}, {\"key\": \"environment\", \"operator\": \"==\", \"values\": [\"test\"]}]",
		},
		expectedResult: true,
		expectErr:      false,
	}

	tests[3] = testData{
		clusterLabels: map[string]string{
			"location":    "europe",
			"environment": "test",
		},
		objectAnnotations: map[string]string{
			federationapi.FederationClusterSelector: "[{\"not able to parse\",}]",
		},
		expectedResult: false,
		expectErr:      true,
	}

	for num, test := range tests {
		result, err := SendToCluster(test.clusterLabels, test.objectAnnotations)
		if err != nil && !test.expectErr {
			t.Errorf("SendToCluster test %v: threw error when not expected: %s", num, err.Error())
		}
		if result != test.expectedResult {
			t.Errorf("SendToCluster test %v values: %+v: expected %v, result %v", num, test, test.expectedResult, result)
		}
	}
}
