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
	"testing"

	"github.com/stretchr/testify/require"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
)

func TestSendToCluster(t *testing.T) {

	clusterLabels := map[string]string{
		"location":    "europe",
		"environment": "prod",
		"version":     "15",
	}

	testCases := map[string]struct {
		objectAnnotations map[string]string
		expectedResult    bool
		expectedErr       bool
	}{
		"match with single annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "location", "operator": "in", "values": ["europe"]}]`,
			},
			expectedResult: true,
		},
		"match on multiple annotations": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "location", "operator": "in", "values": ["europe"]}, {"key": "environment", "operator": "==", "values": ["prod"]}]`,
			},
			expectedResult: true,
		},
		"mismatch on one annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "location", "operator": "in", "values": ["europe"]}, {"key": "environment", "operator": "==", "values": ["test"]}]`,
			},
			expectedResult: false,
		},
		"match on not equal annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "location", "operator": "!=", "values": ["usa"]}, {"key": "environment", "operator": "in", "values": ["prod"]}]`,
			},
			expectedResult: true,
		},
		"match on greater than annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "version", "operator": ">", "values": ["14"]}]`,
			},
			expectedResult: true,
		},
		"mismatch on greater than annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"key": "version", "operator": ">", "values": ["15"]}]`,
			},
			expectedResult: false,
		},
		"unable to parse annotation": {
			objectAnnotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: `[{"not able to parse",}]`,
			},
			expectedResult: false,
			expectedErr:    true,
		},
	}

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			result, err := SendToCluster(clusterLabels, testCase.objectAnnotations)

			if testCase.expectedErr {
				require.Error(t, err, "An error was expected")
			} else {
				require.NoError(t, err, "An error was not expected")
			}

			require.Equal(t, testCase.expectedResult, result, "Unexpected response from SendToCluster")
		})
	}
}
