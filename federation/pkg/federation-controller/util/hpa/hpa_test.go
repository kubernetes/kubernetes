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

package hpa

import (
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/require"
)

func TestGetHpaTargetClusterList(t *testing.T) {
	// Any object is fine for this test.
	obj := &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myhpa",
			Namespace: "myNamespace",
			SelfLink:  "/api/mylink",
		},
	}

	testCases := map[string]struct {
		clusterNames *ClusterNames
		expectedErr  bool
	}{
		"Wrong data set on annotations should return unmarshalling error when retrieving": {
			expectedErr: true,
		},
		"Get clusternames on annotations with 2 clusters, should have same names, which were set": {
			clusterNames: &ClusterNames{
				Names: []string{
					"c1",
					"c2",
				},
			},
		},
	}

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			accessor, _ := meta.Accessor(obj)
			anno := accessor.GetAnnotations()
			if anno == nil {
				anno = make(map[string]string)
				accessor.SetAnnotations(anno)
			}
			if testCase.expectedErr {
				anno[FederatedAnnotationOnHpaTargetObj] = "{" //some random string
			} else {
				anno[FederatedAnnotationOnHpaTargetObj] = testCase.clusterNames.String()
			}

			readNames, err := GetHpaTargetClusterList(obj)

			if testCase.expectedErr {
				require.Error(t, err, "An error was expected")
			} else {
				require.Equal(t, testCase.clusterNames, readNames, "Names should have been equal")
			}
		})
	}
}

func TestSetHpaTargetClusterList(t *testing.T) {
	// Any object is fine for this test.
	obj := &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myhpa",
			Namespace: "myNamespace",
			SelfLink:  "/api/mylink",
		},
	}

	testCases := map[string]struct {
		clusterNames ClusterNames
		expectedErr  bool
	}{
		"Get clusternames on annotations with 2 clusters, should have same names, which were set": {
			clusterNames: ClusterNames{
				Names: []string{
					"c1",
					"c2",
				},
			},
		},
	}

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {

			SetHpaTargetClusterList(obj, testCase.clusterNames)
			readNames, err := GetHpaTargetClusterList(obj)
			require.NoError(t, err, "An error should not have happened")
			require.Equal(t, &testCase.clusterNames, readNames, "Names should have been equal")

		})
	}
}
