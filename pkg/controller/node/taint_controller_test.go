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

package node

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestComputeTaintDifference(t *testing.T) {
	testCases := []struct {
		lhs                []v1.Taint
		rhs                []v1.Taint
		expectedDifference []v1.Taint
		description        string
	}{
		{
			lhs: []v1.Taint{
				{
					Key:   "one",
					Value: "one",
				},
				{
					Key:   "two",
					Value: "two",
				},
			},
			rhs: []v1.Taint{
				{
					Key:   "one",
					Value: "one",
				},
				{
					Key:   "two",
					Value: "two",
				},
			},
			description: "Equal sets",
		},
    {
      lhs: []v1.Taint{
        {
          Key: "one",
          Value: "one",
        },
      },
      expectedDifference: []v1.Taint{
        {
          Key: "one",
          Value: "one",
        },
      },
      description: "Right is empty",
    },
    {
      rhs: []v1.Taint{
        {
          Key: "one",
          Value: "one",
        },
      },
      description: "Left is empty",
    },
    {
      lhs: []v1.Taint{
        {
          Key: "one",
          Value: "one",
        },
        {
          Key: "two",
          Value: "two",
        },
      },
      rhs: []v1.Taint{
        {
          Key: "two",
          Value: "two",
        },
        {
          Key: "three",
          Value: "three",
        },
      },
      expectedDifference: []v1.Taint{
        {
          Key: "one",
          Value: "one",
        },
      },
      description: "Intersecting arrays",
    },
	}

	for _, item := range testCases {
		difference := computeTaintDifference(item.lhs, item.rhs)
		if !api.Semantic.DeepEqual(difference, item.expectedDifference) {
			t.Errorf("%v: difference in not what expected. Got %v, expected %v", item.description, difference, item.expectedDifference)
		}
	}
}
