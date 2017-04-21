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

package util

import (
	"math"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
)

func int64P(num int64) *int64 {
	return &num
}

func TestGetPodTemplateSpecHash(t *testing.T) {
	tests := []struct {
		name            string
		template        *v1.PodTemplateSpec
		uniquifier      *int64
		otherUniquifier *int64
	}{
		{
			name:            "simple",
			template:        &v1.PodTemplateSpec{},
			uniquifier:      int64P(1),
			otherUniquifier: int64P(2),
		},
		{
			name:            "using math.MaxInt64",
			template:        &v1.PodTemplateSpec{},
			uniquifier:      nil,
			otherUniquifier: int64P(int64(math.MaxInt64)),
		},
	}

	for _, test := range tests {
		hash := GetPodTemplateSpecHash(test.template, test.uniquifier)
		otherHash := GetPodTemplateSpecHash(test.template, test.otherUniquifier)

		if hash == otherHash {
			t.Errorf("expected different hashes but got the same: %d", hash)
		}
	}
}
