/*
Copyright 2023 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/test/e2e/framework/internal/unittests/features"
)

func TestTagsEqual(t *testing.T) {
	testcases := []struct {
		a, b        interface{}
		expectEqual bool
	}{
		{1, 2, false},
		{2, 2, false},
		{WithSlow(), 2, false},
		{WithSlow(), WithSerial(), false},
		{WithSerial(), WithSlow(), false},
		{WithSlow(), WithSlow(), true},
		{WithSerial(), WithSerial(), true},
		{WithLabel("hello"), WithLabel("world"), false},
		{WithLabel("hello"), WithLabel("hello"), true},
		{WithFeatureGate(features.Test), WithLabel("Test"), false},
		{WithFeatureGate(features.Test), WithFeatureGate(features.Test), true},
	}

	for _, tc := range testcases {
		t.Run(fmt.Sprintf("%v=%v", tc.a, tc.b), func(t *testing.T) {
			actualEqual := TagsEqual(tc.a, tc.b)
			if actualEqual != tc.expectEqual {
				t.Errorf("expected %v, got %v", tc.expectEqual, actualEqual)
			}
		})
	}
}
