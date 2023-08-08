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

package cdi

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/kubelet/container"
)

func TestGenerateAnnotations(t *testing.T) {
	testCases := []struct {
		description         string
		deviceIDs           []string
		expecteError        error
		expectedAnnotations []container.Annotation
	}{
		{
			description: "no devices",
			deviceIDs:   []string{},
		},
		{
			description:         "one device",
			deviceIDs:           []string{"vendor.com/class=device1"},
			expectedAnnotations: []container.Annotation{{Name: "cdi.k8s.io/test-driver-name_test-claim-uid", Value: "vendor.com/class=device1"}},
		},
	}

	as := assert.New(t)
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			annotations, err := GenerateAnnotations("test-claim-uid", "test-driver-name", tc.deviceIDs)
			as.ErrorIs(err, tc.expecteError)
			as.Equal(tc.expectedAnnotations, annotations)
		})
	}
}
