//go:build !providerless
// +build !providerless

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

package gce

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestInstanceExists(t *testing.T) {
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	nodeNames := []string{"test-node-1"}
	_, err = createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	require.NoError(t, err)

	testcases := []struct {
		name        string
		nodeName    string
		exist       bool
		expectedErr error
	}{
		{
			name:        "node exist",
			nodeName:    "test-node-1",
			exist:       true,
			expectedErr: nil,
		},
		{
			name:        "node not exist",
			nodeName:    "test-node-2",
			exist:       false,
			expectedErr: nil,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}
			exist, err := gce.InstanceExists(context.TODO(), node)
			assert.Equal(t, test.expectedErr, err, test.name)
			assert.Equal(t, test.exist, exist, test.name)
		})
	}
}
