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
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestInstanceExists(t *testing.T) {
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	if err != nil {
		t.Fatal(err)
	}
	nodeNames := []string{"test-node-1"}
	_, err = createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	if err != nil {
		t.Fatal(err)
	}

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
			expectedErr: fmt.Errorf("failed to get instance ID from cloud provider: instance not found"),
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}
			exist, err := gce.InstanceExists(context.TODO(), node)
			if !reflect.DeepEqual(test.expectedErr, err) {
				t.Errorf("TestName(%s): want: %s, got: %s", test.name, test.expectedErr, err)
			}
			if test.exist != exist {
				t.Errorf("TestName(%s): want: %t, got: %t", test.name, test.exist, exist)
			}
		})
	}
}
