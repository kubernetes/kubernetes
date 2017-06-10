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

package nodeidentifier

import (
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestDefaultNodeIdentifier_NodeIdentity(t *testing.T) {
	tests := []struct {
		name           string
		user           user.Info
		expectNodeName string
		expectIsNode   bool
	}{
		{
			name:           "nil user",
			user:           nil,
			expectNodeName: "",
			expectIsNode:   false,
		},
		{
			name:           "node username without group",
			user:           &user.DefaultInfo{Name: "system:node:foo"},
			expectNodeName: "",
			expectIsNode:   false,
		},
		{
			name:           "node group without username",
			user:           &user.DefaultInfo{Name: "foo", Groups: []string{"system:nodes"}},
			expectNodeName: "",
			expectIsNode:   false,
		},
		{
			name:           "node group and username",
			user:           &user.DefaultInfo{Name: "system:node:foo", Groups: []string{"system:nodes"}},
			expectNodeName: "foo",
			expectIsNode:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeName, isNode := NewDefaultNodeIdentifier().NodeIdentity(tt.user)
			if nodeName != tt.expectNodeName {
				t.Errorf("DefaultNodeIdentifier.NodeIdentity() got = %v, want %v", nodeName, tt.expectNodeName)
			}
			if isNode != tt.expectIsNode {
				t.Errorf("DefaultNodeIdentifier.NodeIdentity() got1 = %v, want %v", isNode, tt.expectIsNode)
			}
		})
	}
}
