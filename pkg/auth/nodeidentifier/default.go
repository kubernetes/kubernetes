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
	"strings"

	"k8s.io/apiserver/pkg/authentication/user"
)

// NewDefaultNodeIdentifier returns a default NodeIdentifier implementation,
// which returns isNode=true if the user groups contain the system:nodes group
// and the user name matches the format system:node:<nodeName>, and populates
// nodeName if isNode is true
func NewDefaultNodeIdentifier() NodeIdentifier {
	return defaultNodeIdentifier{}
}

// defaultNodeIdentifier implements NodeIdentifier
type defaultNodeIdentifier struct{}

// nodeUserNamePrefix is the prefix for usernames in the form `system:node:<nodeName>`
const nodeUserNamePrefix = "system:node:"

// NodeIdentity returns isNode=true if the user groups contain the system:nodes
// group and the user name matches the format system:node:<nodeName>, and
// populates nodeName if isNode is true
func (defaultNodeIdentifier) NodeIdentity(u user.Info) (string, bool) {
	// Make sure we're a node, and can parse the node name
	if u == nil {
		return "", false
	}

	userName := u.GetName()
	if !strings.HasPrefix(userName, nodeUserNamePrefix) {
		return "", false
	}

	nodeName := strings.TrimPrefix(userName, nodeUserNamePrefix)

	isNode := false
	for _, g := range u.GetGroups() {
		if g == user.NodesGroup {
			isNode = true
			break
		}
	}
	if !isNode {
		return "", false
	}

	return nodeName, isNode
}
