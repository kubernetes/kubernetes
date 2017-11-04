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
	"k8s.io/apiserver/pkg/authentication/user"
)

// NodeIdentifier determines node information from a given user
type NodeIdentifier interface {
	// NodeIdentity determines node information from the given user.Info.
	// nodeName is the name of the Node API object associated with the user.Info,
	// and may be empty if a specific node cannot be determined.
	// isNode is true if the user.Info represents an identity issued to a node.
	NodeIdentity(user.Info) (nodeName string, isNode bool)
}
