/*
Copyright 2025 The Kubernetes Authors.

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

package nodemanager

import (
	"net"

	v1 "k8s.io/api/core/v1"
	v1informers "k8s.io/client-go/informers/core/v1"
)

// Dummy is a dummy NodeManager. It doesn't do anything, but it can be used as
// a base implementation.
type Dummy struct{}

func (n *Dummy) PrimaryIPFamily() v1.IPFamily {
	return v1.IPFamilyUnknown
}

func (n *Dummy) NodeIPs() map[v1.IPFamily]net.IP {
	return nil
}

func (n *Dummy) PodCIDRs() []string {
	return nil
}

func (n *Dummy) Node() *v1.Node {
	return nil
}

func (n *Dummy) NodeInformer() v1informers.NodeInformer {
	return nil
}

// OnNodeChange is a handler for Node creation and update.
func (n *Dummy) OnNodeChange(node *v1.Node) {}

// OnNodeDelete is a handler for Node deletes.
func (n *Dummy) OnNodeDelete(node *v1.Node) {}

// OnNodeSynced is called after the cache is synced and all pre-existing Nodes have been reported
func (n *Dummy) OnNodeSynced() {}
