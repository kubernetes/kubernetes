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
	"fmt"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/third_party/forked/gonum/graph"
	"k8s.io/kubernetes/third_party/forked/gonum/graph/traverse"
)

// NodeAuthorizer authorizes requests from kubelets, with the following logic:
// 1. If a request is not from a node (IdentifyNode() returns isNode=false), reject
// 2. If a specific node cannot be identified (IdentifyNode() returns nodeName=""), reject
// 3. If a request is for a secret, configmap, persistent volume or persistent volume claim, reject unless the verb is get, and the requested object is related to the requesting node:
//    node <- pod
//    node <- pod <- secret
//    node <- pod <- configmap
//    node <- pod <- pvc
//    node <- pod <- pvc <- pv
//    node <- pod <- pvc <- pv <- secret
// 4. For other resources, authorize all nodes uniformly using statically defined rules
type NodeAuthorizer struct {
	graph      *Graph
	identifier nodeidentifier.NodeIdentifier
	nodeRules  []rbacapi.PolicyRule
}

// NewAuthorizer returns a new node authorizer
func NewAuthorizer(graph *Graph, identifier nodeidentifier.NodeIdentifier, rules []rbacapi.PolicyRule) authorizer.Authorizer {
	return &NodeAuthorizer{
		graph:      graph,
		identifier: identifier,
		nodeRules:  rules,
	}
}

var (
	configMapResource = api.Resource("configmaps")
	secretResource    = api.Resource("secrets")
	pvcResource       = api.Resource("persistentvolumeclaims")
	pvResource        = api.Resource("persistentvolumes")
)

func (r *NodeAuthorizer) Authorize(attrs authorizer.Attributes) (bool, string, error) {
	nodeName, isNode := r.identifier.NodeIdentity(attrs.GetUser())
	if !isNode {
		// reject requests from non-nodes
		return false, "", nil
	}
	if len(nodeName) == 0 {
		// reject requests from unidentifiable nodes
		glog.V(2).Infof("NODE DENY: unknown node for user %q", attrs.GetUser().GetName())
		return false, fmt.Sprintf("unknown node for user %q", attrs.GetUser().GetName()), nil
	}

	// subdivide access to specific resources
	if attrs.IsResourceRequest() {
		requestResource := schema.GroupResource{Group: attrs.GetAPIGroup(), Resource: attrs.GetResource()}
		switch requestResource {
		case secretResource:
			return r.authorizeGet(nodeName, secretVertexType, attrs)
		case configMapResource:
			return r.authorizeGet(nodeName, configMapVertexType, attrs)
		case pvcResource:
			return r.authorizeGet(nodeName, pvcVertexType, attrs)
		case pvResource:
			return r.authorizeGet(nodeName, pvVertexType, attrs)
		}
	}

	// Access to other resources is not subdivided, so just evaluate against the statically defined node rules
	return rbac.RulesAllow(attrs, r.nodeRules...), "", nil
}

// authorizeGet authorizes "get" requests to objects of the specified type if they are related to the specified node
func (r *NodeAuthorizer) authorizeGet(nodeName string, startingType vertexType, attrs authorizer.Attributes) (bool, string, error) {
	if attrs.GetVerb() != "get" || len(attrs.GetName()) == 0 {
		glog.V(2).Infof("NODE DENY: %s %#v", nodeName, attrs)
		return false, "can only get individual resources of this type", nil
	}

	if len(attrs.GetSubresource()) > 0 {
		glog.V(2).Infof("NODE DENY: %s %#v", nodeName, attrs)
		return false, "cannot get subresource", nil
	}

	ok, err := r.hasPathFrom(nodeName, startingType, attrs.GetNamespace(), attrs.GetName())
	if err != nil {
		glog.V(2).Infof("NODE DENY: %v", err)
		return false, "no path found to object", nil
	}
	if !ok {
		glog.V(2).Infof("NODE DENY: %q %#v", nodeName, attrs)
		return false, "no path found to object", nil
	}
	return ok, "", nil
}

// hasPathFrom returns true if there is a directed path from the specified type/namespace/name to the specified Node
func (r *NodeAuthorizer) hasPathFrom(nodeName string, startingType vertexType, startingNamespace, startingName string) (bool, error) {
	r.graph.lock.RLock()
	defer r.graph.lock.RUnlock()

	nodeVertex, exists := r.graph.getVertex_rlocked(nodeVertexType, "", nodeName)
	if !exists {
		return false, fmt.Errorf("unknown node %q cannot get %s %s/%s", nodeName, vertexTypes[startingType], startingNamespace, startingName)
	}

	startingVertex, exists := r.graph.getVertex_rlocked(startingType, startingNamespace, startingName)
	if !exists {
		return false, fmt.Errorf("node %q cannot get unknown %s %s/%s", nodeName, vertexTypes[startingType], startingNamespace, startingName)
	}

	found := false
	traversal := &traverse.VisitingDepthFirst{
		EdgeFilter: func(edge graph.Edge) bool {
			if destinationEdge, ok := edge.(*destinationEdge); ok {
				if destinationEdge.DestinationID() != nodeVertex.ID() {
					// Don't follow edges leading to other nodes
					return false
				}
				// We found an edge leading to the node we want
				found = true
			}
			// Visit this edge
			return true
		},
	}
	traversal.Walk(r.graph.graph, startingVertex, func(n graph.Node) bool {
		if n.ID() == nodeVertex.ID() {
			// We found the node we want
			found = true
		}
		// Stop visiting if we've found the node we want
		return found
	})
	if !found {
		return false, fmt.Errorf("node %q cannot get %s %s/%s, no path was found", nodeName, vertexTypes[startingType], startingNamespace, startingName)
	}
	return true, nil
}
