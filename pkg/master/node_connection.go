/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package master

import (
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/registry/node"
)

type nodeConnectionInfoGetter struct {
	nodeRegistry      node.Registry
	clusterInfoGetter client.ConnectionInfoGetter
}

func (n *nodeConnectionInfoGetter) GetConnectionInfo(ctx api.Context, nodeName string) (string, uint, http.RoundTripper, error) {
	scheme, port, transport, err := n.clusterInfoGetter.GetConnectionInfo(ctx, nodeName)
	if err != nil {
		return "", 0, nil, err
	}
	node, err := n.nodeRegistry.GetNode(ctx, nodeName)
	if err != nil {
		return "", 0, nil, err
	}
	if node.Status.DaemonEndpoints.KubeletEndpoint.Port > 0 {
		port = uint(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
	}
	return scheme, port, transport, nil
}
