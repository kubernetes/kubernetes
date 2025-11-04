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

package kubelet

import (
	"context"
	"strconv"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

// GetCachedNode returns the node object either from the cache or by making a synchronous
// call to the API server based on the useCache parameter.
func (kl *Kubelet) GetCachedNode(ctx context.Context, useCache bool) (*v1.Node, error) {
	if useCache {
		return kl.getCachedNode(ctx)
	}
	return kl.getNodeSync()
}

// getCachedNode compares the currently cached node to the node in the informer,
// and returns the newer one.
func (kl *Kubelet) getCachedNode(ctx context.Context) (*v1.Node, error) {
	logger := klog.FromContext(ctx)
	informerNode, err := kl.GetNode()
	if err != nil {
		if kl.cachedNode != nil {
			logger.Error(err, "failed to list node; using cached node")
			return kl.cachedNode, nil
		}
		return kl.initialNode(ctx)
	}

	if kl.cachedNode == nil {
		kl.cachedNode = informerNode
		return informerNode, nil
	}

	isNewer, err := isNewer(informerNode, kl.cachedNode)
	if err != nil {
		// In error cases, default to the informer node.
		logger.Error(err, "failed to check if node is newer; using informer node")
		kl.cachedNode = informerNode
	}
	if isNewer {
		kl.cachedNode = informerNode
	}
	return kl.cachedNode, nil
}

// getNodeSync forces a refresh of the cache by making a synchronous call to the API server
// for the most recent node info.
func (kl *Kubelet) getNodeSync() (*v1.Node, error) {
	if kl.kubeClient == nil {
		return kl.initialNode(context.Background())
	}
	node, err := kl.kubeClient.CoreV1().Nodes().Get(context.Background(), string(kl.nodeName), metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	kl.cachedNode = node
	return node, nil
}

// isNewer checks if the informer node is newer than the cached node based on the ResourceVersion.
func isNewer(informerNode, cachedNode *v1.Node) (bool, error) {
	if informerNode == cachedNode {
		return true, nil
	}
	informerVersion, err := getObjVersion(informerNode)
	if err != nil {
		return false, err
	}
	cachedVersion, err := getObjVersion(cachedNode)
	if err != nil {
		return false, err
	}
	return informerVersion > cachedVersion, nil
}

func getObjVersion(n *v1.Node) (int64, error) {
	objResourceVersion, err := strconv.ParseInt(n.ResourceVersion, 10, 64)
	if err != nil {
		return -1, err
	}
	return objResourceVersion, nil
}
