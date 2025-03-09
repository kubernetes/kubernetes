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

package peerproxy

import (
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/discovery"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	responsewriterutil "k8s.io/apiserver/pkg/util/responsewriter"
)

func (h *peerProxyHandler) addPeerDiscoveryInfo(obj interface{}) {
	serverIdentityLease, ok := obj.(*v1.Lease)
	if !ok {
		klog.ErrorS(fmt.Errorf("invalid lease object provided to addPeerDiscoveryInfo(), received type: %T", obj), "error adding peer served resources", )
		return
	}

	err := h.addPeerDiscoveryInfoToCache(serverIdentityLease)
	if err != nil {
		klog.ErrorS(err, "error adding peer served resources", "serverID", serverIdentityLease)
	}
}

func (h *peerProxyHandler) deletePeerDiscoveryInfo(obj interface{}) {
	h.peerAggDiscoveryCacheLock.Lock()
	defer h.peerAggDiscoveryCacheLock.Unlock()
	serverIdentityLease, ok := obj.(*v1.Lease)
	if !ok {
		klog.ErrorS(fmt.Errorf("invalid lease object provided to addPeerServedResources()"), "error deleting peer served resources")
		return
	}

	h.peerAggDiscoveryCacheLock.Lock()
	delete(h.peerAggDiscoveryResponseCache, serverIdentityLease.Name)
	h.peerAggDiscoveryCacheLock.Unlock()
}

func (h *peerProxyHandler) updatePeerDiscoveryInfo(oldObj interface{}, newObj interface{}) {
	oldLease, ok := oldObj.(*v1.Lease)
	if !ok {
		klog.Error("invalid lease object provided to updatePeerServedResources()")
		return
	}

	newLease, ok := newObj.(*v1.Lease)
	if !ok {
		klog.Error("invalid lease object provided to updatePeerServedResources()")
		return
	}

	switch {
	case newLease == nil && oldLease == nil:
		return
	case newLease != nil && oldLease != nil:
		// Delete old discovery info if holderIdentity changed, implying
		// the sever was restarted.
		if oldLease.Name == newLease.Name && *oldLease.Spec.HolderIdentity != *newLease.Spec.HolderIdentity {
			h.peerAggDiscoveryCacheLock.Lock()
			delete(h.peerAggDiscoveryResponseCache, oldLease.Name)
			h.peerAggDiscoveryCacheLock.Unlock()
		}

		if err := h.addPeerDiscoveryInfoToCache(newLease); err != nil {
			klog.ErrorS(err, "error adding peer discovery cache", "serverID", newLease.Name)
		}
	case oldLease == nil && newLease != nil:
		if err := h.addPeerDiscoveryInfoToCache(newLease); err != nil {
			klog.ErrorS(err, "error adding peer discovery cache", "serverID", newLease.Name)
		}
	case oldLease != nil && newLease == nil:
		h.peerAggDiscoveryCacheLock.Lock()
		delete(h.peerAggDiscoveryResponseCache, oldLease.Name)
		h.peerAggDiscoveryCacheLock.Unlock()
	}
}

func (h *peerProxyHandler) addPeerDiscoveryInfoToCache(serverIdentityLease *v1.Lease) error {
	// Don't record our own discovery info.
	if serverIdentityLease.Name == h.serverId {
		return nil
	}

	h.peerAggDiscoveryCacheLock.RLock()
	discoveryDoc, ok := h.peerAggDiscoveryResponseCache[serverIdentityLease.Name]
	h.peerAggDiscoveryCacheLock.RUnlock()
	if !ok {
		return h.fetchNewDiscoveryFor(serverIdentityLease.Name, serverIdentityLease.Spec.HolderIdentity)
	}

	if discoveryDoc.holderIdentity != *serverIdentityLease.Spec.HolderIdentity {
		return h.fetchNewDiscoveryFor(serverIdentityLease.Name, serverIdentityLease.Spec.HolderIdentity)
	}

	return nil
}

func (h *peerProxyHandler) fetchNewDiscoveryFor(serverID string, holderIdentity *string) error {
	discoveryDoc := &peerAggDiscoveryInfo{
		holderIdentity:  *holderIdentity,
		servedResources: make(map[schema.GroupVersion][]string),
	}

	hostport, err := h.hostportInfo(serverID)
	if err != nil {
		return fmt.Errorf("failed to get host port info from identity lease for server %s: %w", serverID, err)
	}

	klog.V(4).Infof("Proxying an agg-discovery call from %s to %s", h.serverId, serverID)
	discoveryPaths := []string{"/api", "/apis"}
	for _, path := range discoveryPaths {
		discoveryResponse, err := h.aggregateDiscovery(path, hostport)
		if err != nil {
			klog.ErrorS(err, "error querying discovery endpoint", "path", path, "serverID", serverID)
			continue
		}

		if discoveryResponse == nil {
			klog.V(4).InfoS("discovery response is nil", "path", path, "serverID", serverID)
			continue
		}

		for _, groupDiscovery := range discoveryResponse.Items {
			groupName := groupDiscovery.Name
			if groupName == "" {
				groupName = "core"
			}

			for _, version := range groupDiscovery.Versions {
				gv := schema.GroupVersion{Group: groupName, Version: version.Version}
				resources := make([]string, 0, len(version.Resources))
				for _, resource := range version.Resources {
					resources = append(resources, resource.Resource)
				}
				discoveryDoc.servedResources[gv] = resources
			}
		}
	}

	klog.V(4).Infof("Agg discovery done successfully by %s for %s", h.serverId, serverID)
	h.peerAggDiscoveryCacheLock.Lock()
	h.peerAggDiscoveryResponseCache[serverID] = discoveryDoc
	h.peerAggDiscoveryCacheLock.Unlock()
	return nil
}

func (h *peerProxyHandler) aggregateDiscovery(path string, hostport string) (*apidiscoveryv2.APIGroupDiscoveryList, error) {
	req, err := http.NewRequest(http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	apiServerUser := &user.DefaultInfo{
		Name:   user.APIServerUser,
		UID:    user.APIServerUser,
		Groups: []string{user.SystemPrivilegedGroup},
	}

	ctx := apirequest.WithUser(req.Context(), apiServerUser)
	req = req.WithContext(ctx)

	req.Header.Add("Accept", discovery.AcceptV2)

	writer := responsewriterutil.NewInMemoryResponseWriter()
	h.proxyRequestToDestinationAPIServer(req, writer, hostport)
	if writer.RespCode() != http.StatusOK {
		return nil, fmt.Errorf("discovery request failed with status: %d", writer.RespCode())
	}

	parsed := &apidiscoveryv2.APIGroupDiscoveryList{}
	if err := runtime.DecodeInto(h.discoverySerializer.UniversalDecoder(), writer.Data(), parsed); err != nil {
		return nil, fmt.Errorf("error decoding discovery response: %w", err)
	}

	return parsed, nil
}
