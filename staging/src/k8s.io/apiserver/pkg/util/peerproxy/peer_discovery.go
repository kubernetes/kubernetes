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
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/discovery"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	responsewriterutil "k8s.io/apiserver/pkg/util/responsewriter"
)

const (
	controllerName = "peer-discovery-cache-sync"
	// maxRetries is set to 20 to handle the race condition during API server startup
	// where identity leases are created before endpoint leases. During initialization,
	// peer discovery sync may attempt to fetch discovery from a peer before that peer
	// has created its endpoint lease, resulting in "missing port in address" errors.
	// With the default rate limiting (exponential backoff starting at 5ms), 20 retries
	// provides approximately 60-90 seconds of retry window, which is sufficient for
	// both identity and endpoint leases to be established during normal startup.
	maxRetries = 20
)

func (h *peerProxyHandler) RunPeerDiscoveryCacheSync(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer h.peerLeaseQueue.ShutDown()
	defer func() {
		err := h.apiserverIdentityInformer.Informer().RemoveEventHandler(h.leaseRegistration)
		if err != nil {
			klog.Warning("error removing leaseInformer eventhandler")
		}
	}()

	klog.Infof("Workers: %d", workers)
	for i := 0; i < workers; i++ {
		klog.Infof("Starting worker")
		go wait.UntilWithContext(ctx, h.runWorker, time.Second)
	}
	<-ctx.Done()
}

func (h *peerProxyHandler) enqueueLease(lease *v1.Lease) {
	h.peerLeaseQueue.Add(lease.Name)
}

func (h *peerProxyHandler) runWorker(ctx context.Context) {
	for h.processNextElectionItem(ctx) {
	}
}

func (h *peerProxyHandler) processNextElectionItem(ctx context.Context) bool {
	key, shutdown := h.peerLeaseQueue.Get()
	if shutdown {
		return false
	}
	defer h.peerLeaseQueue.Done(key)

	err := h.syncPeerDiscoveryCache(ctx)
	h.handleErr(err, key)
	return true
}

func (h *peerProxyHandler) syncPeerDiscoveryCache(ctx context.Context) error {
	var fetchDiscoveryErr error
	// Rebuild the peer discovery cache from available leases.
	leases, err := h.apiserverIdentityInformer.Lister().List(h.identityLeaseLabelSelector)
	if err != nil {
		utilruntime.HandleError(err)
		return err
	}

	newCache := map[string]PeerDiscoveryCacheEntry{}
	for _, l := range leases {
		_, ok := h.isValidPeerIdentityLease(l)
		if !ok {
			continue
		}

		discoveryEntry, err := h.fetchNewDiscoveryFor(ctx, l.Name)
		if err != nil {
			fetchDiscoveryErr = err
		}
		// Only add if there is at least one GVR or group
		if len(discoveryEntry.GVRs) > 0 || len(discoveryEntry.GroupDiscovery) > 0 {
			newCache[l.Name] = discoveryEntry
		}
	}

	// Overwrite cache with new contents.
	h.peerDiscoveryInfoCache.Store(newCache)

	if len(newCache) != 0 {
		// After updating peer discovery cache, invalidate discovery manager cache.
		for _, cb := range h.cacheInvalidationCallbacks {
			cb()
		}
	}
	return fetchDiscoveryErr
}

func (h *peerProxyHandler) fetchNewDiscoveryFor(ctx context.Context, serverID string) (PeerDiscoveryCacheEntry, error) {
	hostport, err := h.hostportInfo(serverID)
	if err != nil {
		return PeerDiscoveryCacheEntry{}, fmt.Errorf("failed to get host port info from identity lease for server %s: %w", serverID, err)
	}

	klog.V(4).Infof("Proxying an agg-discovery call from %s to %s", h.serverID, serverID)
	gvrMap := make(map[schema.GroupVersionResource]bool)
	var discoveryErr error
	var discoveryResponse *apidiscoveryv2.APIGroupDiscoveryList
	discoveryPaths := []string{"/api", "/apis"}
	groupMap := make(map[string]apidiscoveryv2.APIGroupDiscovery)
	for _, path := range discoveryPaths {
		discoveryResponse, discoveryErr = h.aggregateDiscovery(ctx, path, hostport)
		if discoveryErr != nil {
			klog.ErrorS(discoveryErr, "error querying discovery endpoint for serverID", "path", path, "serverID", serverID)
			continue
		}

		for _, groupDiscovery := range discoveryResponse.Items {
			for _, version := range groupDiscovery.Versions {
				for _, resource := range version.Resources {
					gvr := schema.GroupVersionResource{
						Group:    groupDiscovery.Name,
						Version:  version.Version,
						Resource: resource.Resource,
					}
					gvrMap[gvr] = true
				}
			}
			// Skip core/v1 group from peer-aggregated discovery since its not served from /apis.
			// We still want to re-route core/v1 requests to the peer, but we don't want it
			// to appear in the merged discovery document.
			if groupDiscovery.Name == "" {
				continue
			}
			groupMap[groupDiscovery.Name] = groupDiscovery
		}
	}

	groupList := make([]apidiscoveryv2.APIGroupDiscovery, 0, len(groupMap))
	for _, group := range groupMap {
		groupList = append(groupList, group)
	}

	klog.V(4).Infof("Agg discovery done successfully by %s for %s", h.serverID, serverID)
	return PeerDiscoveryCacheEntry{
		GVRs:           gvrMap,
		GroupDiscovery: groupList,
	}, discoveryErr
}

func (h *peerProxyHandler) aggregateDiscovery(ctx context.Context, path string, hostport string) (*apidiscoveryv2.APIGroupDiscoveryList, error) {
	req, err := http.NewRequest(http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	apiServerUser := &user.DefaultInfo{
		Name:   user.APIServerUser,
		UID:    user.APIServerUser,
		Groups: []string{user.AllAuthenticated},
	}

	ctx = apirequest.WithUser(ctx, apiServerUser)
	req = req.WithContext(ctx)

	// Fallback to V2 and V1 in that order if V2Unmerged is not recognized.
	req.Header.Add("Accept", discovery.AcceptV2Unmerged+","+discovery.AcceptV2+","+discovery.AcceptV1)

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

// handleErr checks if an error happened and makes sure we will retry later.
func (h *peerProxyHandler) handleErr(err error, key string) {
	if err == nil {
		h.peerLeaseQueue.Forget(key)
		return
	}

	if h.peerLeaseQueue.NumRequeues(key) < maxRetries {
		klog.Infof("Error syncing discovery for peer lease %v: %v", key, err)
		h.peerLeaseQueue.AddRateLimited(key)
		return
	}

	h.peerLeaseQueue.Forget(key)
	utilruntime.HandleError(err)
	klog.Infof("Dropping lease %s out of the queue: %v", key, err)
}
