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
	maxRetries     = 5
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

	newCache := map[string]map[schema.GroupVersionResource]bool{}
	for _, l := range leases {
		_, ok := h.isValidPeerIdentityLease(l)
		if !ok {
			continue
		}

		discoveryInfo, err := h.fetchNewDiscoveryFor(ctx, l.Name, *l.Spec.HolderIdentity)
		if err != nil {
			fetchDiscoveryErr = err
		}

		if discoveryInfo != nil {
			newCache[l.Name] = discoveryInfo
		}
	}

	// Overwrite cache with new contents.
	h.peerDiscoveryInfoCache.Store(newCache)
	return fetchDiscoveryErr
}

func (h *peerProxyHandler) fetchNewDiscoveryFor(ctx context.Context, serverID string, holderIdentity string) (map[schema.GroupVersionResource]bool, error) {
	hostport, err := h.hostportInfo(serverID)
	if err != nil {
		return nil, fmt.Errorf("failed to get host port info from identity lease for server %s: %w", serverID, err)
	}

	klog.V(4).Infof("Proxying an agg-discovery call from %s to %s", h.serverID, serverID)
	servedResources := make(map[schema.GroupVersionResource]bool)
	var discoveryErr error
	var discoveryResponse *apidiscoveryv2.APIGroupDiscoveryList
	discoveryPaths := []string{"/api", "/apis"}
	for _, path := range discoveryPaths {
		discoveryResponse, discoveryErr = h.aggregateDiscovery(ctx, path, hostport)
		if err != nil {
			klog.ErrorS(err, "error querying discovery endpoint for serverID", "path", path, "serverID", serverID)
			continue
		}

		for _, groupDiscovery := range discoveryResponse.Items {
			groupName := groupDiscovery.Name
			if groupName == "" {
				groupName = "core"
			}

			for _, version := range groupDiscovery.Versions {
				for _, resource := range version.Resources {
					gvr := schema.GroupVersionResource{Group: groupName, Version: version.Version, Resource: resource.Resource}
					servedResources[gvr] = true
				}
			}
		}
	}

	klog.V(4).Infof("Agg discovery done successfully by %s for %s", h.serverID, serverID)
	return servedResources, discoveryErr
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
