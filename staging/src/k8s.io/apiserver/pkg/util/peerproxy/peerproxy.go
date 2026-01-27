/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/workqueue"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coordinationv1informers "k8s.io/client-go/informers/coordination/v1"
)

const (
	// localDiscoveryRefreshInterval is the interval at which the local discovery cache is refreshed.
	// This cache is only used for mixed version proxy routing decisions (shouldServeLocally), not for serving discovery responses.
	// Periodic refreshes ensure that we prefer serving requests locally when possible. Without this refresh,
	// a stale cache could cause us to proxy a request to a peer even when we can serve it locally, resulting
	// in unnecessary network hops. This is particularly important during upgrades when new built-in APIs become
	// available on this server.
	localDiscoveryRefreshInterval = 30 * time.Minute
	// defaultExclusionGracePeriod is the default duration to wait before
	// removing a groupversion from the exclusion set after it is deleted from
	// CRDs and aggregated APIs.
	// This is to allow time for all peer API servers to also observe
	// the deleted CRD or aggregated API before this server stops excluding it
	// in peer-aggregated discovery and while proxying requests to peers.
	defaultExclusionGracePeriod = 5 * time.Minute
	// defaultExclusionReaperInterval is the interval at which the we
	// clean up deleted groups from the exclusion list.
	defaultExclusionReaperInterval = 1 * time.Minute
)

// Interface defines how the Mixed Version Proxy filter interacts with the underlying system.
type Interface interface {
	WrapHandler(handler http.Handler) http.Handler
	WaitForCacheSync(stopCh <-chan struct{}) error
	HasFinishedSync() bool
	RunLocalDiscoveryCacheSync(stopCh <-chan struct{}) error
	RunPeerDiscoveryCacheSync(ctx context.Context, workers int)
	GetPeerResources() map[string][]apidiscoveryv2.APIGroupDiscovery
	RegisterCacheInvalidationCallback(cb func())

	// RegisterCRDInformerHandlers registers event handlers on the CRD informer to track
	// which GroupVersions are served locally by CRDs. When a CRD is created or updated,
	// its GV is added to the exclusion set. When deleted, the GV is marked for exclusion
	// during a grace period to allow peers to observe the deletion. The extractor function
	// extracts the GroupVersion from a CRD object.
	//
	// This exclusion is necessary because peer discovery is not refreshed when a local
	// CRD is deleted. Without exclusion, the deleted GV might still appear in cached peer
	// discovery data, causing requests to be incorrectly routed to a peer for a GV that
	// no longer exists locally. Therefore, we intentionally exclude CRD GVs from peer
	// discovery from the start and only rely on the local apiserver's view of the CRD
	// to serve it in peer-aggregated discovery.
	RegisterCRDInformerHandlers(crdInformer cache.SharedIndexInformer, extractor GVExtractor) error

	// RegisterAPIServiceInformerHandlers registers event handlers on the APIService informer
	// to track which GroupVersions are served locally by aggregated APIServices. When an
	// APIService is created or updated, its GV is added to the exclusion set. When deleted,
	// the GV is marked for exclusion during a grace period.
	//
	// This exclusion is necessary because peer discovery is not refreshed when a local
	// aggregated APIService is deleted. Without exclusion, the deleted GV might still appear
	// in cached peer discovery data, causing requests to be incorrectly routed to a peer.
	// Therefore, we intentionally exclude aggregated APIService GVs from peer discovery
	// from the start and only rely on the local apiserver's view to serve them in
	// peer-aggregated discovery.
	RegisterAPIServiceInformerHandlers(apiServiceInformer cache.SharedIndexInformer, extractor GVExtractor) error

	// RunPeerDiscoveryActiveGVTracker starts a worker that processes CRD/APIService informer
	// events to rebuild the set of actively served GroupVersions. This worker is triggered
	// whenever a CRD or APIService is added or updated and updates the exclusion
	// set accordingly.
	RunPeerDiscoveryActiveGVTracker(ctx context.Context)

	// RunPeerDiscoveryReaper starts a background worker that periodically removes expired
	// GroupVersions from the exclusion set. When a CRD/APIService is deleted, its GV remains
	// in the exclusion set for a grace period (default 5 minutes) to allow all peer API servers
	// to observe the deletion. The reaper runs at a configured interval (default 1 minute)
	// and removes GVs whose grace period has elapsed.
	RunPeerDiscoveryReaper(ctx context.Context)

	// RunPeerDiscoveryRefilter starts a worker that re-applies exclusion filtering to the
	// cached peer discovery data whenever the exclusion set changes. This ensures that
	// already-cached peer discovery responses are immediately updated to exclude newly added
	// or updated local GVs, rather than waiting for the next peer lease event to trigger a
	// cache refresh of peer discovery data.
	RunPeerDiscoveryRefilter(ctx context.Context)
}

// New creates a new instance to implement unknown version proxy
// This method is used for an alpha feature UnknownVersionInteroperabilityProxy
// and is subject to future modifications.
func NewPeerProxyHandler(
	serverId string,
	identityLeaseLabelSelector string,
	leaseInformer coordinationv1informers.LeaseInformer,
	reconciler reconcilers.PeerEndpointLeaseReconciler,
	ser runtime.NegotiatedSerializer,
	loopbackClientConfig *rest.Config,
	proxyClientConfig *transport.Config,
) (*peerProxyHandler, error) {
	h := &peerProxyHandler{
		name:                             "PeerProxyHandler",
		serverID:                         serverId,
		reconciler:                       reconciler,
		serializer:                       ser,
		localDiscoveryInfoCache:          atomic.Value{},
		localDiscoveryCacheTicker:        time.NewTicker(localDiscoveryRefreshInterval),
		localDiscoveryInfoCachePopulated: make(chan struct{}),
		rawPeerDiscoveryCache:            atomic.Value{},
		peerLeaseQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: peerDiscoveryControllerName,
			}),
		apiserverIdentityInformer: leaseInformer,
	}

	h.gvExclusionManager = NewGVExclusionManager(
		defaultExclusionGracePeriod,
		defaultExclusionReaperInterval,
		&h.rawPeerDiscoveryCache,
		&h.cacheInvalidationCallback,
	)

	if parts := strings.Split(identityLeaseLabelSelector, "="); len(parts) != 2 {
		return nil, fmt.Errorf("invalid identityLeaseLabelSelector provided, must be of the form key=value, received: %v", identityLeaseLabelSelector)
	}
	selector, err := labels.Parse(identityLeaseLabelSelector)
	if err != nil {
		return nil, fmt.Errorf("failed to parse label selector: %w", err)
	}
	h.identityLeaseLabelSelector = selector

	discoveryScheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(discoveryScheme))
	h.discoverySerializer = serializer.NewCodecFactory(discoveryScheme)

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(loopbackClientConfig)
	if err != nil {
		return nil, fmt.Errorf("error creating discovery client: %w", err)
	}

	// Always use local discovery to get local view of resources.
	discoveryClient.NoPeerDiscovery = true
	h.discoveryClient = discoveryClient
	h.localDiscoveryInfoCache.Store(map[schema.GroupVersionResource]bool{})
	h.rawPeerDiscoveryCache.Store(map[string]PeerDiscoveryCacheEntry{})

	proxyTransport, err := transport.New(proxyClientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy transport: %w", err)
	}
	h.proxyTransport = proxyTransport

	peerDiscoveryRegistration, err := h.apiserverIdentityInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if lease, ok := h.isValidPeerIdentityLease(obj); ok {
				h.enqueueLease(lease)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			oldLease, oldLeaseOk := h.isValidPeerIdentityLease(oldObj)
			newLease, newLeaseOk := h.isValidPeerIdentityLease(newObj)
			if oldLeaseOk && newLeaseOk &&
				oldLease.Name == newLease.Name && *oldLease.Spec.HolderIdentity != *newLease.Spec.HolderIdentity {
				h.enqueueLease(newLease)
			}
		},
		DeleteFunc: func(obj interface{}) {
			if lease, ok := h.isValidPeerIdentityLease(obj); ok {
				h.enqueueLease(lease)
			}
		},
	})
	if err != nil {
		return nil, err
	}

	h.leaseRegistration = peerDiscoveryRegistration
	return h, nil
}

// RegisterCRDInformerHandlers registers event handlers for CRD informer.
func (h *peerProxyHandler) RegisterCRDInformerHandlers(crdInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	if h.gvExclusionManager != nil {
		return h.gvExclusionManager.RegisterCRDInformerHandlers(crdInformer, extractor)
	}
	return nil
}

// RegisterAPIServiceInformerHandlers registers event handlers for APIService informer.
func (h *peerProxyHandler) RegisterAPIServiceInformerHandlers(apiServiceInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	if h.gvExclusionManager != nil {
		return h.gvExclusionManager.RegisterAPIServiceInformerHandlers(apiServiceInformer, extractor)
	}
	return nil
}

// RunPeerDiscoveryActiveGVTracker starts the worker that tracks active GVs from CRDs/APIServices.
func (h *peerProxyHandler) RunPeerDiscoveryActiveGVTracker(ctx context.Context) {
	if h.gvExclusionManager != nil {
		h.gvExclusionManager.RunPeerDiscoveryActiveGVTracker(ctx)
	}
}

// RunPeerDiscoveryReaper starts the worker that removes expired GVs from the exclusion set.
func (h *peerProxyHandler) RunPeerDiscoveryReaper(ctx context.Context) {
	if h.gvExclusionManager != nil {
		h.gvExclusionManager.RunPeerDiscoveryReaper(ctx)
	}
}

// RunPeerDiscoveryRefilter starts the worker that refilters peer discovery cache.
func (h *peerProxyHandler) RunPeerDiscoveryRefilter(ctx context.Context) {
	if h.gvExclusionManager != nil {
		h.gvExclusionManager.RunPeerDiscoveryRefilter(ctx)
	}
}
