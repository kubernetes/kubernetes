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
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/apiserver/pkg/util/peerproxy/metrics"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverproxyutil "k8s.io/apiserver/pkg/util/proxy"
	coordinationv1informers "k8s.io/client-go/informers/coordination/v1"
)

const (
	PeerProxiedHeader = "x-kubernetes-peer-proxied"
)

type peerProxyHandler struct {
	name string
	// Identity for this server.
	serverID     string
	finishedSync atomic.Bool

	// Label to check against in identity leases to make sure
	// we are working with apiserver identity leases only.
	identityLeaseLabelSelector labels.Selector
	apiserverIdentityInformer  coordinationv1informers.LeaseInformer
	leaseRegistration          cache.ResourceEventHandlerRegistration
	// Reconciler that is used to fetch host port of peer apiserver when proxying request to a peer.
	reconciler reconcilers.PeerEndpointLeaseReconciler
	// Client to make discovery calls locally.
	discoveryClient     *discovery.DiscoveryClient
	discoverySerializer serializer.CodecFactory
	// Cache that stores resources served by this apiserver. Refreshed periodically.
	// We always look up in the local discovery cache first, to check whether the
	// request can be served by this apiserver instead of proxying it to a peer.
	localDiscoveryInfoCache              atomic.Value
	localDiscoveryCacheTicker            *time.Ticker
	localDiscoveryInfoCachePopulated     chan struct{}
	localDiscoveryInfoCachePopulatedOnce sync.Once
	// Cache that stores resources and groups served by peer apiservers.
	// The map is from string to PeerDiscoveryCacheEntry where the string
	// is the serverID of the peer apiserver.
	// Refreshed if a new apiserver identity lease is added, deleted or
	// holderIndentity change is observed in the lease.
	peerDiscoveryInfoCache atomic.Value // map[string]struct{GVRs map[schema.GroupVersionResource]bool; Groups []apidiscoveryv2.APIGroupDiscovery}
	proxyTransport         http.RoundTripper
	// Worker queue that keeps the peerDiscoveryInfoCache up-to-date.
	peerLeaseQueue            workqueue.TypedRateLimitingInterface[string]
	serializer                runtime.NegotiatedSerializer
	cacheInvalidationCallback atomic.Pointer[func()]
	// Exclusion set for groups that should not be included in peer proxying
	// or peer-aggregated discovery (e.g., CRDs/APIServices)
	//
	// This map has three states for a group:
	// - Not in map: Group is not excluded.
	// - In map, value is nil: Group is actively excluded.
	// - In map, value is non-nil: Group is pending deletion (grace period).
	excludedGVs         map[schema.GroupVersion]*time.Time
	excludedGVsMu       sync.RWMutex
	crdInformer         cache.SharedIndexInformer
	crdExtractor        GVExtractor
	apiServiceInformer  cache.SharedIndexInformer
	apiServiceExtractor GVExtractor

	exclusionGracePeriod time.Duration
	reaperCheckInterval  time.Duration

	// Worker queue for processing GV deletions asynchronously
	gvDeletionQueue workqueue.TypedRateLimitingInterface[string]
}

// PeerDiscoveryCacheEntry holds the GVRs and group-level discovery info for a peer.
type PeerDiscoveryCacheEntry struct {
	GVRs           map[schema.GroupVersionResource]bool
	GroupDiscovery []apidiscoveryv2.APIGroupDiscovery
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	w   http.ResponseWriter
	ctx context.Context
}

func (h *peerProxyHandler) RegisterCacheInvalidationCallback(cb func()) {
	h.cacheInvalidationCallback.Store(&cb)
}

func (h *peerProxyHandler) HasFinishedSync() bool {
	return h.finishedSync.Load()
}

func (h *peerProxyHandler) WaitForCacheSync(stopCh <-chan struct{}) error {
	ok := cache.WaitForNamedCacheSync("mixed-version-proxy", stopCh, h.apiserverIdentityInformer.Informer().HasSynced)
	if !ok {
		return fmt.Errorf("error while waiting for initial cache sync")
	}

	if !cache.WaitForNamedCacheSync(peerDiscoveryControllerName, stopCh, h.leaseRegistration.HasSynced) {
		return fmt.Errorf("error while waiting for peer-identity-lease event handler registration sync")
	}

	if h.crdInformer != nil && !cache.WaitForNamedCacheSync("peer-discovery-crd-informer", stopCh, h.crdInformer.HasSynced) {
		return fmt.Errorf("error while waiting for crd informer sync")
	}

	if h.apiServiceInformer != nil && !cache.WaitForNamedCacheSync("peer-discovery-api-service-informer", stopCh, h.apiServiceInformer.HasSynced) {
		return fmt.Errorf("error while waiting for apiservice informer sync")
	}

	// Wait for localDiscoveryInfoCache to be populated.
	select {
	case <-h.localDiscoveryInfoCachePopulated:
	case <-stopCh:
		return fmt.Errorf("stop signal received while waiting for local discovery cache population")
	}

	h.finishedSync.Store(true)
	return nil
}

// WrapHandler will fetch the apiservers that can serve the request and either serve it locally
// or route it to a peer
func (h *peerProxyHandler) WrapHandler(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			responsewriters.InternalError(w, r, errors.New("no RequestInfo found in the context"))
			return
		}

		// Allow non-resource requests
		if !requestInfo.IsResourceRequest {
			klog.V(3).Infof("Not a resource request skipping proxying")
			handler.ServeHTTP(w, r)
			return
		}

		// Request has already been proxied once, it must be served locally
		if r.Header.Get(PeerProxiedHeader) == "true" {
			klog.V(3).Infof("Already rerouted once, skipping proxying to peer")
			handler.ServeHTTP(w, r)
			return
		}

		// Apiserver Identity Informers is not synced yet, pass request to next handler
		// This will happen for self requests from the kube-apiserver because we have a poststarthook
		// to ensure that external requests are not served until the ApiserverIdentity Informer has synced
		if !h.HasFinishedSync() {
			handler.ServeHTTP(w, r)
			return
		}

		gvr := schema.GroupVersionResource{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion, Resource: requestInfo.Resource}

		if h.shouldServeLocally(gvr) {
			handler.ServeHTTP(w, r)
			return
		}

		// find servers that are capable of serving this request
		peerServerIDs := h.findServiceableByPeerFromPeerDiscoveryCache(gvr)
		if len(peerServerIDs) == 0 {
			klog.Errorf("gvr %v is not served by anything in this cluster", gvr)
			handler.ServeHTTP(w, r)
			return
		}

		peerEndpoints, err := h.resolveServingLocation(peerServerIDs)
		if err != nil {
			gv := schema.GroupVersion{Group: gvr.Group, Version: gvr.Version}
			klog.ErrorS(err, "error finding serviceable-by apiservers for the requested resource", "gvr", gvr)
			responsewriters.ErrorNegotiated(apierrors.NewServiceUnavailable("Error getting ip and port info of the remote server while proxying"), h.serializer, gv, w, r)
			return
		}

		endpointIndex := rand.Intn(len(peerEndpoints))
		peerEndpoint := peerEndpoints[endpointIndex]
		h.proxyRequestToDestinationAPIServer(r, w, peerEndpoint)
	})
}

// resolveServingLocation resolves the host:port addresses for the given peer IDs.
func (h *peerProxyHandler) resolveServingLocation(peerIDs []string) ([]string, error) {
	var peerServerEndpoints []string
	var errs []error

	for _, id := range peerIDs {
		hostPort, err := h.hostportInfo(id)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		peerServerEndpoints = append(peerServerEndpoints, hostPort)
	}

	// reset err if there was atleast one valid peer server found.
	if len(peerServerEndpoints) > 0 {
		errs = nil
	}

	return peerServerEndpoints, errors.Join(errs...)
}

func (h *peerProxyHandler) hostportInfo(apiserverKey string) (string, error) {
	hostPort, err := h.reconciler.GetEndpoint(apiserverKey)
	if err != nil {
		return "", err
	}

	_, _, err = net.SplitHostPort(hostPort)
	if err != nil {
		return "", err
	}

	return hostPort, nil
}

func (h *peerProxyHandler) proxyRequestToDestinationAPIServer(req *http.Request, rw http.ResponseWriter, host string) {
	// write a new location based on the existing request pointed at the target service
	location := &url.URL{}
	location.Scheme = "https"
	location.Host = host
	location.Path = req.URL.Path
	location.RawQuery = req.URL.Query().Encode()

	newReq, cancelFn := apiserverproxyutil.NewRequestForProxy(location, req)
	newReq.Header.Add(PeerProxiedHeader, "true")
	defer cancelFn()

	proxyRoundTripper, err := h.buildProxyRoundtripper(req)
	if err != nil {
		klog.Errorf("failed to build proxy round tripper: %v", err)
		return
	}

	delegate := &epmetrics.ResponseWriterDelegator{ResponseWriter: rw}
	w := responsewriter.WrapForHTTP1Or2(delegate)
	handler := proxy.NewUpgradeAwareHandler(location, proxyRoundTripper, true, false, &responder{w: w, ctx: req.Context()})
	klog.Infof("Proxying request for %s from %s to %s", req.URL.Path, req.Host, location.Host)
	handler.ServeHTTP(w, newReq)
	metrics.IncPeerProxiedRequest(req.Context(), strconv.Itoa(delegate.Status()))
}

func (h *peerProxyHandler) buildProxyRoundtripper(req *http.Request) (http.RoundTripper, error) {
	user, ok := apirequest.UserFrom(req.Context())
	if !ok {
		return nil, apierrors.NewBadRequest("no user details present in request")
	}

	return transport.NewAuthProxyRoundTripper(user.GetName(), user.GetUID(), user.GetGroups(), user.GetExtra(), h.proxyTransport), nil
}

func (r *responder) Error(w http.ResponseWriter, req *http.Request, err error) {
	klog.ErrorS(err, "Error while proxying request to destination apiserver")
	http.Error(w, err.Error(), http.StatusServiceUnavailable)
}

// GetPeerResources implements PeerDiscoveryProvider interface
// Returns a map of serverID -> []apidiscoveryv2.APIGroupDiscovery served by peer servers
func (h *peerProxyHandler) GetPeerResources() map[string][]apidiscoveryv2.APIGroupDiscovery {
	result := make(map[string][]apidiscoveryv2.APIGroupDiscovery)

	peerCache := h.peerDiscoveryInfoCache.Load()
	if peerCache == nil {
		klog.V(4).Infof("GetPeerResources: peer cache is nil")
		return result
	}

	cacheMap, ok := peerCache.(map[string]PeerDiscoveryCacheEntry)
	if !ok {
		klog.Warning("Invalid cache type in peerDiscoveryGVRCache")
		return result
	}

	for serverID, peerData := range cacheMap {
		if serverID == h.serverID {
			klog.V(4).Infof("GetPeerResources: skipping local server %s", serverID)
			continue // Skip local server
		}
		result[serverID] = peerData.GroupDiscovery
	}

	return result
}
