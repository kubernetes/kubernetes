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
	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverproxyutil "k8s.io/apiserver/pkg/util/proxy"
	coordinationv1 "k8s.io/client-go/listers/coordination/v1"
	restclient "k8s.io/client-go/rest"
)

const (
	PeerProxiedHeader = "x-kubernetes-peer-proxied"
)

type peerAggDiscoveryInfo struct {
	holderIdentity  string
	servedResources map[schema.GroupVersion][]string
}

type peerProxyHandler struct {
	name                      string
	apiserverIdentityInformer cache.SharedIndexInformer
	// ApiserverIdentityLister is used to fetch all apiservers in the cluster.
	apiserverIdentityLister coordinationv1.LeaseLister
	// identity for this server
	serverId string
	// reconciler that is used to fetch host port of peer apiserver when proxying request to a peer
	reconciler                  reconcilers.PeerEndpointLeaseReconciler
	serializer                  runtime.NegotiatedSerializer
	discoverySerializer         serializer.CodecFactory
	loopbackClientConfig        *restclient.Config
	proxyClientConfig           *transport.Config
	finishedSync                atomic.Bool
	localDiscoveryResponseCache map[schema.GroupVersion][]string
	localDiscoveryCacheTicker   *time.Ticker
	// peerAggDiscoveryResponseCache is a map for each peer API server's ID to
	// its aggregated discovery information
	peerAggDiscoveryResponseCache map[string]*peerAggDiscoveryInfo
	peerAggDiscoveryCacheLock     sync.RWMutex
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	w   http.ResponseWriter
	ctx context.Context
}

func (h *peerProxyHandler) HasFinishedSync() bool {
	return h.finishedSync.Load()
}

func (h *peerProxyHandler) WaitForCacheSync(stopCh <-chan struct{}) error {
	ok := cache.WaitForNamedCacheSync("unknown-version-proxy", stopCh, h.apiserverIdentityInformer.HasSynced)
	if !ok {
		return fmt.Errorf("error while waiting for initial cache sync")
	}
	h.startLocalDiscoveryCacheInvalidation(stopCh)
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
		if requestInfo.APIGroup == "" {
			gvr.Group = "core"
		}

		if h.shouldServeLocally(gvr) {
			handler.ServeHTTP(w, r)
			return
		}

		// find servers that are capable of serving this request
		peerServerIDs := h.findServiceableByPeerFromAggDiscoveryCache(gvr)
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

		rand := rand.Intn(len(peerEndpoints))
		peerEndpoint := peerEndpoints[rand]
		h.proxyRequestToDestinationAPIServer(r, w, peerEndpoint)
	})
}

func (h *peerProxyHandler) startLocalDiscoveryCacheInvalidation(stopCh <-chan struct{}) {
	go func() {
		klog.Info("localDiscoveryCacheInvalidation goroutine started")
		for {
			select {
			case <-h.localDiscoveryCacheTicker.C:
				klog.V(4).Infof("Invalidating local discovery cache")
				if err := h.populateLocalDiscoveryCache(); err != nil {
					klog.Errorf("Failed to repopulate local discovery cache: %v", err)
				}
			case <-stopCh:
				klog.Info("localDiscoveryCacheInvalidation goroutine received stop signal")
				if h.localDiscoveryCacheTicker != nil {
					h.localDiscoveryCacheTicker.Stop()
					klog.Info("localDiscoveryCacheTicker stopped")
				}
				klog.Info("localDiscoveryCacheInvalidation goroutine exiting")
				return
			}
		}
	}()
}

func (h *peerProxyHandler) populateLocalDiscoveryCache() error {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(h.loopbackClientConfig)
	if err != nil {
		return fmt.Errorf("error creating discovery client: %w", err)
	}

	_, resourcesByGV, _, err := discoveryClient.GroupsAndMaybeResources()
	if err != nil {
		return fmt.Errorf("error getting API group resources from discovery: %w", err)
	}

	freshLocalDiscoveryResponse := map[schema.GroupVersion][]string{}
	for gv, resources := range resourcesByGV {
		if gv.Group == "" {
			gv.Group = "core"
		}
		freshLocalDiscoveryResponse[gv] = []string{}
		for _, resource := range resources.APIResources {
			freshLocalDiscoveryResponse[gv] = append(freshLocalDiscoveryResponse[gv], resource.Name)
		}
	}

	h.localDiscoveryResponseCache = freshLocalDiscoveryResponse
	return nil
}

func (h *peerProxyHandler) shouldServeLocally(gvr schema.GroupVersionResource) bool {
	resources, ok := h.localDiscoveryResponseCache[gvr.GroupVersion()]
	if !ok {
		klog.V(4).Infof("resource not found for %v in local discovery cache\n", gvr.GroupVersion())
		return false
	}

	for _, resourceName := range resources {
		if gvr.GroupResource().Resource == resourceName {
			return true
		}
	}

	return false
}

func (h *peerProxyHandler) findServiceableByPeerFromAggDiscoveryCache(gvr schema.GroupVersionResource) []string {
	var serviceableByIDs []string
	var foundResource bool

	h.peerAggDiscoveryCacheLock.RLock()
	defer h.peerAggDiscoveryCacheLock.RUnlock()
	for peerID, discoveryDoc := range h.peerAggDiscoveryResponseCache {
		// Ignore local apiserver.
		if peerID == h.serverId {
			continue
		}

		foundResource = false
		for gv, resources := range discoveryDoc.servedResources {
			if gv.Group == gvr.Group && gv.Version == gvr.Version {
				for _, resource := range resources {
					if resource == gvr.Resource {
						serviceableByIDs = append(serviceableByIDs, peerID)
						foundResource = true
						break
					}
				}
			}

			if foundResource {
				break
			}
		}
	}

	return serviceableByIDs
}

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
	handler.ServeHTTP(w, newReq)
	metrics.IncPeerProxiedRequest(req.Context(), strconv.Itoa(delegate.Status()))
}

func (h *peerProxyHandler) buildProxyRoundtripper(req *http.Request) (http.RoundTripper, error) {
	proxyTransport, err := transport.New(h.proxyClientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy transport")
	}

	user, ok := apirequest.UserFrom(req.Context())
	if !ok {
		return nil, err
	}

	return transport.NewAuthProxyRoundTripper(user.GetName(), user.GetUID(), user.GetGroups(), user.GetExtra(), proxyTransport), nil
}

func (r *responder) Error(w http.ResponseWriter, req *http.Request, err error) {
	klog.ErrorS(err, "Error while proxying request to destination apiserver")
	http.Error(w, err.Error(), http.StatusServiceUnavailable)
}
