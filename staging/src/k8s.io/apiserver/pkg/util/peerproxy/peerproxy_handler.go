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
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/apiserver/pkg/util/peerproxy/metrics"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverproxyutil "k8s.io/apiserver/pkg/util/proxy"
	responsewriterutil "k8s.io/apiserver/pkg/util/responsewriter"
	coordinationv1 "k8s.io/client-go/listers/coordination/v1"
	restclient "k8s.io/client-go/rest"
)

const (
	PeerProxiedHeader = "x-kubernetes-peer-proxied"
)

type peerProxyHandler struct {
	name                      string
	apiserverIdentityInformer cache.SharedIndexInformer
	// ApiserverIdentityLister is used to fetch all apiservers in the cluster.
	apiserverIdentityLister coordinationv1.LeaseLister
	// identity for this server
	serverId string
	// reconciler that is used to fetch host port of peer apiserver when proxying request to a peer
	reconciler                    reconcilers.PeerEndpointLeaseReconciler
	serializer                    runtime.NegotiatedSerializer
	discoverySerializer           serializer.CodecFactory
	loopbackClientConfig          *restclient.Config
	proxyClientConfig             *transport.Config
	finishedSync                  atomic.Bool
	localDiscoveryResponseCache   map[schema.GroupVersion][]string
	peerAggDiscoveryResponseCache map[string]map[schema.GroupVersion][]string
	discoveryResponseCacheLock    sync.RWMutex
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
	if err := h.populateLocalDiscoveryCache(); err != nil {
		return fmt.Errorf("failed to populate discovery cache: %w", err)
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
		peerEndpoint, err := h.findServiceableByPeerFromAggDiscoveryCache(gvr)
		if err != nil {
			klog.ErrorS(err, "error fetching remote server details while proxying")
			responsewriters.ErrorNegotiated(apierrors.NewServiceUnavailable("Error getting ip and port info of the remote server while proxying"), h.serializer, gvr.GroupVersion(), w, r)
			return
		}

		h.proxyRequestToDestinationAPIServer(r, w, peerEndpoint)

	})
}

func (h *peerProxyHandler) populateLocalDiscoveryCache() error {
	clear(h.localDiscoveryResponseCache)
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(h.loopbackClientConfig)
	if err != nil {
		return fmt.Errorf("error creating discovery client: %w", err)
	}

	_, resourcesByGV, _, err := discoveryClient.GroupsAndMaybeResources()
	if err != nil {
		return fmt.Errorf("error getting API group resources from discovery: %w", err)
	}

	h.discoveryResponseCacheLock.Lock()
	defer h.discoveryResponseCacheLock.Unlock()
	for gv, resources := range resourcesByGV {
		h.localDiscoveryResponseCache[gv] = []string{}
		for _, resource := range resources.APIResources {
			h.localDiscoveryResponseCache[gv] = append(h.localDiscoveryResponseCache[gv], resource.Name)
		}
	}

	return nil
}

func (h *peerProxyHandler) shouldServeLocally(gvr schema.GroupVersionResource) bool {
	return h.localDiscoverySuccessful(gvr)
}

func (h *peerProxyHandler) localDiscoverySuccessful(gvr schema.GroupVersionResource) bool {
	resources, ok := h.localDiscoveryResponseCache[gvr.GroupVersion()]
	if !ok {
		klog.Errorf("resource not found for %v in local discovery cache\n", gvr.GroupVersion())
		return false
	}

	for _, resourceName := range resources {
		if gvr.GroupResource().Resource == resourceName {
			return true
		}
	}

	return false
}

func (h *peerProxyHandler) findServiceableByPeerFromAggDiscoveryCache(gvr schema.GroupVersionResource) (string, error) {
	var peerServerID, peerEndpoint string
	var foundResource bool
	var err error

	for serverID, resourceMap := range h.peerAggDiscoveryResponseCache {
		if serverID == h.serverId {
			continue
		}

		peerServerID = ""
		foundResource = false

		for gv, resources := range resourceMap {
			if gv.Group == gvr.Group && gv.Version == gvr.Version {
				for _, resource := range resources {
					if resource == gvr.Resource {
						peerServerID = serverID
						foundResource = true
						break
					}
				}
			}

			if foundResource {
				break
			}
		}

		if foundResource {
			peerEndpoint, err = getHostPortInfoFromLease(peerServerID, h.reconciler)
			if err != nil {
				klog.Errorf("failed to get host port info from identity lease for server %s: %v", peerServerID, err)
				continue
			}
		}

		if peerEndpoint != "" {
			return peerEndpoint, nil
		}
	}

	return "", fmt.Errorf("failed to fetch any apiserver capable of handling the request")
}

func getHostPortInfoFromLease(apiserverKey string, reconciler reconcilers.PeerEndpointLeaseReconciler) (string, error) {
	hostPort, err := reconciler.GetEndpoint(apiserverKey)
	if err != nil {
		return "", err
	}
	// check ip format
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

func (h *peerProxyHandler) addPeerDiscoveryInfo(obj interface{}) {
	serverIdentityLease, ok := obj.(*v1.Lease)
	if !ok {
		klog.Error("Invalid lease object provided to addPeerServedResources()")
		return
	}

	// We only want to record peers' discovery info.
	if serverIdentityLease.Name == h.serverId {
		return
	}

	h.updatePeerServedResourcesCache(nil, serverIdentityLease)
}

func (h *peerProxyHandler) updatePeerDiscoveryInfo(oldObj interface{}, newObj interface{}) {
	oldServerIdentityLease, ok := oldObj.(*v1.Lease)
	if !ok {
		klog.Error("Invalid lease object provided to updatePeerServedResources()")
		return
	}

	newServerIdentityLease, ok := newObj.(*v1.Lease)
	if !ok {
		klog.Error("Invalid lease object provided to updatePeerServedResources()")
		return
	}

	h.updatePeerServedResourcesCache(oldServerIdentityLease, newServerIdentityLease)
}

func (h *peerProxyHandler) deletePeerDiscoveryInfo(obj interface{}) {
	serverIdentityLease, ok := obj.(*v1.Lease)
	if !ok {
		klog.Error("Invalid lease object provided to addPeerServedResources()")
		return
	}

	if h.serverId == serverIdentityLease.Name {
		return
	}

	h.deletePeerDiscoveryInfoFromCache(serverIdentityLease.Name)
}

func (h *peerProxyHandler) updatePeerServedResourcesCache(oldLease *v1.Lease, newLease *v1.Lease) {
	if oldLease != nil && newLease != nil {
		// update peer discovery info only if the holderIdentity changed, meaning
		// the sever was restarted
		if oldLease.Name == newLease.Name {
			// We only want to record peers' discovery info.
			if newLease.Name == h.serverId {
				return
			}

			if *oldLease.Spec.HolderIdentity != *newLease.Spec.HolderIdentity {
				h.deletePeerDiscoveryInfoFromCache(oldLease.Name)
				h.addPeerDiscoveryInfoToCache(newLease.Name)
				return
			}
		}
	}

	if newLease != nil {
		// add new peer's served-resources info
		err := h.addPeerDiscoveryInfoToCache(newLease.Name)
		if err != nil {
			klog.ErrorS(err, "error while updating peer discovery cache for server ID:", newLease.Name)
		}
		return
	}

	if oldLease != nil {
		// delete old peer's served-resources info
		h.deletePeerDiscoveryInfoFromCache(oldLease.Name)
	}
}

func (h *peerProxyHandler) addPeerDiscoveryInfoToCache(serverID string) error {
	hostport, err := getHostPortInfoFromLease(serverID, h.reconciler)
	if err != nil {
		return fmt.Errorf("failed to get host port info from identity lease for server %s: %v", serverID, err)
	}

	if h.peerAggDiscoveryResponseCache == nil {
		h.peerAggDiscoveryResponseCache = make(map[string]map[schema.GroupVersion][]string)
	}

	gvrMap, ok := h.peerAggDiscoveryResponseCache[serverID]
	if !ok {
		gvrMap = make(map[schema.GroupVersion][]string)
		h.peerAggDiscoveryResponseCache[serverID] = gvrMap
	}

	discoveryPaths := []string{"/api", "/apis"}
	for _, path := range discoveryPaths {
		discoveryResponse, err := h.aggregateDiscovery(path, hostport) // Pass context
		if err != nil {
			klog.Errorf("error while querying discovery endpoint: %v", err)
			continue
		}

		if discoveryResponse == nil {
			continue
		}

		for _, groupDiscovery := range discoveryResponse.Items {
			groupName := groupDiscovery.Name
			if groupName == "" || groupName == "core" {
				groupName = "core"
			}
			for _, version := range groupDiscovery.Versions {
				gv := schema.GroupVersion{Group: groupName, Version: version.Version}
				resources := make([]string, 0, len(version.Resources))
				for _, resource := range version.Resources {
					resources = append(resources, resource.Resource)
				}
				gvrMap[gv] = resources
			}
		}
	}
	return nil
}

func (h *peerProxyHandler) deletePeerDiscoveryInfoFromCache(serverID string) {
	if h.peerAggDiscoveryResponseCache == nil {
		return
	}

	delete(h.peerAggDiscoveryResponseCache, serverID)
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
