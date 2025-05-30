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
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	v1 "k8s.io/api/coordination/v1"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coordinationv1informers "k8s.io/client-go/informers/coordination/v1"
)

// Local discovery cache needs to be refreshed periodically to store
// updates made to custom resources or aggregated resource that can
// change dynamically.
const localDiscoveryRefreshInterval = 30 * time.Minute

// Interface defines how the Mixed Version Proxy filter interacts with the underlying system.
type Interface interface {
	WrapHandler(handler http.Handler) http.Handler
	WaitForCacheSync(stopCh <-chan struct{}) error
	HasFinishedSync() bool
	RunLocalDiscoveryCacheSync(stopCh <-chan struct{}) error
	RunPeerDiscoveryCacheSync(ctx context.Context, workers int)
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
		peerDiscoveryInfoCache:           atomic.Value{},
		peerLeaseQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName,
			}),
		apiserverIdentityInformer: leaseInformer,
	}

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
	h.discoveryClient = discoveryClient
	h.localDiscoveryInfoCache.Store(map[schema.GroupVersionResource]bool{})
	h.peerDiscoveryInfoCache.Store(map[string]map[schema.GroupVersionResource]bool{})

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

func (h *peerProxyHandler) isValidPeerIdentityLease(obj interface{}) (*v1.Lease, bool) {
	lease, ok := obj.(*v1.Lease)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return nil, false
		}
		if lease, ok = tombstone.Obj.(*v1.Lease); !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %T", obj))
			return nil, false
		}
	}

	if lease == nil {
		klog.Error(fmt.Errorf("nil lease object provided"))
		return nil, false
	}

	if h.identityLeaseLabelSelector != nil && h.identityLeaseLabelSelector.String() != "" {
		identityLeaseLabel := strings.Split(h.identityLeaseLabelSelector.String(), "=")
		if len(identityLeaseLabel) != 2 {
			klog.Errorf("invalid identityLeaseLabelSelector format: %s", h.identityLeaseLabelSelector.String())
			return nil, false
		}

		if lease.Labels == nil || lease.Labels[identityLeaseLabel[0]] != identityLeaseLabel[1] {
			klog.V(4).Infof("lease %s/%s does not match label selector: %s=%s", lease.Namespace, lease.Name, identityLeaseLabel[0], identityLeaseLabel[1])
			return nil, false
		}

	}

	// Ignore self.
	if lease.Name == h.serverID {
		return nil, false
	}

	if lease.Spec.HolderIdentity == nil {
		klog.Error(fmt.Errorf("invalid lease object provided, missing holderIdentity in lease obj"))
		return nil, false
	}

	return lease, true
}
