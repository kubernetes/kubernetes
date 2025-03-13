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
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	kubeinformers "k8s.io/client-go/informers"
	informers "k8s.io/client-go/informers/coordination/v1"
)

const localDiscoveryRefreshInterval = 30 * time.Minute

// Interface defines how the Mixed Version Proxy filter interacts with the underlying system.
type Interface interface {
	WrapHandler(handler http.Handler) http.Handler
	WaitForCacheSync(stopCh <-chan struct{}) error
	HasFinishedSync() bool
}

// New creates a new instance to implement unknown version proxy
// This method is used for an alpha feature UnknownVersionInteroperabilityProxy
// and is subject to future modifications.
func NewPeerProxyHandler(identityLeaseLabelSelector string,
	informerFactory kubeinformers.SharedInformerFactory,
	serverId string,
	reconciler reconcilers.PeerEndpointLeaseReconciler,
	ser runtime.NegotiatedSerializer,
	loopbackClientConfig *rest.Config,
	proxyClientConfig *transport.Config) (*peerProxyHandler, error) {

	if parts := strings.Split(identityLeaseLabelSelector, "="); len(parts) != 2 {
		return nil, fmt.Errorf("invalid identityLeaseLabelSelector provided, must be of the form key=value, received: %v", identityLeaseLabelSelector)
	}

	client, err := kubernetes.NewForConfig(loopbackClientConfig)
	if err != nil {
		return nil, err
	}

	apiserverIdentityInformer := informers.NewFilteredLeaseInformer(
		client,
		metav1.NamespaceSystem,
		0,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(listOptions *metav1.ListOptions) {
			listOptions.LabelSelector = identityLeaseLabelSelector
		},
	)

	h := &peerProxyHandler{
		name:                          "PeerProxyHandler",
		serverId:                      serverId,
		reconciler:                    reconciler,
		serializer:                    ser,
		loopbackClientConfig:          loopbackClientConfig,
		proxyClientConfig:             proxyClientConfig,
		localDiscoveryResponseCache:   make(map[schema.GroupVersion][]string),
		localDiscoveryCacheTicker:     time.NewTicker(localDiscoveryRefreshInterval),
		peerAggDiscoveryResponseCache: make(map[string]*peerAggDiscoveryInfo),
		apiserverIdentityInformer:     apiserverIdentityInformer,
	}

	discoveryScheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(discoveryScheme))
	h.discoverySerializer = serializer.NewCodecFactory(discoveryScheme)
	_, err = h.apiserverIdentityInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			h.addPeerDiscoveryInfo(obj, identityLeaseLabelSelector)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			h.updatePeerDiscoveryInfo(oldObj, newObj, identityLeaseLabelSelector)
		},
		DeleteFunc: func(obj interface{}) {
			h.deletePeerDiscoveryInfo(obj, identityLeaseLabelSelector)
		},
	})

	if err != nil {
		return nil, fmt.Errorf("unable to add apiserver identity lease event handler: %w", err)
	}

	return h, nil
}
