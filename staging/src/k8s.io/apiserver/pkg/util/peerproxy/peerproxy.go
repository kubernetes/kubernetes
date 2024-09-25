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
	"net/http"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/transport"

	kubeinformers "k8s.io/client-go/informers"
)

// Interface defines how the Unknown Version Proxy filter interacts with the underlying system.
type Interface interface {
	WrapHandler(handler http.Handler) http.Handler
	WaitForCacheSync(stopCh <-chan struct{}) error
	HasFinishedSync() bool
}

// New creates a new instance to implement unknown version proxy
func NewPeerProxyHandler(informerFactory kubeinformers.SharedInformerFactory,
	serverId string,
	reconciler reconcilers.PeerEndpointLeaseReconciler,
	serializer runtime.NegotiatedSerializer,
	discoverySerlializer serializer.CodecFactory,
	loopbackClientConfig *rest.Config,
	proxyClientConfig *transport.Config) *peerProxyHandler {
	h := &peerProxyHandler{
		name:                 "PeerProxyHandler",
		serverId:             serverId,
		reconciler:           reconciler,
		serializer:           serializer,
		discoverySerializer:  discoverySerlializer,
		loopbackClientConfig: loopbackClientConfig,
		proxyClientConfig:    proxyClientConfig,
	}
	h.apiserverIdentityInformer = informerFactory.Coordination().V1().Leases().Informer()
	h.apiserverIdentityLister = informerFactory.Coordination().V1().Leases().Lister()
	return h
}
