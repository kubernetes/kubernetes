/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"context"
	"fmt"
	"net/http"
	"net/url"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	translator "k8s.io/apiserver/pkg/util/proxy"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/core/pod"
)

// ProxyREST implements the proxy subresource for a Pod
type ProxyREST struct {
	Store          *genericregistry.Store
	ProxyTransport http.RoundTripper
}

// Implement Connecter
var _ = rest.Connecter(&ProxyREST{})

var proxyMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

// New returns an empty podProxyOptions object.
func (r *ProxyREST) New() runtime.Object {
	return &api.PodProxyOptions{}
}

// Destroy cleans up resources on shutdown.
func (r *ProxyREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// ConnectMethods returns the list of HTTP methods that can be proxied
func (r *ProxyREST) ConnectMethods() []string {
	return proxyMethods
}

// NewConnectOptions returns versioned resource that represents proxy parameters
func (r *ProxyREST) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodProxyOptions{}, true, "path"
}

// Connect returns a handler for the pod proxy
func (r *ProxyREST) Connect(ctx context.Context, id string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	proxyOpts, ok := opts.(*api.PodProxyOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.ResourceLocation(ctx, r.Store, r.ProxyTransport, id)
	if err != nil {
		return nil, err
	}
	location.Path = net.JoinPreservingTrailingSlash(location.Path, proxyOpts.Path)
	// Return a proxy handler that uses the desired transport, wrapped with additional proxy handling (to get URL rewriting, X-Forwarded-* headers, etc)
	return newThrottledUpgradeAwareProxyHandler(location, transport, true, false, responder), nil
}

// Support both GET and POST methods. We must support GET for browsers that want to use WebSockets.
var upgradeableMethods = []string{"GET", "POST"}

// AttachREST implements the attach subresource for a Pod
type AttachREST struct {
	Store       *genericregistry.Store
	KubeletConn client.ConnectionInfoGetter
	Authorizer  authorizer.Authorizer
}

// Implement Connecter
var _ = rest.Connecter(&AttachREST{})

// New creates a new podAttachOptions object.
func (r *AttachREST) New() runtime.Object {
	return &api.PodAttachOptions{}
}

// Destroy cleans up resources on shutdown.
func (r *AttachREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Connect returns a handler for the pod exec proxy
func (r *AttachREST) Connect(ctx context.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	// Forces a authz check for "create", if feature gate enabled.
	// See: https://github.com/kubernetes/kubernetes/issues/133515
	if utilfeature.DefaultFeatureGate.Enabled(features.AuthorizePodWebsocketUpgradeCreatePermission) {
		if err := ensureAuthorizedForVerb(ctx, r.Authorizer, "create"); err != nil {
			return nil, err
		}
	}

	attachOpts, ok := opts.(*api.PodAttachOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.AttachLocation(ctx, r.Store, r.KubeletConn, name, attachOpts)
	if err != nil {
		return nil, err
	}
	handler := newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder)
	if utilfeature.DefaultFeatureGate.Enabled(features.TranslateStreamCloseWebsocketRequests) {
		// Wrap the upgrade aware handler to implement stream translation
		// for WebSocket/V5 upgrade requests.
		streamOptions := translator.Options{
			Stdin:  attachOpts.Stdin,
			Stdout: attachOpts.Stdout,
			Stderr: attachOpts.Stderr,
			Tty:    attachOpts.TTY,
		}
		maxBytesPerSec := capabilities.Get().PerConnectionBandwidthLimitBytesPerSec
		streamtranslator := translator.NewStreamTranslatorHandler(location, transport, maxBytesPerSec, streamOptions)
		handler = translator.NewTranslatingHandler(handler, streamtranslator, wsstream.IsWebSocketRequestWithStreamCloseProtocol)
	}
	return handler, nil
}

// NewConnectOptions returns the versioned object that represents exec parameters
func (r *AttachREST) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodAttachOptions{}, false, ""
}

// ConnectMethods returns the methods supported by exec
func (r *AttachREST) ConnectMethods() []string {
	return upgradeableMethods
}

// ExecREST implements the exec subresource for a Pod
type ExecREST struct {
	Store       *genericregistry.Store
	KubeletConn client.ConnectionInfoGetter
	Authorizer  authorizer.Authorizer
}

// Implement Connecter
var _ = rest.Connecter(&ExecREST{})

// New creates a new podExecOptions object.
func (r *ExecREST) New() runtime.Object {
	return &api.PodExecOptions{}
}

// Destroy cleans up resources on shutdown.
func (r *ExecREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Connect returns a handler for the pod exec proxy
func (r *ExecREST) Connect(ctx context.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	// Forces a authz check for "create", if feature gate enabled.
	// See: https://github.com/kubernetes/kubernetes/issues/133515
	if utilfeature.DefaultFeatureGate.Enabled(features.AuthorizePodWebsocketUpgradeCreatePermission) {
		if err := ensureAuthorizedForVerb(ctx, r.Authorizer, "create"); err != nil {
			return nil, err
		}
	}

	execOpts, ok := opts.(*api.PodExecOptions)
	if !ok {
		return nil, fmt.Errorf("invalid options object: %#v", opts)
	}
	location, transport, err := pod.ExecLocation(ctx, r.Store, r.KubeletConn, name, execOpts)
	if err != nil {
		return nil, err
	}
	handler := newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder)
	if utilfeature.DefaultFeatureGate.Enabled(features.TranslateStreamCloseWebsocketRequests) {
		// Wrap the upgrade aware handler to implement stream translation
		// for WebSocket/V5 upgrade requests.
		streamOptions := translator.Options{
			Stdin:  execOpts.Stdin,
			Stdout: execOpts.Stdout,
			Stderr: execOpts.Stderr,
			Tty:    execOpts.TTY,
		}
		maxBytesPerSec := capabilities.Get().PerConnectionBandwidthLimitBytesPerSec
		streamtranslator := translator.NewStreamTranslatorHandler(location, transport, maxBytesPerSec, streamOptions)
		handler = translator.NewTranslatingHandler(handler, streamtranslator, wsstream.IsWebSocketRequestWithStreamCloseProtocol)
	}
	return handler, nil
}

// NewConnectOptions returns the versioned object that represents exec parameters
func (r *ExecREST) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodExecOptions{}, false, ""
}

// ConnectMethods returns the methods supported by exec
func (r *ExecREST) ConnectMethods() []string {
	return upgradeableMethods
}

// PortForwardREST implements the portforward subresource for a Pod
type PortForwardREST struct {
	Store       *genericregistry.Store
	KubeletConn client.ConnectionInfoGetter
	Authorizer  authorizer.Authorizer
}

// Implement Connecter
var _ = rest.Connecter(&PortForwardREST{})

// New returns an empty podPortForwardOptions object
func (r *PortForwardREST) New() runtime.Object {
	return &api.PodPortForwardOptions{}
}

// Destroy cleans up resources on shutdown.
func (r *PortForwardREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// NewConnectOptions returns the versioned object that represents the
// portforward parameters
func (r *PortForwardREST) NewConnectOptions() (runtime.Object, bool, string) {
	return &api.PodPortForwardOptions{}, false, ""
}

// ConnectMethods returns the methods supported by portforward
func (r *PortForwardREST) ConnectMethods() []string {
	return upgradeableMethods
}

// Connect returns a handler for the pod portforward proxy
func (r *PortForwardREST) Connect(ctx context.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	// Forces a authz check for "create", if feature gate enabled.
	// See: https://github.com/kubernetes/kubernetes/issues/133515
	if utilfeature.DefaultFeatureGate.Enabled(features.AuthorizePodWebsocketUpgradeCreatePermission) {
		if err := ensureAuthorizedForVerb(ctx, r.Authorizer, "create"); err != nil {
			return nil, err
		}
	}

	portForwardOpts, ok := opts.(*api.PodPortForwardOptions)
	if !ok {
		return nil, fmt.Errorf("invalid options object: %#v", opts)
	}
	location, transport, err := pod.PortForwardLocation(ctx, r.Store, r.KubeletConn, name, portForwardOpts)
	if err != nil {
		return nil, err
	}
	handler := newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder)
	if utilfeature.DefaultFeatureGate.Enabled(features.PortForwardWebsockets) {
		tunnelingHandler := translator.NewTunnelingHandler(handler)
		handler = translator.NewTranslatingHandler(handler, tunnelingHandler, wsstream.IsWebSocketRequestWithTunnelingProtocol)
	}
	return handler, nil
}

func newThrottledUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, wrapTransport, upgradeRequired bool, responder rest.Responder) http.Handler {
	handler := proxy.NewUpgradeAwareHandler(location, transport, wrapTransport, upgradeRequired, proxy.NewErrorResponder(responder))
	handler.MaxBytesPerSec = capabilities.Get().PerConnectionBandwidthLimitBytesPerSec
	return handler
}
