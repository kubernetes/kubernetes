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
	"fmt"
	"net/http"
	"net/url"
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	genericrest "k8s.io/kubernetes/pkg/registry/generic/rest"
	"k8s.io/kubernetes/pkg/registry/pod"
	"k8s.io/kubernetes/pkg/runtime"
)

// ProxyREST implements the proxy subresource for a Pod
type ProxyREST struct {
	Store          *registry.Store
	ProxyTransport http.RoundTripper
}

// Implement Connecter
var _ = rest.Connecter(&ProxyREST{})

var proxyMethods = []string{"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}

// New returns an empty pod resource
func (r *ProxyREST) New() runtime.Object {
	return &api.Pod{}
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
func (r *ProxyREST) Connect(ctx api.Context, id string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	proxyOpts, ok := opts.(*api.PodProxyOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.ResourceLocation(r.Store, r.ProxyTransport, ctx, id)
	if err != nil {
		return nil, err
	}
	location.Path = path.Join(location.Path, proxyOpts.Path)
	// Return a proxy handler that uses the desired transport, wrapped with additional proxy handling (to get URL rewriting, X-Forwarded-* headers, etc)
	return newThrottledUpgradeAwareProxyHandler(location, transport, true, false, responder), nil
}

// Support both GET and POST methods. We must support GET for browsers that want to use WebSockets.
var upgradeableMethods = []string{"GET", "POST"}

// AttachREST implements the attach subresource for a Pod
type AttachREST struct {
	Store       *registry.Store
	KubeletConn client.ConnectionInfoGetter
}

// Implement Connecter
var _ = rest.Connecter(&AttachREST{})

// New creates a new Pod object
func (r *AttachREST) New() runtime.Object {
	return &api.Pod{}
}

// Connect returns a handler for the pod exec proxy
func (r *AttachREST) Connect(ctx api.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	attachOpts, ok := opts.(*api.PodAttachOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.AttachLocation(r.Store, r.KubeletConn, ctx, name, attachOpts)
	if err != nil {
		return nil, err
	}
	return newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder), nil
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
	Store       *registry.Store
	KubeletConn client.ConnectionInfoGetter
}

// Implement Connecter
var _ = rest.Connecter(&ExecREST{})

// New creates a new Pod object
func (r *ExecREST) New() runtime.Object {
	return &api.Pod{}
}

// Connect returns a handler for the pod exec proxy
func (r *ExecREST) Connect(ctx api.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	execOpts, ok := opts.(*api.PodExecOptions)
	if !ok {
		return nil, fmt.Errorf("invalid options object: %#v", opts)
	}
	location, transport, err := pod.ExecLocation(r.Store, r.KubeletConn, ctx, name, execOpts)
	if err != nil {
		return nil, err
	}
	return newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder), nil
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
	Store       *registry.Store
	KubeletConn client.ConnectionInfoGetter
}

// Implement Connecter
var _ = rest.Connecter(&PortForwardREST{})

// New returns an empty pod object
func (r *PortForwardREST) New() runtime.Object {
	return &api.Pod{}
}

// NewConnectOptions returns nil since portforward doesn't take additional parameters
func (r *PortForwardREST) NewConnectOptions() (runtime.Object, bool, string) {
	return nil, false, ""
}

// ConnectMethods returns the methods supported by portforward
func (r *PortForwardREST) ConnectMethods() []string {
	return upgradeableMethods
}

// Connect returns a handler for the pod portforward proxy
func (r *PortForwardREST) Connect(ctx api.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	location, transport, err := pod.PortForwardLocation(r.Store, r.KubeletConn, ctx, name)
	if err != nil {
		return nil, err
	}
	return newThrottledUpgradeAwareProxyHandler(location, transport, false, true, responder), nil
}

func newThrottledUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, wrapTransport, upgradeRequired bool, responder rest.Responder) *genericrest.UpgradeAwareProxyHandler {
	handler := genericrest.NewUpgradeAwareProxyHandler(location, transport, wrapTransport, upgradeRequired, responder)
	handler.MaxBytesPerSec = capabilities.Get().PerConnectionBandwidthLimitBytesPerSec
	return handler
}
