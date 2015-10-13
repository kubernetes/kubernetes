/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"fmt"
	"net/http"
	"net/url"
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	etcderr "k8s.io/kubernetes/pkg/api/errors/etcd"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/capabilities"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	genericrest "k8s.io/kubernetes/pkg/registry/generic/rest"
	"k8s.io/kubernetes/pkg/registry/pod"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// PodStorage includes storage for pods and all sub resources
type PodStorage struct {
	Pod         *REST
	Binding     *BindingREST
	Status      *StatusREST
	Log         *LogREST
	Proxy       *ProxyREST
	Exec        *ExecREST
	Attach      *AttachREST
	PortForward *PortForwardREST
}

// REST implements a RESTStorage for pods against etcd
type REST struct {
	*etcdgeneric.Etcd
	proxyTransport http.RoundTripper
}

// NewStorage returns a RESTStorage object that will work against pods.
func NewStorage(s storage.Interface, useCacher bool, k client.ConnectionInfoGetter, proxyTransport http.RoundTripper) PodStorage {
	prefix := "/pods"

	storageInterface := s
	if useCacher {
		config := storage.CacherConfig{
			CacheCapacity:  1000,
			Storage:        s,
			Type:           &api.Pod{},
			ResourcePrefix: prefix,
			KeyFunc: func(obj runtime.Object) (string, error) {
				return storage.NamespaceKeyFunc(prefix, obj)
			},
			NewListFunc: func() runtime.Object { return &api.PodList{} },
		}
		storageInterface = storage.NewCacher(config)
	}

	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Pod{} },
		NewListFunc: func() runtime.Object { return &api.PodList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Pod).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return pod.MatchPod(label, field)
		},
		EndpointName: "pods",

		CreateStrategy:      pod.Strategy,
		UpdateStrategy:      pod.Strategy,
		DeleteStrategy:      pod.Strategy,
		ReturnDeletedObject: true,

		Storage: storageInterface,
	}
	statusStore := *store

	statusStore.UpdateStrategy = pod.StatusStrategy

	return PodStorage{
		Pod:         &REST{store, proxyTransport},
		Binding:     &BindingREST{store: store},
		Status:      &StatusREST{store: &statusStore},
		Log:         &LogREST{store: store, kubeletConn: k},
		Proxy:       &ProxyREST{store: store, proxyTransport: proxyTransport},
		Exec:        &ExecREST{store: store, kubeletConn: k},
		Attach:      &AttachREST{store: store, kubeletConn: k},
		PortForward: &PortForwardREST{store: store, kubeletConn: k},
	}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a pods location from its HostIP
func (r *REST) ResourceLocation(ctx api.Context, name string) (*url.URL, http.RoundTripper, error) {
	return pod.ResourceLocation(r, r.proxyTransport, ctx, name)
}

// BindingREST implements the REST endpoint for binding pods to nodes when etcd is in use.
type BindingREST struct {
	store *etcdgeneric.Etcd
}

// New creates a new binding resource
func (r *BindingREST) New() runtime.Object {
	return &api.Binding{}
}

var _ = rest.Creater(&BindingREST{})

// Create ensures a pod is bound to a specific host.
func (r *BindingREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	binding := obj.(*api.Binding)
	// TODO: move me to a binding strategy
	if len(binding.Target.Kind) != 0 && (binding.Target.Kind != "Node" && binding.Target.Kind != "Minion") {
		return nil, errors.NewInvalid("binding", binding.Name, fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("to.kind", binding.Target.Kind, "must be empty, 'Node', or 'Minion'")})
	}
	if len(binding.Target.Name) == 0 {
		return nil, errors.NewInvalid("binding", binding.Name, fielderrors.ValidationErrorList{fielderrors.NewFieldRequired("to.name")})
	}
	err = r.assignPod(ctx, binding.Name, binding.Target.Name, binding.Annotations)
	out = &unversioned.Status{Status: unversioned.StatusSuccess}
	return
}

// setPodHostAndAnnotations sets the given pod's host to 'machine' if and only if it was
// previously 'oldMachine' and merges the provided annotations with those of the pod.
// Returns the current state of the pod, or an error.
func (r *BindingREST) setPodHostAndAnnotations(ctx api.Context, podID, oldMachine, machine string, annotations map[string]string) (finalPod *api.Pod, err error) {
	podKey, err := r.store.KeyFunc(ctx, podID)
	if err != nil {
		return nil, err
	}
	err = r.store.Storage.GuaranteedUpdate(ctx, podKey, &api.Pod{}, false, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.DeletionTimestamp != nil {
			return nil, fmt.Errorf("pod %s is being deleted, cannot be assigned to a host", pod.Name)
		}
		if pod.Spec.NodeName != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to node %q", pod.Name, pod.Spec.NodeName)
		}
		pod.Spec.NodeName = machine
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			pod.Annotations[k] = v
		}
		finalPod = pod
		return pod, nil
	}))
	return finalPod, err
}

// assignPod assigns the given pod to the given machine.
func (r *BindingREST) assignPod(ctx api.Context, podID string, machine string, annotations map[string]string) (err error) {
	if _, err = r.setPodHostAndAnnotations(ctx, podID, "", machine, annotations); err != nil {
		err = etcderr.InterpretGetError(err, "pod", podID)
		err = etcderr.InterpretUpdateError(err, "pod", podID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewConflict("binding", podID, err)
		}
	}
	return
}

// StatusREST implements the REST endpoint for changing the status of a pod.
type StatusREST struct {
	store *etcdgeneric.Etcd
}

// New creates a new pod resource
func (r *StatusREST) New() runtime.Object {
	return &api.Pod{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// LogREST implements the log endpoint for a Pod
// TODO: move me into pod/rest - I'm generic to store type via ResourceGetter
type LogREST struct {
	store       *etcdgeneric.Etcd
	kubeletConn client.ConnectionInfoGetter
}

// LogREST implements GetterWithOptions
var _ = rest.GetterWithOptions(&LogREST{})

// New creates a new Pod log options object
func (r *LogREST) New() runtime.Object {
	// TODO - return a resource that represents a log
	return &api.Pod{}
}

// Get retrieves a runtime.Object that will stream the contents of the pod log
func (r *LogREST) Get(ctx api.Context, name string, opts runtime.Object) (runtime.Object, error) {
	logOpts, ok := opts.(*api.PodLogOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	if errs := validation.ValidatePodLogOptions(logOpts); len(errs) > 0 {
		return nil, errors.NewInvalid("podlogs", name, errs)
	}
	location, transport, err := pod.LogLocation(r.store, r.kubeletConn, ctx, name, logOpts)
	if err != nil {
		return nil, err
	}
	return &genericrest.LocationStreamer{
		Location:        location,
		Transport:       transport,
		ContentType:     "text/plain",
		Flush:           logOpts.Follow,
		ResponseChecker: genericrest.NewGenericHttpResponseChecker("Pod", name),
	}, nil
}

// NewGetOptions creates a new options object
func (r *LogREST) NewGetOptions() (runtime.Object, bool, string) {
	return &api.PodLogOptions{}, false, ""
}

// ProxyREST implements the proxy subresource for a Pod
// TODO: move me into pod/rest - I'm generic to store type via ResourceGetter
type ProxyREST struct {
	store          *etcdgeneric.Etcd
	proxyTransport http.RoundTripper
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
	location, transport, err := pod.ResourceLocation(r.store, r.proxyTransport, ctx, id)
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
// TODO: move me into pod/rest - I'm generic to store type via ResourceGetter
type AttachREST struct {
	store       *etcdgeneric.Etcd
	kubeletConn client.ConnectionInfoGetter
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
	location, transport, err := pod.AttachLocation(r.store, r.kubeletConn, ctx, name, attachOpts)
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
// TODO: move me into pod/rest - I'm generic to store type via ResourceGetter
type ExecREST struct {
	store       *etcdgeneric.Etcd
	kubeletConn client.ConnectionInfoGetter
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
	location, transport, err := pod.ExecLocation(r.store, r.kubeletConn, ctx, name, execOpts)
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
// TODO: move me into pod/rest - I'm generic to store type via ResourceGetter
type PortForwardREST struct {
	store       *etcdgeneric.Etcd
	kubeletConn client.ConnectionInfoGetter
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
	location, transport, err := pod.PortForwardLocation(r.store, r.kubeletConn, ctx, name)
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
