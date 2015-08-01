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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	genericrest "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
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
	etcdgeneric.Etcd
}

// NewStorage returns a RESTStorage object that will work against pods.
func NewStorage(s storage.Interface, k client.ConnectionInfoGetter) PodStorage {
	prefix := "/pods"
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

		Storage: s,
	}
	statusStore := *store

	bindings := &podLifecycle{}
	store.CreateStrategy = pod.Strategy
	store.UpdateStrategy = pod.Strategy
	store.AfterUpdate = bindings.AfterUpdate
	store.DeleteStrategy = pod.Strategy
	store.ReturnDeletedObject = true
	store.AfterDelete = bindings.AfterDelete

	statusStore.UpdateStrategy = pod.StatusStrategy

	return PodStorage{
		Pod:         &REST{*store},
		Binding:     &BindingREST{store: store},
		Status:      &StatusREST{store: &statusStore},
		Log:         &LogREST{store: store, kubeletConn: k},
		Proxy:       &ProxyREST{store: store},
		Exec:        &ExecREST{store: store, kubeletConn: k},
		Attach:      &AttachREST{store: store, kubeletConn: k},
		PortForward: &PortForwardREST{store: store, kubeletConn: k},
	}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a pods location from its HostIP
func (r *REST) ResourceLocation(ctx api.Context, name string) (*url.URL, http.RoundTripper, error) {
	return pod.ResourceLocation(r, ctx, name)
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
	out = &api.Status{Status: api.StatusSuccess}
	return
}

// setPodHostAndAnnotations sets the given pod's host to 'machine' iff it was previously 'oldMachine' and merges
// the provided annotations with those of the pod.
// Returns the current state of the pod, or an error.
func (r *BindingREST) setPodHostAndAnnotations(ctx api.Context, podID, oldMachine, machine string, annotations map[string]string) (finalPod *api.Pod, err error) {
	podKey, err := r.store.KeyFunc(ctx, podID)
	if err != nil {
		return nil, err
	}
	err = r.store.Storage.GuaranteedUpdate(podKey, &api.Pod{}, false, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
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

type podLifecycle struct{}

func (h *podLifecycle) AfterUpdate(obj runtime.Object) error {
	return nil
}

func (h *podLifecycle) AfterDelete(obj runtime.Object) error {
	return nil
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
	location, transport, err := pod.LogLocation(r.store, r.kubeletConn, ctx, name, logOpts)
	if err != nil {
		return nil, err
	}
	return &genericrest.LocationStreamer{
		Location:    location,
		Transport:   transport,
		ContentType: "text/plain",
		Flush:       logOpts.Follow,
	}, nil
}

// NewGetOptions creates a new options object
func (r *LogREST) NewGetOptions() (runtime.Object, bool, string) {
	return &api.PodLogOptions{}, false, ""
}

// ProxyREST implements the proxy subresource for a Pod
type ProxyREST struct {
	store *etcdgeneric.Etcd
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
func (r *ProxyREST) Connect(ctx api.Context, id string, opts runtime.Object) (rest.ConnectHandler, error) {
	proxyOpts, ok := opts.(*api.PodProxyOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, _, err := pod.ResourceLocation(r.store, ctx, id)
	if err != nil {
		return nil, err
	}
	location.Path = path.Join(location.Path, proxyOpts.Path)
	return newUpgradeAwareProxyHandler(location, nil, false), nil
}

// Support both GET and POST methods. Over time, we want to move all clients to start using POST and then stop supporting GET.
var upgradeableMethods = []string{"GET", "POST"}

// AttachREST implements the attach subresource for a Pod
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
func (r *AttachREST) Connect(ctx api.Context, name string, opts runtime.Object) (rest.ConnectHandler, error) {
	attachOpts, ok := opts.(*api.PodAttachOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.AttachLocation(r.store, r.kubeletConn, ctx, name, attachOpts)
	if err != nil {
		return nil, err
	}
	return genericrest.NewUpgradeAwareProxyHandler(location, transport, true), nil
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
func (r *ExecREST) Connect(ctx api.Context, name string, opts runtime.Object) (rest.ConnectHandler, error) {
	execOpts, ok := opts.(*api.PodExecOptions)
	if !ok {
		return nil, fmt.Errorf("Invalid options object: %#v", opts)
	}
	location, transport, err := pod.ExecLocation(r.store, r.kubeletConn, ctx, name, execOpts)
	if err != nil {
		return nil, err
	}
	return newUpgradeAwareProxyHandler(location, transport, true), nil
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
func (r *PortForwardREST) Connect(ctx api.Context, name string, opts runtime.Object) (rest.ConnectHandler, error) {
	location, transport, err := pod.PortForwardLocation(r.store, r.kubeletConn, ctx, name)
	if err != nil {
		return nil, err
	}
	return newUpgradeAwareProxyHandler(location, transport, true), nil
}

func newUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, upgradeRequired bool) *genericrest.UpgradeAwareProxyHandler {
	handler := genericrest.NewUpgradeAwareProxyHandler(location, transport, upgradeRequired)
	handler.MaxBytesPerSec = capabilities.Get().PerConnectionBandwidthLimitBytesPerSec
	return handler
}
