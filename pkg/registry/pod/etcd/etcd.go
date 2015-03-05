/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// rest implements a RESTStorage for pods against etcd
type REST struct {
	etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against pods.
func NewREST(h tools.EtcdHelper) (*REST, *BindingREST, *StatusREST) {
	prefix := "/registry/pods"
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

		Helper: h,
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

	return &REST{*store}, &BindingREST{store: store}, &StatusREST{store: &statusStore}
}

// ResourceLocation returns a pods location from its HostIP
func (r *REST) ResourceLocation(ctx api.Context, name string) (string, error) {
	return pod.ResourceLocation(r, ctx, name)
}

// WithPodStatus returns a rest object that decorates returned responses with extra
// status information.
func (r *REST) WithPodStatus(cache pod.PodStatusGetter) *REST {
	store := *r
	store.Decorator = pod.PodStatusDecorator(cache)
	store.AfterDelete = rest.AllFuncs(store.AfterDelete, pod.PodStatusReset(cache))
	return &store
}

// BindingREST implements the REST endpoint for binding pods to nodes when etcd is in use.
type BindingREST struct {
	store *etcdgeneric.Etcd
}

func (r *BindingREST) New() runtime.Object {
	return &api.Binding{}
}

// Create ensures a pod is bound to a specific host.
func (r *BindingREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	binding := obj.(*api.Binding)
	// TODO: move me to a binding strategy
	if len(binding.Target.Kind) != 0 && (binding.Target.Kind != "Node" && binding.Target.Kind != "Minion") {
		return nil, errors.NewInvalid("binding", binding.Name, errors.ValidationErrorList{errors.NewFieldInvalid("to.kind", binding.Target.Kind, "must be empty, 'Node', or 'Minion'")})
	}
	if len(binding.Target.Name) == 0 {
		return nil, errors.NewInvalid("binding", binding.Name, errors.ValidationErrorList{errors.NewFieldRequired("to.name")})
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
	err = r.store.Helper.AtomicUpdate(podKey, &api.Pod{}, false, func(obj runtime.Object) (runtime.Object, uint64, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, 0, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.Spec.Host != oldMachine || pod.Status.Host != oldMachine {
			return nil, 0, fmt.Errorf("pod %v is already assigned to host %q or %q", pod.Name, pod.Spec.Host, pod.Status.Host)
		}
		pod.Spec.Host = machine
		pod.Status.Host = machine
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			pod.Annotations[k] = v
		}
		finalPod = pod
		return pod, 0, nil
	})
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

func (r *StatusREST) New() runtime.Object {
	return &api.Pod{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}
