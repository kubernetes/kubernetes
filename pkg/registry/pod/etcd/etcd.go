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
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/constraint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// rest implements a RESTStorage for pods against etcd
type REST struct {
	store *etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against pods.
func NewREST(h tools.EtcdHelper, factory pod.BoundPodFactory) (*REST, *BindingREST) {
	prefix := "/registry/pods"
	bindings := &podLifecycle{h}
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
		EndpointName: "pods",

		CreateStrategy: pod.Strategy,

		UpdateStrategy: pod.Strategy,
		AfterUpdate:    bindings.AfterUpdate,

		ReturnDeletedObject: true,
		AfterDelete:         bindings.AfterDelete,

		Helper: h,
	}
	return &REST{store: store}, &BindingREST{store: store, factory: factory}
}

// WithPodStatus returns a rest object that decorates returned responses with extra
// status information.
func (r *REST) WithPodStatus(cache pod.PodStatusGetter) *REST {
	store := *r.store
	store.Decorator = pod.PodStatusDecorator(cache)
	store.AfterDelete = rest.AllFuncs(store.AfterDelete, pod.PodStatusReset(cache))
	return &REST{store: &store}
}

// New returns a new object
func (r *REST) New() runtime.Object {
	return r.store.NewFunc()
}

// NewList returns a new list object
func (r *REST) NewList() runtime.Object {
	return r.store.NewListFunc()
}

// List obtains a list of pods with labels that match selector.
func (r *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return r.store.List(ctx, pod.MatchPod(label, field))
}

// Watch begins watching for new, changed, or deleted pods.
func (r *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return r.store.Watch(ctx, pod.MatchPod(label, field), resourceVersion)
}

// Get gets a specific pod specified by its ID.
func (r *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Create creates a pod based on a specification.
func (r *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	return r.store.Create(ctx, obj)
}

// Update changes a pod specification.
func (r *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// Delete deletes an existing pod specified by its ID.
func (r *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Delete(ctx, name)
}

// ResourceLocation returns a pods location from its HostIP
func (r *REST) ResourceLocation(ctx api.Context, name string) (string, error) {
	return pod.ResourceLocation(r, ctx, name)
}

func makeBoundPodsKey(machine string) string {
	return "/registry/nodes/" + machine + "/boundpods"
}

// BindingREST implements the REST endpoint for binding pods to nodes when etcd is in use.
type BindingREST struct {
	store   *etcdgeneric.Etcd
	factory pod.BoundPodFactory
}

func (r *BindingREST) New() runtime.Object {
	return &api.Binding{}
}

// Create ensures a pod is bound to a specific host.
func (r *BindingREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	binding := obj.(*api.Binding)
	err = r.assignPod(ctx, binding.PodID, binding.Host)
	err = etcderr.InterpretCreateError(err, "binding", "")
	out = &api.Status{Status: api.StatusSuccess}
	return
}

// setPodHostTo sets the given pod's host to 'machine' iff it was previously 'oldMachine'.
// Returns the current state of the pod, or an error.
func (r *BindingREST) setPodHostTo(ctx api.Context, podID, oldMachine, machine string) (finalPod *api.Pod, err error) {
	podKey, err := r.store.KeyFunc(ctx, podID)
	if err != nil {
		return nil, err
	}
	err = r.store.Helper.AtomicUpdate(podKey, &api.Pod{}, false, func(obj runtime.Object) (runtime.Object, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.Status.Host != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to host %v", pod.Name, pod.Status.Host)
		}
		pod.Status.Host = machine
		finalPod = pod
		return pod, nil
	})
	return finalPod, err
}

// assignPod assigns the given pod to the given machine.
func (r *BindingREST) assignPod(ctx api.Context, podID string, machine string) error {
	finalPod, err := r.setPodHostTo(ctx, podID, "", machine)
	if err != nil {
		return err
	}
	boundPod, err := r.factory.MakeBoundPod(machine, finalPod)
	if err != nil {
		return err
	}
	// Doing the constraint check this way provides atomicity guarantees.
	contKey := makeBoundPodsKey(machine)
	err = r.store.Helper.AtomicUpdate(contKey, &api.BoundPods{}, true, func(in runtime.Object) (runtime.Object, error) {
		boundPodList := in.(*api.BoundPods)
		boundPodList.Items = append(boundPodList.Items, *boundPod)
		if errors := constraint.Allowed(boundPodList.Items); len(errors) > 0 {
			return nil, fmt.Errorf("the assignment would cause the following constraints violation: %v", errors)
		}
		return boundPodList, nil
	})
	if err != nil {
		// Put the pod's host back the way it was. This is a terrible hack, but
		// can't really be helped, since there's not really a way to do atomic
		// multi-object changes in etcd.
		if _, err2 := r.setPodHostTo(ctx, podID, machine, ""); err2 != nil {
			glog.Errorf("Stranding pod %v; couldn't clear host after previous error: %v", podID, err2)
		}
	}
	return err
}

type podLifecycle struct {
	tools.EtcdHelper
}

func (h *podLifecycle) AfterUpdate(obj runtime.Object) error {
	pod := obj.(*api.Pod)
	if len(pod.Status.Host) == 0 {
		return nil
	}
	containerKey := makeBoundPodsKey(pod.Status.Host)
	return h.AtomicUpdate(containerKey, &api.BoundPods{}, true, func(in runtime.Object) (runtime.Object, error) {
		boundPods := in.(*api.BoundPods)
		for ix := range boundPods.Items {
			if boundPods.Items[ix].Name == pod.Name && boundPods.Items[ix].Namespace == pod.Namespace {
				boundPods.Items[ix].Spec = pod.Spec
				return boundPods, nil
			}
		}
		// This really shouldn't happen
		glog.Warningf("Couldn't find: %s in %#v", pod.Name, boundPods)
		return boundPods, fmt.Errorf("failed to update pod, couldn't find %s in %#v", pod.Name, boundPods)
	})
}

func (h *podLifecycle) AfterDelete(obj runtime.Object) error {
	pod := obj.(*api.Pod)
	if len(pod.Status.Host) == 0 {
		return nil
	}
	containerKey := makeBoundPodsKey(pod.Status.Host)
	return h.AtomicUpdate(containerKey, &api.BoundPods{}, true, func(in runtime.Object) (runtime.Object, error) {
		pods := in.(*api.BoundPods)
		newPods := make([]api.BoundPod, 0, len(pods.Items))
		found := false
		for _, boundPod := range pods.Items {
			if boundPod.Name != pod.Name || boundPod.Namespace != pod.Namespace {
				newPods = append(newPods, boundPod)
			} else {
				found = true
			}
		}
		if !found {
			// This really shouldn't happen, it indicates something is broken, and likely
			// there is a lost pod somewhere.
			// However it is "deleted" so log it and move on
			glog.Warningf("Couldn't find: %s in %#v", pod.Name, pods)
		}
		pods.Items = newPods
		return pods, nil
	})
}
