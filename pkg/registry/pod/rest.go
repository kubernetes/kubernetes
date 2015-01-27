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

package pod

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type PodStatusGetter interface {
	GetPodStatus(namespace, name string) (*api.PodStatus, error)
	ClearPodStatus(namespace, name string)
}

// REST implements the RESTStorage interface in terms of a PodRegistry.
type REST struct {
	podCache PodStatusGetter
	registry Registry
}

type RESTConfig struct {
	PodCache PodStatusGetter
	Registry Registry
}

// NewREST returns a new REST.
func NewREST(config *RESTConfig) *REST {
	return &REST{
		podCache: config.PodCache,
		registry: config.Registry,
	}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	pod := obj.(*api.Pod)
	if !api.ValidNamespace(ctx, &pod.ObjectMeta) {
		return nil, errors.NewConflict("pod", pod.Namespace, fmt.Errorf("Pod.Namespace does not match the provided context"))
	}
	api.FillObjectMetaSystemFields(ctx, &pod.ObjectMeta)
	if len(pod.Name) == 0 {
		// TODO properly handle auto-generated names.
		// See https://github.com/GoogleCloudPlatform/kubernetes/issues/148 170 & 1135
		pod.Name = string(pod.UID)
	}
	if errs := validation.ValidatePod(pod); len(errs) > 0 {
		return nil, errors.NewInvalid("pod", pod.Name, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := rs.registry.CreatePod(ctx, pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(ctx, pod.Name)
	}), nil
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		namespace, found := api.NamespaceFrom(ctx)
		if !found {
			return &api.Status{Status: api.StatusFailure}, nil
		}
		rs.podCache.ClearPodStatus(namespace, id)

		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeletePod(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	pod, err := rs.registry.GetPod(ctx, id)
	if err != nil {
		return pod, err
	}
	if pod == nil {
		return pod, nil
	}
	host := pod.Status.Host
	if status, err := rs.podCache.GetPodStatus(pod.Namespace, pod.Name); err != nil {
		pod.Status = api.PodStatus{
			Phase: api.PodUnknown,
		}
	} else {
		pod.Status = *status
	}
	// Make sure not to hide a recent host with an old one from the cache.
	// TODO: move host to spec
	pod.Status.Host = host
	return pod, err
}

func PodToSelectableFields(pod *api.Pod) labels.Set {

	// TODO we are populating both Status and DesiredState because selectors are not aware of API versions
	// see https://github.com/GoogleCloudPlatform/kubernetes/pull/2503

	var olderPodStatus v1beta1.PodStatus
	api.Scheme.Convert(pod.Status.Phase, &olderPodStatus)

	return labels.Set{
		"name":                pod.Name,
		"Status.Phase":        string(pod.Status.Phase),
		"Status.Host":         pod.Status.Host,
		"DesiredState.Status": string(olderPodStatus),
		"DesiredState.Host":   pod.Status.Host,
	}
}

// filterFunc returns a predicate based on label & field selectors that can be passed to registry's
// ListPods & WatchPods.
func (rs *REST) filterFunc(label, field labels.Selector) func(*api.Pod) bool {
	return func(pod *api.Pod) bool {
		fields := PodToSelectableFields(pod)
		return label.Matches(labels.Set(pod.Labels)) && field.Matches(fields)
	}
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	pods, err := rs.registry.ListPodsPredicate(ctx, rs.filterFunc(label, field))
	if err == nil {
		for i := range pods.Items {
			pod := &pods.Items[i]
			host := pod.Status.Host
			if status, err := rs.podCache.GetPodStatus(pod.Namespace, pod.Name); err != nil {
				pod.Status = api.PodStatus{
					Phase: api.PodUnknown,
				}
			} else {
				pod.Status = *status
			}
			// Make sure not to hide a recent host with an old one from the cache.
			// This is tested by the integration test.
			// TODO: move host to spec
			pod.Status.Host = host
		}
	}
	return pods, err
}

// Watch begins watching for new, changed, or deleted pods.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	// TODO: Add pod status to watch command
	return rs.registry.WatchPods(ctx, label, field, resourceVersion)
}

func (*REST) New() runtime.Object {
	return &api.Pod{}
}

func (*REST) NewList() runtime.Object {
	return &api.PodList{}
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	pod := obj.(*api.Pod)
	if !api.ValidNamespace(ctx, &pod.ObjectMeta) {
		return nil, errors.NewConflict("pod", pod.Namespace, fmt.Errorf("Pod.Namespace does not match the provided context"))
	}
	if errs := validation.ValidatePod(pod); len(errs) > 0 {
		return nil, errors.NewInvalid("pod", pod.Name, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := rs.registry.UpdatePod(ctx, pod); err != nil {
			return nil, err
		}
		return rs.registry.GetPod(ctx, pod.Name)
	}), nil
}

// ResourceLocation returns a URL to which one can send traffic for the specified pod.
func (rs *REST) ResourceLocation(ctx api.Context, id string) (string, error) {
	// Allow ID as "podname" or "podname:port".  If port is not specified,
	// try to use the first defined port on the pod.
	parts := strings.Split(id, ":")
	if len(parts) > 2 {
		return "", errors.NewBadRequest(fmt.Sprintf("invalid pod request %q", id))
	}
	name := parts[0]
	port := ""
	if len(parts) == 2 {
		// TODO: if port is not a number but a "(container)/(portname)", do a name lookup.
		port = parts[1]
	}

	obj, err := rs.Get(ctx, name)
	if err != nil {
		return "", err
	}
	pod := obj.(*api.Pod)
	if pod == nil {
		return "", nil
	}

	// Try to figure out a port.
	if port == "" {
		for i := range pod.Spec.Containers {
			if len(pod.Spec.Containers[i].Ports) > 0 {
				port = fmt.Sprintf("%d", pod.Spec.Containers[i].Ports[0].ContainerPort)
				break
			}
		}
	}

	// We leave off the scheme ('http://') because we have no idea what sort of server
	// is listening at this endpoint.
	loc := pod.Status.PodIP
	if port != "" {
		loc += fmt.Sprintf(":%s", port)
	}
	return loc, nil
}
