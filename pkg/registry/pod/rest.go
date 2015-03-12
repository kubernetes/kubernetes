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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// podStrategy implements behavior for Pods
// TODO: move to a pod specific package.
type podStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Pod
// objects via the REST API.
// TODO: Create other strategies for updating status, bindings, etc
var Strategy = podStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for pods.
func (podStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (podStrategy) ResetBeforeCreate(obj runtime.Object) {
	pod := obj.(*api.Pod)
	pod.Status = api.PodStatus{
		Host:  pod.Spec.Host,
		Phase: api.PodPending,
	}
}

// Validate validates a new pod.
func (podStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	pod := obj.(*api.Pod)
	return validation.ValidatePod(pod)
}

// AllowCreateOnUpdate is false for pods.
func (podStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	return validation.ValidatePodUpdate(obj.(*api.Pod), old.(*api.Pod))
}

type podStatusStrategy struct {
	podStrategy
}

var StatusStrategy = podStatusStrategy{Strategy}

func (podStatusStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	// TODO: merge valid fields after update
	return validation.ValidatePodStatusUpdate(obj.(*api.Pod), old.(*api.Pod))
}

// PodStatusGetter is an interface used by Pods to fetch and retrieve status info.
type PodStatusGetter interface {
	GetPodStatus(namespace, name string) (*api.PodStatus, error)
	ClearPodStatus(namespace, name string)
}

// PodStatusDecorator returns a function that updates pod.Status based
// on the provided pod cache.
func PodStatusDecorator(cache PodStatusGetter) rest.ObjectFunc {
	return func(obj runtime.Object) error {
		pod := obj.(*api.Pod)
		host := pod.Status.Host
		if status, err := cache.GetPodStatus(pod.Namespace, pod.Name); err != nil {
			pod.Status = api.PodStatus{
				Phase: api.PodUnknown,
			}
		} else {
			pod.Status = *status
		}
		pod.Status.Host = host
		return nil
	}
}

// PodStatusReset returns a function that clears the pod cache when the object
// is deleted.
func PodStatusReset(cache PodStatusGetter) rest.ObjectFunc {
	return func(obj runtime.Object) error {
		pod := obj.(*api.Pod)
		cache.ClearPodStatus(pod.Namespace, pod.Name)
		return nil
	}
}

// MatchPod returns a generic matcher for a given label and field selector.
func MatchPod(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		podObj, ok := obj.(*api.Pod)
		if !ok {
			return false, fmt.Errorf("not a pod")
		}
		fields := PodToSelectableFields(podObj)
		return label.Matches(labels.Set(podObj.Labels)) && field.Matches(fields), nil
	})
}

// PodToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *api.Pod) labels.Set {
	return labels.Set{
		"name":         pod.Name,
		"status.phase": string(pod.Status.Phase),
		"status.host":  pod.Status.Host,
	}
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(api.Context, string) (runtime.Object, error)
}

// ResourceLocation returns a URL to which one can send traffic for the specified pod.
func ResourceLocation(getter ResourceGetter, ctx api.Context, id string) (string, error) {
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

	obj, err := getter.Get(ctx, name)
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
