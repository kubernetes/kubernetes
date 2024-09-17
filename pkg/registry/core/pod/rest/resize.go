/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/core"
)

// PortForwardREST implements the portforward subresource for a Pod
type ResizeREST struct {
	PodStore *genericregistry.Store
}

var _ = rest.GroupVersionKindProvider(&ResizeREST{})

func (r *ResizeREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return core.SchemeGroupVersion.WithKind("Resize")
}

// New returns an empty podPortForwardOptions object
func (r *ResizeREST) New() runtime.Object {
	return &core.Resize{}
}

// Destroy cleans up resources on shutdown.
func (r *ResizeREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

func (r *ResizeREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.PodStore.Get(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(core.Resource("resize"), name)
	}
	pod := obj.(*core.Pod)
	return resizeOptionFromPod(*pod), nil
}

func (r *ResizeREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, _ bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, _, err := r.PodStore.Update(
		ctx,
		name,
		&resizeUpdatedObjectInfo{name, objInfo},
		toResizeCreateValidation(createValidation),
		toResizeUpdateValidation(updateValidation),
		false,
		options,
	)
	if err != nil {
		return nil, false, err
	}
	pod := obj.(*core.Pod)
	return resizeOptionFromPod(*pod), false, nil
}

func toResizeCreateValidation(f rest.ValidateObjectFunc) rest.ValidateObjectFunc {
	return func(ctx context.Context, obj runtime.Object) error {
		return f(ctx, resizeOptionFromPod(*obj.(*core.Pod)))
	}
}

func toResizeUpdateValidation(f rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		return f(
			ctx,
			resizeOptionFromPod(*obj.(*core.Pod)),
			resizeOptionFromPod(*old.(*core.Pod)),
		)
	}
}

func resizeOptionFromPod(pod core.Pod) *core.Resize {
	containersSpec := map[string]core.ResourceRequirements{}
	for _, container := range pod.Spec.Containers {
		containersSpec[container.Name] = container.Resources
	}

	containersStatus := map[string]core.ResourceRequirements{}
	for _, container := range pod.Status.ContainerStatuses {
		if container.Resources == nil {
			continue
		}
		containersStatus[container.Name] = *container.Resources
	}

	return &core.Resize{
		ObjectMeta: metav1.ObjectMeta{
			Name:              pod.Name,
			Namespace:         pod.Namespace,
			UID:               pod.UID,
			ResourceVersion:   pod.ResourceVersion,
			CreationTimestamp: pod.CreationTimestamp,
		},
		Spec: core.ResizeOptionsSpec{
			Resize: containersSpec,
		},
		Status: core.ResizeOptionsStatus{
			Current: containersStatus,
		},
	}
}

// ValidateScale validates a Scale and returns an ErrorList with any errors.
func validateResize(resize *core.Resize) field.ErrorList {
	allErrs := field.ErrorList{}

	// TODO

	return allErrs
}

type resizeUpdatedObjectInfo struct {
	name       string
	reqObjInfo rest.UpdatedObjectInfo
}

func (i *resizeUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	return i.reqObjInfo.Preconditions()
}

func (i *resizeUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	pod, ok := oldObj.DeepCopyObject().(*core.Pod)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected existing object type to be Pod, got %T", pod))
	}
	// if zero-value, the existing object does not exist
	if len(pod.ResourceVersion) == 0 {
		return nil, errors.NewNotFound(core.Resource("pod/resize"), i.name)
	}

	// deployment -> old scale
	oldResize := resizeOptionFromPod(*pod)

	// old scale -> new scale
	newResizeObj, err := i.reqObjInfo.UpdatedObject(ctx, oldResize)
	if err != nil {
		return nil, err
	}
	if newResizeObj == nil {
		return nil, errors.NewBadRequest("nil update passed to Resize")
	}
	newResize, ok := newResizeObj.(*core.Resize)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected input object type to be Resize, but %T", newResizeObj))
	}

	// validate
	if errs := validateResize(newResize); len(errs) > 0 {
		return nil, errors.NewInvalid(autoscaling.Kind("Resize"), pod.Name, errs)
	}

	// validate precondition if specified (resourceVersion matching is handled by storage)
	if len(newResize.UID) > 0 && newResize.UID != pod.UID {
		return nil, errors.NewConflict(
			core.Resource("pod/resize"),
			pod.Name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", newResize.UID, pod.UID),
		)
	}

	containerNameToIdx := make(map[string]int, len(pod.Spec.Containers))
	for idx, container := range pod.Spec.Containers {
		containerNameToIdx[container.Name] = idx
	}

	for containerName, resources := range newResize.Spec.Resize {
		containerIdx := containerNameToIdx[containerName]
		pod.Spec.Containers[containerIdx].Resources = resources
	}
	pod.ResourceVersion = newResize.ResourceVersion

	//updatedEntries, err := managedFieldsHandler.ToParent(scale.ManagedFields)
	//if err != nil {
	//	return nil, err
	//}
	//deployment.ManagedFields = updatedEntries

	return pod, nil
}
