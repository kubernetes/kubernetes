/*
Copyright 2017 The Kubernetes Authors.

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

package customresource

import (
	"context"
	"fmt"
	"strings"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
)

// CustomResourceStorage includes dummy storage for CustomResources, and their Status and Scale subresources.
type CustomResourceStorage struct {
	CustomResource *REST
	Status         *StatusREST
	Scale          *ScaleREST
}

func NewStorage(resource schema.GroupResource, listKind schema.GroupVersionKind, strategy customResourceStrategy, optsGetter generic.RESTOptionsGetter, categories []string, tableConvertor rest.TableConvertor) CustomResourceStorage {
	customResourceREST, customResourceStatusREST := newREST(resource, listKind, strategy, optsGetter, categories, tableConvertor)

	s := CustomResourceStorage{
		CustomResource: customResourceREST,
	}

	if strategy.status != nil {
		s.Status = customResourceStatusREST
	}

	if scale := strategy.scale; scale != nil {
		var labelSelectorPath string
		if scale.LabelSelectorPath != nil {
			labelSelectorPath = *scale.LabelSelectorPath
		}

		s.Scale = &ScaleREST{
			store:              customResourceREST.Store,
			specReplicasPath:   scale.SpecReplicasPath,
			statusReplicasPath: scale.StatusReplicasPath,
			labelSelectorPath:  labelSelectorPath,
		}
	}

	return s
}

// REST implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
	categories []string
}

// newREST returns a RESTStorage object that will work against API services.
func newREST(resource schema.GroupResource, listKind schema.GroupVersionKind, strategy customResourceStrategy, optsGetter generic.RESTOptionsGetter, categories []string, tableConvertor rest.TableConvertor) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &unstructured.Unstructured{} },
		NewListFunc: func() runtime.Object {
			// lists are never stored, only manufactured, so stomp in the right kind
			ret := &unstructured.UnstructuredList{}
			ret.SetGroupVersionKind(listKind)
			return ret
		},
		PredicateFunc:            strategy.MatchCustomResourceDefinitionStorage,
		DefaultQualifiedResource: resource,

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,

		TableConvertor: tableConvertor,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: strategy.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = NewStatusStrategy(strategy)
	return &REST{store, categories}, &StatusREST{store: &statusStore}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// List returns a list of items matching labels and field according to the store's PredicateFunc.
func (e *REST) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	l, err := e.Store.List(ctx, options)
	if err != nil {
		return nil, err
	}

	// Shallow copy ObjectMeta in returned list for each item. Native types have `Items []Item` fields and therefore
	// implicitly shallow copy ObjectMeta. The generic store sets the self-link for each item. So this is necessary
	// to avoid mutation of the objects from the cache.
	if ul, ok := l.(*unstructured.UnstructuredList); ok {
		for i := range ul.Items {
			shallowCopyObjectMeta(&ul.Items[i])
		}
	}

	return l, nil
}

func (r *REST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	o, err := r.Store.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	if u, ok := o.(*unstructured.Unstructured); ok {
		shallowCopyObjectMeta(u)
	}
	return o, nil
}

func shallowCopyObjectMeta(u runtime.Unstructured) {
	obj := shallowMapDeepCopy(u.UnstructuredContent())
	if metadata, ok := obj["metadata"]; ok {
		if metadata, ok := metadata.(map[string]interface{}); ok {
			obj["metadata"] = shallowMapDeepCopy(metadata)
			u.SetUnstructuredContent(obj)
		}
	}
}

func shallowMapDeepCopy(in map[string]interface{}) map[string]interface{} {
	if in == nil {
		return nil
	}

	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = v
	}

	return out
}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return r.categories
}

// StatusREST implements the REST endpoint for changing the status of a CustomResource
type StatusREST struct {
	store *genericregistry.Store
}

var _ = rest.Patcher(&StatusREST{})

func (r *StatusREST) New() runtime.Object {
	return &unstructured.Unstructured{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

type ScaleREST struct {
	store              *genericregistry.Store
	specReplicasPath   string
	statusReplicasPath string
	labelSelectorPath  string
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscalingv1.Scale{}
}

func (r *ScaleREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.store.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	cr := obj.(*unstructured.Unstructured)

	scaleObject, replicasFound, err := scaleFromCustomResource(cr, r.specReplicasPath, r.statusReplicasPath, r.labelSelectorPath)
	if err != nil {
		return nil, err
	}
	if !replicasFound {
		return nil, apierrors.NewInternalError(fmt.Errorf("the spec replicas field %q does not exist", r.specReplicasPath))
	}
	return scaleObject, err
}

func (r *ScaleREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := r.store.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}
	cr := obj.(*unstructured.Unstructured)

	const invalidSpecReplicas = -2147483648 // smallest int32
	oldScale, replicasFound, err := scaleFromCustomResource(cr, r.specReplicasPath, r.statusReplicasPath, r.labelSelectorPath)
	if err != nil {
		return nil, false, err
	}
	if !replicasFound {
		oldScale.Spec.Replicas = invalidSpecReplicas // signal that this was not set before
	}

	obj, err = objInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, false, err
	}
	if obj == nil {
		return nil, false, apierrors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}

	scale, ok := obj.(*autoscalingv1.Scale)
	if !ok {
		return nil, false, apierrors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}

	if scale.Spec.Replicas == invalidSpecReplicas {
		return nil, false, apierrors.NewBadRequest(fmt.Sprintf("the spec replicas field %q cannot be empty", r.specReplicasPath))
	}

	specReplicasPath := strings.TrimPrefix(r.specReplicasPath, ".") // ignore leading period
	if err = unstructured.SetNestedField(cr.Object, int64(scale.Spec.Replicas), strings.Split(specReplicasPath, ".")...); err != nil {
		return nil, false, err
	}
	cr.SetResourceVersion(scale.ResourceVersion)

	obj, _, err = r.store.Update(ctx, cr.GetName(), rest.DefaultUpdatedObjectInfo(cr), createValidation, updateValidation, false, options)
	if err != nil {
		return nil, false, err
	}
	cr = obj.(*unstructured.Unstructured)

	newScale, _, err := scaleFromCustomResource(cr, r.specReplicasPath, r.statusReplicasPath, r.labelSelectorPath)
	if err != nil {
		return nil, false, apierrors.NewBadRequest(err.Error())
	}
	return newScale, false, err
}

// scaleFromCustomResource returns a scale subresource for a customresource and a bool signalling wether
// the specReplicas value was found.
func scaleFromCustomResource(cr *unstructured.Unstructured, specReplicasPath, statusReplicasPath, labelSelectorPath string) (*autoscalingv1.Scale, bool, error) {
	specReplicasPath = strings.TrimPrefix(specReplicasPath, ".") // ignore leading period
	specReplicas, foundSpecReplicas, err := unstructured.NestedInt64(cr.UnstructuredContent(), strings.Split(specReplicasPath, ".")...)
	if err != nil {
		return nil, false, err
	} else if !foundSpecReplicas {
		specReplicas = 0
	}

	statusReplicasPath = strings.TrimPrefix(statusReplicasPath, ".") // ignore leading period
	statusReplicas, found, err := unstructured.NestedInt64(cr.UnstructuredContent(), strings.Split(statusReplicasPath, ".")...)
	if err != nil {
		return nil, false, err
	} else if !found {
		statusReplicas = 0
	}

	var labelSelector string
	if len(labelSelectorPath) > 0 {
		labelSelectorPath = strings.TrimPrefix(labelSelectorPath, ".") // ignore leading period
		labelSelector, found, err = unstructured.NestedString(cr.UnstructuredContent(), strings.Split(labelSelectorPath, ".")...)
		if err != nil {
			return nil, false, err
		}
	}

	scale := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              cr.GetName(),
			Namespace:         cr.GetNamespace(),
			UID:               cr.GetUID(),
			ResourceVersion:   cr.GetResourceVersion(),
			CreationTimestamp: cr.GetCreationTimestamp(),
		},
		Spec: autoscalingv1.ScaleSpec{
			Replicas: int32(specReplicas),
		},
		Status: autoscalingv1.ScaleStatus{
			Replicas: int32(statusReplicas),
			Selector: labelSelector,
		},
	}

	return scale, foundSpecReplicas, nil
}
