/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"context"
	"fmt"
	"sync"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	policygeneric "k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/client-go/tools/cache"
)

const crdGroupKindIndex = "paramKindGroupKind"

type crdParamKindResolver struct {
	lock                sync.RWMutex
	informer            cache.SharedIndexInformer
	handlerRegistration cache.ResourceEventHandlerRegistration
	informerExpected    bool
	// Retain GVKs once served by an established CRD so stale discovery cannot resurrect them after removal.
	knownGroupVersionKinds map[schema.GroupVersionKind]struct{}
	callbacks              map[uint64]func(schema.GroupKind)
	nextCallbackID         uint64
}

var _ policygeneric.ParamKindResolver = &crdParamKindResolver{}

func newCRDParamKindResolver() *crdParamKindResolver {
	return &crdParamKindResolver{
		knownGroupVersionKinds: map[schema.GroupVersionKind]struct{}{},
		callbacks:              map[uint64]func(schema.GroupKind){},
	}
}

func (r *crdParamKindResolver) ExpectCustomResourceDefinitionInformer() {
	r.lock.Lock()
	r.informerExpected = true
	r.lock.Unlock()
}

func (r *crdParamKindResolver) SetCustomResourceDefinitionInformer(informer apiextensionsinformers.CustomResourceDefinitionInformer) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if r.informer != nil {
		return fmt.Errorf("custom resource definition informer is already configured")
	}

	sharedInformer := informer.Informer()
	if err := sharedInformer.AddIndexers(cache.Indexers{crdGroupKindIndex: crdGroupKindIndexFunc}); err != nil {
		return fmt.Errorf("failed to add paramKind index to custom resource definition informer: %w", err)
	}
	handlerRegistration, err := sharedInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			r.notifyChanges(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			r.notifyChanges(oldObj, newObj)
		},
		DeleteFunc: func(obj interface{}) {
			r.notifyChanges(obj)
		},
	})
	if err != nil {
		return fmt.Errorf("failed to register custom resource definition event handler: %w", err)
	}
	r.informer = sharedInformer
	r.handlerRegistration = handlerRegistration
	r.informerExpected = true
	return nil
}

func (r *crdParamKindResolver) Resolve(_ context.Context, paramKind schema.GroupVersionKind) (*meta.RESTMapping, error) {
	if paramKind.Group == "" {
		return nil, nil
	}

	r.lock.RLock()
	informer := r.informer
	r.lock.RUnlock()
	if informer == nil {
		return nil, nil
	}

	objects, err := informer.GetIndexer().ByIndex(crdGroupKindIndex, groupKindIndexKey(paramKind.GroupKind()))
	if err != nil {
		return nil, err
	}

	for _, object := range objects {
		crd, ok := object.(*apiextensionsv1.CustomResourceDefinition)
		if !ok {
			return nil, fmt.Errorf("unexpected object type %T in custom resource definition informer", object)
		}
		if crd.Spec.Group != paramKind.Group || crd.Status.AcceptedNames.Kind != paramKind.Kind || !isCRDEstablished(crd) {
			continue
		}
		if !isCRDVersionServed(crd, paramKind.Version) {
			return nil, nil
		}

		scope := meta.RESTScopeRoot
		if crd.Spec.Scope == apiextensionsv1.NamespaceScoped {
			scope = meta.RESTScopeNamespace
		}
		return &meta.RESTMapping{
			Resource:         paramKind.GroupVersion().WithResource(crd.Status.AcceptedNames.Plural),
			GroupVersionKind: paramKind,
			Scope:            scope,
		}, nil
	}

	return nil, nil
}

func (r *crdParamKindResolver) Handles(paramKind schema.GroupVersionKind) bool {
	r.lock.RLock()
	defer r.lock.RUnlock()
	_, known := r.knownGroupVersionKinds[paramKind]
	return known
}

func (r *crdParamKindResolver) HasSynced() bool {
	r.lock.RLock()
	informer := r.informer
	handlerRegistration := r.handlerRegistration
	informerExpected := r.informerExpected
	r.lock.RUnlock()
	return (!informerExpected && informer == nil) || (handlerRegistration != nil && handlerRegistration.HasSynced())
}

func (r *crdParamKindResolver) RegisterForChanges(callback func(schema.GroupKind)) func() {
	r.lock.Lock()
	callbackID := r.nextCallbackID
	r.nextCallbackID++
	r.callbacks[callbackID] = callback
	r.lock.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			r.lock.Lock()
			delete(r.callbacks, callbackID)
			r.lock.Unlock()
		})
	}
}

func (r *crdParamKindResolver) notifyChanges(objects ...interface{}) {
	changedKinds := map[schema.GroupKind]struct{}{}
	authoritativeKinds := map[schema.GroupVersionKind]struct{}{}
	for _, object := range objects {
		crd, ok := object.(*apiextensionsv1.CustomResourceDefinition)
		if !ok {
			if tombstone, tombstoneOK := object.(cache.DeletedFinalStateUnknown); tombstoneOK {
				crd, ok = tombstone.Obj.(*apiextensionsv1.CustomResourceDefinition)
			}
		}
		if !ok {
			continue
		}
		if crd.Spec.Names.Kind != "" {
			changedKinds[schema.GroupKind{Group: crd.Spec.Group, Kind: crd.Spec.Names.Kind}] = struct{}{}
		}
		if crd.Status.AcceptedNames.Kind != "" {
			changedKinds[schema.GroupKind{Group: crd.Spec.Group, Kind: crd.Status.AcceptedNames.Kind}] = struct{}{}
		}
		if !isCRDEstablished(crd) || crd.Status.AcceptedNames.Kind == "" || crd.Status.AcceptedNames.Plural == "" {
			continue
		}
		for _, version := range crd.Spec.Versions {
			if version.Served {
				authoritativeKinds[schema.GroupVersionKind{
					Group:   crd.Spec.Group,
					Version: version.Name,
					Kind:    crd.Status.AcceptedNames.Kind,
				}] = struct{}{}
			}
		}
	}

	r.lock.Lock()
	for paramKind := range authoritativeKinds {
		r.knownGroupVersionKinds[paramKind] = struct{}{}
	}
	callbacks := make([]func(schema.GroupKind), 0, len(r.callbacks))
	for _, callback := range r.callbacks {
		callbacks = append(callbacks, callback)
	}
	r.lock.Unlock()

	for groupKind := range changedKinds {
		for _, callback := range callbacks {
			callback(groupKind)
		}
	}
}

func crdGroupKindIndexFunc(object interface{}) ([]string, error) {
	crd, ok := object.(*apiextensionsv1.CustomResourceDefinition)
	if !ok {
		return nil, fmt.Errorf("expected custom resource definition, got %T", object)
	}
	keys := map[string]struct{}{}
	if crd.Spec.Names.Kind != "" {
		keys[groupKindIndexKey(schema.GroupKind{Group: crd.Spec.Group, Kind: crd.Spec.Names.Kind})] = struct{}{}
	}
	if crd.Status.AcceptedNames.Kind != "" {
		keys[groupKindIndexKey(schema.GroupKind{Group: crd.Spec.Group, Kind: crd.Status.AcceptedNames.Kind})] = struct{}{}
	}
	result := make([]string, 0, len(keys))
	for key := range keys {
		result = append(result, key)
	}
	return result, nil
}

func groupKindIndexKey(groupKind schema.GroupKind) string {
	return groupKind.Group + "/" + groupKind.Kind
}

func isCRDEstablished(crd *apiextensionsv1.CustomResourceDefinition) bool {
	for _, condition := range crd.Status.Conditions {
		if condition.Type == apiextensionsv1.Established && condition.Status == apiextensionsv1.ConditionTrue {
			return true
		}
	}
	return false
}

func isCRDVersionServed(crd *apiextensionsv1.CustomResourceDefinition, version string) bool {
	for _, crdVersion := range crd.Spec.Versions {
		if crdVersion.Name == version {
			return crdVersion.Served
		}
	}
	return false
}
