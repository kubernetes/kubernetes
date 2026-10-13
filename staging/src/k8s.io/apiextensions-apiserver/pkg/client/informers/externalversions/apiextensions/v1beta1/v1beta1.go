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

package v1beta1

import (
	"time"

	"k8s.io/apiextensions/pkg/client/clientset/clientset"
	v1beta1 "k8s.io/apiextensions/pkg/client/informers/externalversions/apiextensions/v1beta1"
	"k8s.io/apiextensions/pkg/client/informers/externalversions/internalinterfaces"
	"k8s.io/client-go/tools/cache"
)

//go:fix inline
type CustomResourceDefinitionDetailedHandlerFuncs = v1beta1.CustomResourceDefinitionDetailedHandlerFuncs

//go:fix inline
type CustomResourceDefinitionFilteringHandler = v1beta1.CustomResourceDefinitionFilteringHandler

//go:fix inline
type CustomResourceDefinitionHandlerFuncs = v1beta1.CustomResourceDefinitionHandlerFuncs

//go:fix inline
type CustomResourceDefinitionIndexInformer = v1beta1.CustomResourceDefinitionIndexInformer

//go:fix inline
type CustomResourceDefinitionIndexers = v1beta1.CustomResourceDefinitionIndexers

//go:fix inline
type CustomResourceDefinitionInformer = v1beta1.CustomResourceDefinitionInformer

//go:fix inline
type DeletedCustomResourceDefinition = v1beta1.DeletedCustomResourceDefinition

//go:fix inline
type Interface = v1beta1.Interface

//go:fix inline
type TypedCustomResourceDefinitionInformer = v1beta1.TypedCustomResourceDefinitionInformer

//go:fix inline
func New(f internalinterfaces.SharedInformerFactory, namespace string, tweakListOptions internalinterfaces.TweakListOptionsFunc) v1beta1.Interface {
	return v1beta1.New(f, namespace, tweakListOptions)
}

//go:fix inline
func NewCustomResourceDefinitionInformer(client clientset.Interface, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return v1beta1.NewCustomResourceDefinitionInformer(client, resyncPeriod, indexers)
}

//go:fix inline
func NewCustomResourceDefinitionInformerWithOptions(client clientset.Interface, options internalinterfaces.InformerOptions) cache.SharedIndexInformer {
	return v1beta1.NewCustomResourceDefinitionInformerWithOptions(client, options)
}

//go:fix inline
func NewFilteredCustomResourceDefinitionInformer(client clientset.Interface, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return v1beta1.NewFilteredCustomResourceDefinitionInformer(client, resyncPeriod, indexers, tweakListOptions)
}

//go:fix inline
func NewTypedCustomResourceDefinitionInformer(client clientset.Interface, resyncPeriod time.Duration, indexers v1beta1.CustomResourceDefinitionIndexers) v1beta1.CustomResourceDefinitionIndexInformer {
	return v1beta1.NewTypedCustomResourceDefinitionInformer(client, resyncPeriod, indexers)
}

//go:fix inline
func NewTypedCustomResourceDefinitionInformerWithOptions(client clientset.Interface, options internalinterfaces.InformerOptions) v1beta1.CustomResourceDefinitionIndexInformer {
	return v1beta1.NewTypedCustomResourceDefinitionInformerWithOptions(client, options)
}

//go:fix inline
func NewTypedFilteredCustomResourceDefinitionInformer(client clientset.Interface, resyncPeriod time.Duration, indexers v1beta1.CustomResourceDefinitionIndexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) v1beta1.CustomResourceDefinitionIndexInformer {
	return v1beta1.NewTypedFilteredCustomResourceDefinitionInformer(client, resyncPeriod, indexers, tweakListOptions)
}

//go:fix inline
func ToCustomResourceDefinitionIndexInformer(informer cache.SharedIndexInformer) v1beta1.CustomResourceDefinitionIndexInformer {
	return v1beta1.ToCustomResourceDefinitionIndexInformer(informer)
}

//go:fix inline
func ToTypedCustomResourceDefinitionInformer(informer v1beta1.CustomResourceDefinitionInformer) v1beta1.TypedCustomResourceDefinitionInformer {
	return v1beta1.ToTypedCustomResourceDefinitionInformer(informer)
}
