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

package externalversions

import (
	"time"

	"k8s.io/apiextensions/pkg/client/clientset/clientset"
	externalversions "k8s.io/apiextensions/pkg/client/informers/externalversions"
	"k8s.io/apiextensions/pkg/client/informers/externalversions/internalinterfaces"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
)

//go:fix inline
type GenericInformer = externalversions.GenericInformer

//go:fix inline
type SharedInformerFactory = externalversions.SharedInformerFactory

//go:fix inline
type SharedInformerOption = externalversions.SharedInformerOption

// Deprecated: Please use NewSharedInformerFactoryWithOptions instead
//
//go:fix inline
func NewFilteredSharedInformerFactory(client clientset.Interface, defaultResync time.Duration, namespace string, tweakListOptions internalinterfaces.TweakListOptionsFunc) externalversions.SharedInformerFactory {
	return externalversions.NewFilteredSharedInformerFactory(client, defaultResync, namespace, tweakListOptions)
}

//go:fix inline
func NewSharedInformerFactory(client clientset.Interface, defaultResync time.Duration) externalversions.SharedInformerFactory {
	return externalversions.NewSharedInformerFactory(client, defaultResync)
}

//go:fix inline
func NewSharedInformerFactoryWithOptions(client clientset.Interface, defaultResync time.Duration, options ...externalversions.SharedInformerOption) externalversions.SharedInformerFactory {
	return externalversions.NewSharedInformerFactoryWithOptions(client, defaultResync, options...)
}

//go:fix inline
func WithCustomResyncConfig(resyncConfig map[v1.Object]time.Duration) externalversions.SharedInformerOption {
	return externalversions.WithCustomResyncConfig(resyncConfig)
}

//go:fix inline
func WithInformerName(informerName *cache.InformerName) externalversions.SharedInformerOption {
	return externalversions.WithInformerName(informerName)
}

//go:fix inline
func WithNamespace(namespace string) externalversions.SharedInformerOption {
	return externalversions.WithNamespace(namespace)
}

//go:fix inline
func WithTransform(transform cache.TransformFunc) externalversions.SharedInformerOption {
	return externalversions.WithTransform(transform)
}

//go:fix inline
func WithTweakListOptions(tweakListOptions internalinterfaces.TweakListOptionsFunc) externalversions.SharedInformerOption {
	return externalversions.WithTweakListOptions(tweakListOptions)
}
