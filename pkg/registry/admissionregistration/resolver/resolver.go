/*
Copyright 2022 The Kubernetes Authors.

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

package resolver

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

type ResourceResolver interface {
	Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error)
}

type discoveryResourceResolver struct {
	restMapper meta.RESTMapper
}

func (d *discoveryResourceResolver) Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	mapping, err := d.restMapper.RESTMapping(schema.GroupKind{
		Group: gvk.Group,
		Kind:  gvk.Kind,
	}, gvk.Version)
	if err != nil {
		return schema.GroupVersionResource{}, err
	}
	return mapping.Resource, nil
}

func NewDiscoveryResourceResolver(clientConfig *restclient.Config) (ResourceResolver, error) {
	clientset, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientset.Discovery())
	discoveryRESTMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	return &discoveryResourceResolver{restMapper: discoveryRESTMapper}, nil
}

type ResourceResolverFunc func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error)

func (f ResourceResolverFunc) Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	return f(gvk)
}
