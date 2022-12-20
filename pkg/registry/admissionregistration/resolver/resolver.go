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
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
)

type ResourceResolver interface {
	Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error)
}

type discoveryResourceResolver struct {
	client discovery.DiscoveryInterface
}

func (d *discoveryResourceResolver) Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	gv := gvk.GroupVersion()
	// TODO: refactor this into an efficient gvk --> gvr resolver that remembers hits and re-resolves group/version info on misses
	resources, err := d.client.ServerResourcesForGroupVersion(gv.String())
	if err != nil {
		return schema.GroupVersionResource{}, err
	}
	for _, resource := range resources.APIResources {
		if resource.Kind != gvk.Kind {
			// ignore unrelated kinds
			continue
		}
		if strings.Contains(resource.Name, "/") {
			// ignore subresources
			continue
		}
		if resource.Group != "" && resource.Group != gvk.Group {
			// group didn't match
			continue
		}
		if resource.Version != "" && resource.Version != gvk.Version {
			// version didn't match
			continue
		}
		return gv.WithResource(resource.Name), nil
	}
	return schema.GroupVersionResource{}, &meta.NoKindMatchError{GroupKind: gvk.GroupKind(), SearchedVersions: []string{gvk.Version}}
}

func NewDiscoveryResourceResolver(client discovery.DiscoveryInterface) (ResourceResolver, error) {
	return &discoveryResourceResolver{client: client}, nil
}

type ResourceResolverFunc func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error)

func (f ResourceResolverFunc) Resolve(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	return f(gvk)
}
