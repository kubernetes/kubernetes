/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type GenericNamespacer interface {
	Generic(ns string, gvk unversioned.GroupVersionKind) GenericResourceInterface
}

// GenericResourceInterface provides a simple raw interface to arbitrary types
type GenericResourceInterface interface {
	List(opts api.ListOptions) ([]byte, error)
	Get(name string) ([]byte, error)
	Create(name string, obj interface{}) ([]byte, error)
	Update(name string, obj interface{}) ([]byte, error)
	Delete(name string) ([]byte, error)
}

func newGenericResources(c *ExtensionsClient, namespace string, gvk unversioned.GroupVersionKind) GenericResourceInterface {
	return &genericResources{
		namespace: namespace,
		gvk:       gvk,
		client:    c,
	}
}

type genericResources struct {
	namespace string
	gvk       unversioned.GroupVersionKind
	client    *ExtensionsClient
}

func (g *genericResources) List(opts api.ListOptions) ([]byte, error) {
	plural, _ := meta.KindToResource(g.gvk.Kind, false)
	return g.client.Get().
		AbsPath(fmt.Sprintf("/apis/%s/%s/%s", g.gvk.Group, g.gvk.Version, plural)).
		VersionedParams(&opts, api.Scheme).
		DoRaw()
}

func (g *genericResources) Get(name string) ([]byte, error) {
	plural, _ := meta.KindToResource(g.gvk.Kind, false)
	return g.client.Get().
		AbsPath(fmt.Sprintf("/apis/%s/%s/%s/%s", g.gvk.Group, g.gvk.Version, plural, name)).
		DoRaw()
}

func (g *genericResources) Create(name string, obj interface{}) ([]byte, error) {
	plural, _ := meta.KindToResource(g.gvk.Kind, false)
	data, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return g.client.Post().
		AbsPath(fmt.Sprintf("/apis/%s/%s/%s", g.gvk.Group, g.gvk.Version, plural)).
		Body(data).
		DoRaw()
}

func (g *genericResources) Update(name string, obj interface{}) ([]byte, error) {
	plural, _ := meta.KindToResource(g.gvk.Kind, false)
	data, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return g.client.Put().
		AbsPath(fmt.Sprintf("/apis/%s/%s/%s/%s", g.gvk.Group, g.gvk.Version, plural, name)).
		Body(data).
		DoRaw()
}

func (g *genericResources) Delete(name string) ([]byte, error) {
	plural, _ := meta.KindToResource(g.gvk.Kind, false)
	return g.client.Delete().
		AbsPath(fmt.Sprintf("/apis/%s/%s/%s/%s", g.gvk.Group, g.gvk.Version, plural, name)).
		DoRaw()
}
