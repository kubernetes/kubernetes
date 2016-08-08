/*
Copyright 2014 The Kubernetes Authors.

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

package resource

import (
	"errors"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// Selector is a Visitor for resources that match a label selector.
type Selector struct {
	Client    RESTClient
	Mapping   *meta.RESTMapping
	Namespace string
	Selector  labels.Selector
	Export    bool
	Dynamic   bool
}

// NewSelector creates a resource selector which hides details of getting items by their label selector.
func NewSelector(client RESTClient, mapping *meta.RESTMapping, namespace string, selector labels.Selector, export bool, dynamic bool) *Selector {
	return &Selector{
		Client:    client,
		Mapping:   mapping,
		Namespace: namespace,
		Selector:  selector,
		Export:    export,
		Dynamic:   dynamic,
	}
}

// Visit implements Visitor
func (r *Selector) Visit(fn VisitorFunc) error {
	list, err := r.list()
	if err != nil {
		if apierrors.IsBadRequest(err) || apierrors.IsNotFound(err) {
			if se, ok := err.(*apierrors.StatusError); ok {
				// modify the message without hiding this is an API error
				if r.Selector.Empty() {
					se.ErrStatus.Message = fmt.Sprintf("Unable to list %q: %v", r.Mapping.Resource, se.ErrStatus.Message)
				} else {
					se.ErrStatus.Message = fmt.Sprintf("Unable to find %q that match the selector %q: %v", r.Mapping.Resource, r.Selector, se.ErrStatus.Message)
				}
				return se
			}
			if r.Selector.Empty() {
				return fmt.Errorf("Unable to list %q: %v", r.Mapping.Resource, err)
			} else {
				return fmt.Errorf("Unable to find %q that match the selector %q: %v", r.Mapping.Resource, r.Selector, err)
			}
		}
		return err
	}
	accessor := r.Mapping.MetadataAccessor
	resourceVersion, _ := accessor.ResourceVersion(list)
	info := &Info{
		Client:    r.Client,
		Mapping:   r.Mapping,
		Namespace: r.Namespace,

		Object:          list,
		ResourceVersion: resourceVersion,
		Dynamic:         r.Dynamic,
	}
	return fn(info, nil)
}

func (r *Selector) dynamicClient() (*dynamic.Client, error) {
	rc, ok := r.Client.(*restclient.RESTClient)
	if !ok {
		return nil, errors.New("could not build dynamic client from RESTClient")
	}

	dc, err := dynamic.FromRESTClient(rc)
	if err != nil {
		return nil, err
	}

	return dc.ParameterCodec(dynamic.VersionedParameterEncoderWithV1Fallback), nil
}

func (r *Selector) list() (runtime.Object, error) {
	if !r.Dynamic {
		return NewHelper(r.Client, r.Mapping).List(r.Namespace, r.ResourceMapping().GroupVersionKind.GroupVersion().String(), r.Selector, r.Export)
	}

	dc, err := r.dynamicClient()
	if err != nil {
		return nil, err
	}

	resource := &unversioned.APIResource{
		Name:       r.Mapping.Resource,
		Namespaced: r.Mapping.Scope.Name() == meta.RESTScopeNameNamespace,
		Kind:       r.Mapping.GroupVersionKind.Kind,
	}

	opts := &api.ListOptions{
		LabelSelector: r.Selector,
	}

	// TODO how to put ?export=true for handling r.Export
	return dc.Resource(resource, r.Namespace).List(opts)
}

func (r *Selector) Watch(resourceVersion string) (watch.Interface, error) {
	if !r.Dynamic {
		return NewHelper(r.Client, r.Mapping).Watch(r.Namespace, resourceVersion, r.ResourceMapping().GroupVersionKind.GroupVersion().String(), r.Selector)
	}

	dc, err := r.dynamicClient()
	if err != nil {
		return nil, err
	}

	resource := &unversioned.APIResource{
		Name:       r.Mapping.Resource,
		Namespaced: r.Mapping.Scope.Name() == meta.RESTScopeNameNamespace,
		Kind:       r.Mapping.GroupVersionKind.Kind,
	}

	opts := &api.ListOptions{
		LabelSelector:   r.Selector,
		ResourceVersion: resourceVersion,
		Watch:           true,
	}

	return dc.Resource(resource, r.Namespace).Watch(opts)
}

// ResourceMapping returns the mapping for this resource and implements ResourceMapping
func (r *Selector) ResourceMapping() *meta.RESTMapping {
	return r.Mapping
}
