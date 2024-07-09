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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	cmdfeaturegate "k8s.io/kubectl/pkg/cmd/util/featuregate"
	"k8s.io/utils/ptr"
)

// Selector is a Visitor for resources that match a label selector.
type Selector struct {
	Client        RESTClient
	Mapping       *meta.RESTMapping
	Namespace     string
	LabelSelector string
	FieldSelector string
	LimitChunks   int64
}

// NewSelector creates a resource selector which hides details of getting items by their label selector.
func NewSelector(client RESTClient, mapping *meta.RESTMapping, namespace, labelSelector, fieldSelector string, limitChunks int64) *Selector {
	return &Selector{
		Client:        client,
		Mapping:       mapping,
		Namespace:     namespace,
		LabelSelector: labelSelector,
		FieldSelector: fieldSelector,
		LimitChunks:   limitChunks,
	}
}

// Visit implements Visitor and uses request chunking by default.
func (r *Selector) Visit(fn VisitorFunc) error {
	helper := NewHelper(r.Client, r.Mapping)
	initialOpts := metav1.ListOptions{
		LabelSelector: r.LabelSelector,
		FieldSelector: r.FieldSelector,
		Limit:         r.LimitChunks,
	}

	// WatchList is disabled by default
	if cmdfeaturegate.WatchList.IsEnabled() {
		return r.watchList(fn, initialOpts, helper)
	}

	return FollowContinue(&initialOpts, func(options metav1.ListOptions) (runtime.Object, error) {
		list, err := helper.List(
			r.Namespace,
			r.ResourceMapping().GroupVersionKind.GroupVersion().String(),
			&options,
		)
		if err != nil {
			return nil, EnhanceListError(err, options, r.Mapping.Resource.String())
		}
		resourceVersion, _ := metadataAccessor.ResourceVersion(list)

		info := &Info{
			Client:  r.Client,
			Mapping: r.Mapping,

			Namespace:       r.Namespace,
			ResourceVersion: resourceVersion,

			Object: list,
		}

		if err := fn(info, nil); err != nil {
			return nil, err
		}
		return list, nil
	})
}

func (r *Selector) watchList(fn VisitorFunc, listOptions metav1.ListOptions, helper *Helper) error {
	listOptions.ResourceVersion = "" // request latest version
	listOptions.SendInitialEvents = ptr.To(true)
	listOptions.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	listOptions.AllowWatchBookmarks = true

	w, err := helper.Watch(
		r.Namespace,
		r.ResourceMapping().GroupVersionKind.GroupVersion().String(),
		&listOptions,
	)
	if err != nil {
		return err
	}

	var (
		resourceVersion string
		objects         []runtime.Object
		tables          []*unstructured.Unstructured
	)

	for event := range w.ResultChan() {
		resourceVersion, _ = metadataAccessor.ResourceVersion(event.Object)

		// first bookmark must be `metav1.InitialEventsAnnotationKey` event.
		if event.Type == watch.Bookmark {
			break
		}

		if t, ok := event.Object.(*unstructured.Unstructured); ok {
			tables = append(tables, t)
		} else {
			objects = append(objects, event.Object)
		}
	}

	infoTmpl := Info{
		Client:  r.Client,
		Mapping: r.Mapping,

		Namespace:       r.Namespace,
		ResourceVersion: resourceVersion,
	}

	if len(objects) > 0 {
		info := infoTmpl
		info.Object = toV1List(objects, resourceVersion)
		return fn(&info, nil)
	}

	for _, table := range tables {
		info := infoTmpl

		table.SetResourceVersion(resourceVersion)
		info.Object = table
		if err := fn(&info, nil); err != nil {
			return err
		}
	}

	return nil
}

func (r *Selector) Watch(resourceVersion string) (watch.Interface, error) {
	return NewHelper(r.Client, r.Mapping).Watch(r.Namespace, r.ResourceMapping().GroupVersionKind.GroupVersion().String(),
		&metav1.ListOptions{ResourceVersion: resourceVersion, LabelSelector: r.LabelSelector, FieldSelector: r.FieldSelector})
}

// ResourceMapping returns the mapping for this resource and implements ResourceMapping
func (r *Selector) ResourceMapping() *meta.RESTMapping {
	return r.Mapping
}
