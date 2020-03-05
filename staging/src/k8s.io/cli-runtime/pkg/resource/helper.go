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
	"context"
	"strconv"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
)

var metadataAccessor = meta.NewAccessor()

// Helper provides methods for retrieving or mutating a RESTful
// resource.
type Helper struct {
	// The name of this resource as the server would recognize it
	Resource string
	// A RESTClient capable of mutating this resource.
	RESTClient RESTClient
	// True if the resource type is scoped to namespaces
	NamespaceScoped bool
	// If true, then use server-side dry-run to not persist changes to storage
	// for verbs and resources that support server-side dry-run.
	//
	// Note this should only be used against an apiserver with dry-run enabled,
	// and on resources that support dry-run. If the apiserver or the resource
	// does not support dry-run, then the change will be persisted to storage.
	ServerDryRun bool

	// FieldManager is the name associated with the actor or entity that is making
	// changes.
	FieldManager string
}

// NewHelper creates a Helper from a ResourceMapping
func NewHelper(client RESTClient, mapping *meta.RESTMapping) *Helper {
	return &Helper{
		Resource:        mapping.Resource.Resource,
		RESTClient:      client,
		NamespaceScoped: mapping.Scope.Name() == meta.RESTScopeNameNamespace,
	}
}

// DryRun, if true, will use server-side dry-run to not persist changes to storage.
// Otherwise, changes will be persisted to storage.
func (m *Helper) DryRun(dryRun bool) *Helper {
	m.ServerDryRun = dryRun
	return m
}

// WithFieldManager sets the field manager option to indicate the actor or entity
// that is making changes in a create or update operation.
func (m *Helper) WithFieldManager(fieldManager string) *Helper {
	m.FieldManager = fieldManager
	return m
}

func (m *Helper) Get(namespace, name string, export bool) (runtime.Object, error) {
	req := m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name)
	if export {
		// TODO: I should be part of GetOptions
		req.Param("export", strconv.FormatBool(export))
	}
	return req.Do(context.TODO()).Get()
}

func (m *Helper) List(namespace, apiVersion string, export bool, options *metav1.ListOptions) (runtime.Object, error) {
	req := m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		VersionedParams(options, metav1.ParameterCodec)
	if export {
		// TODO: I should be part of ListOptions
		req.Param("export", strconv.FormatBool(export))
	}
	return req.Do(context.TODO()).Get()
}

func (m *Helper) Watch(namespace, apiVersion string, options *metav1.ListOptions) (watch.Interface, error) {
	options.Watch = true
	return m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		VersionedParams(options, metav1.ParameterCodec).
		Watch(context.TODO())
}

func (m *Helper) WatchSingle(namespace, name, resourceVersion string) (watch.Interface, error) {
	return m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		VersionedParams(&metav1.ListOptions{
			ResourceVersion: resourceVersion,
			Watch:           true,
			FieldSelector:   fields.OneTermEqualSelector("metadata.name", name).String(),
		}, metav1.ParameterCodec).
		Watch(context.TODO())
}

func (m *Helper) Delete(namespace, name string) (runtime.Object, error) {
	return m.DeleteWithOptions(namespace, name, nil)
}

func (m *Helper) DeleteWithOptions(namespace, name string, options *metav1.DeleteOptions) (runtime.Object, error) {
	if options == nil {
		options = &metav1.DeleteOptions{}
	}
	if m.ServerDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}

	return m.RESTClient.Delete().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name).
		Body(options).
		Do(context.TODO()).
		Get()
}

func (m *Helper) Create(namespace string, modify bool, obj runtime.Object) (runtime.Object, error) {
	return m.CreateWithOptions(namespace, modify, obj, nil)
}

func (m *Helper) CreateWithOptions(namespace string, modify bool, obj runtime.Object, options *metav1.CreateOptions) (runtime.Object, error) {
	if options == nil {
		options = &metav1.CreateOptions{}
	}
	if m.ServerDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	if m.FieldManager != "" {
		options.FieldManager = m.FieldManager
	}
	if modify {
		// Attempt to version the object based on client logic.
		version, err := metadataAccessor.ResourceVersion(obj)
		if err != nil {
			// We don't know how to clear the version on this object, so send it to the server as is
			return m.createResource(m.RESTClient, m.Resource, namespace, obj, options)
		}
		if version != "" {
			if err := metadataAccessor.SetResourceVersion(obj, ""); err != nil {
				return nil, err
			}
		}
	}

	return m.createResource(m.RESTClient, m.Resource, namespace, obj, options)
}

func (m *Helper) createResource(c RESTClient, resource, namespace string, obj runtime.Object, options *metav1.CreateOptions) (runtime.Object, error) {
	return c.Post().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(resource).
		VersionedParams(options, metav1.ParameterCodec).
		Body(obj).
		Do(context.TODO()).
		Get()
}
func (m *Helper) Patch(namespace, name string, pt types.PatchType, data []byte, options *metav1.PatchOptions) (runtime.Object, error) {
	if options == nil {
		options = &metav1.PatchOptions{}
	}
	if m.ServerDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	if m.FieldManager != "" {
		options.FieldManager = m.FieldManager
	}
	return m.RESTClient.Patch(pt).
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name).
		VersionedParams(options, metav1.ParameterCodec).
		Body(data).
		Do(context.TODO()).
		Get()
}

func (m *Helper) Replace(namespace, name string, overwrite bool, obj runtime.Object) (runtime.Object, error) {
	c := m.RESTClient
	var options = &metav1.UpdateOptions{}
	if m.ServerDryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	if m.FieldManager != "" {
		options.FieldManager = m.FieldManager
	}

	// Attempt to version the object based on client logic.
	version, err := metadataAccessor.ResourceVersion(obj)
	if err != nil {
		// We don't know how to version this object, so send it to the server as is
		return m.replaceResource(c, m.Resource, namespace, name, obj, options)
	}
	if version == "" && overwrite {
		// Retrieve the current version of the object to overwrite the server object
		serverObj, err := c.Get().NamespaceIfScoped(namespace, m.NamespaceScoped).Resource(m.Resource).Name(name).Do(context.TODO()).Get()
		if err != nil {
			// The object does not exist, but we want it to be created
			return m.replaceResource(c, m.Resource, namespace, name, obj, options)
		}
		serverVersion, err := metadataAccessor.ResourceVersion(serverObj)
		if err != nil {
			return nil, err
		}
		if err := metadataAccessor.SetResourceVersion(obj, serverVersion); err != nil {
			return nil, err
		}
	}

	return m.replaceResource(c, m.Resource, namespace, name, obj, options)
}

func (m *Helper) replaceResource(c RESTClient, resource, namespace, name string, obj runtime.Object, options *metav1.UpdateOptions) (runtime.Object, error) {
	return c.Put().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(resource).
		Name(name).
		VersionedParams(options, metav1.ParameterCodec).
		Body(obj).
		Do(context.TODO()).
		Get()
}
