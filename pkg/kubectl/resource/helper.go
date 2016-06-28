/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	apierr "k8s.io/kubernetes/pkg/api/errors"
)

// Helper provides methods for retrieving or mutating a RESTful
// resource.
type Helper struct {
	// The name of this resource as the server would recognize it
	Resource string
	// A RESTClient capable of mutating this resource.
	RESTClient RESTClient
	// An interface for reading or writing the resource version of this
	// type.
	Versioner runtime.ResourceVersioner
	// True if the resource type is scoped to namespaces
	NamespaceScoped bool
}

// NewHelper creates a Helper from a ResourceMapping
func NewHelper(client RESTClient, mapping *meta.RESTMapping) *Helper {
	return &Helper{
		Resource:        mapping.Resource,
		RESTClient:      client,
		Versioner:       mapping.MetadataAccessor,
		NamespaceScoped: mapping.Scope.Name() == meta.RESTScopeNameNamespace,
	}
}

func (m *Helper) Get(namespace, name string, export bool) (runtime.Object, error) {
	req := m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name)
	if export {
		req.Param("export", strconv.FormatBool(export))
	}
	return req.Do().Get()
}

// TODO: add field selector
func (m *Helper) List(namespace, apiVersion string, selector labels.Selector, export bool) (runtime.Object, error) {
	req := m.RESTClient.Get().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		LabelsSelectorParam(selector)
	if export {
		req.Param("export", strconv.FormatBool(export))
	}
	return req.Do().Get()
}

func (m *Helper) Watch(namespace, resourceVersion, apiVersion string, labelSelector labels.Selector) (watch.Interface, error) {
	return m.RESTClient.Get().
		Prefix("watch").
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(labelSelector).
		Watch()
}

func (m *Helper) WatchSingle(namespace, name, resourceVersion string) (watch.Interface, error) {
	return m.RESTClient.Get().
		Prefix("watch").
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name).
		Param("resourceVersion", resourceVersion).
		Watch()
}

func (m *Helper) Delete(namespace, name string) error {
	return m.RESTClient.Delete().
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name).
		Do().
		Error()
}

func (m *Helper) Create(namespace string, name string, modify bool, obj runtime.Object) (runtime.Object, error) {
	if modify {
		/* Attempt to version the object based on client logic.
		   If we don't know how to clear the version on this object, so send it to the server as is
		*/
		version, err := m.Versioner.ResourceVersion(obj)
		if err == nil && version != "" {
			if err := m.Versioner.SetResourceVersion(obj, ""); err != nil {
				return nil, err
			}
		}
	}

	ret_obj, err := m.createResource(m.RESTClient, m.Resource, namespace, obj)

	if err != nil && apierr.IsAlreadyExists(err) {
		//check whether the existent pod/job is terminated
		if m.Resource == "pods" || m.Resource == "jobs" {
			exist_obj, get_err := m.Get(namespace, name, false)
			if get_err == nil {
				append_warn_msg := fmt.Sprintf("\n Existant %s is in termanited state! Please check its status and delete it!\n Use `kubectl get %s/%s` \n",
					map[bool]string{true: "pod", false: "job"}[m.Resource == "pods"], m.Resource, name)
				switch exist_obj.(type) {
					case *api.Pod:
						pod, _ := exist_obj.(*api.Pod)
						if get_err == nil && api.IsPodTerminated(pod) {
							status_err, _ := err.(*apierr.StatusError)
							status_err.ErrStatus.Message += append_warn_msg
						}
					case *batch.Job:
						job, _ := exist_obj.(*batch.Job)
						if get_err == nil && batch.IsJobFinished(*job) {
							status_err, _ := err.(*apierr.StatusError)
							status_err.ErrStatus.Message += append_warn_msg
						}
				}

			}
		}
	}

	return ret_obj, err
}

func (m *Helper) createResource(c RESTClient, resource, namespace string, obj runtime.Object) (runtime.Object, error) {
	return c.Post().NamespaceIfScoped(namespace, m.NamespaceScoped).Resource(resource).Body(obj).Do().Get()
}
func (m *Helper) Patch(namespace, name string, pt api.PatchType, data []byte) (runtime.Object, error) {
	return m.RESTClient.Patch(pt).
		NamespaceIfScoped(namespace, m.NamespaceScoped).
		Resource(m.Resource).
		Name(name).
		Body(data).
		Do().
		Get()
}

func (m *Helper) Replace(namespace, name string, overwrite bool, obj runtime.Object) (runtime.Object, error) {
	c := m.RESTClient

	// Attempt to version the object based on client logic.
	version, err := m.Versioner.ResourceVersion(obj)
	if err != nil {
		// We don't know how to version this object, so send it to the server as is
		return m.replaceResource(c, m.Resource, namespace, name, obj)
	}
	if version == "" && overwrite {
		// Retrieve the current version of the object to overwrite the server object
		serverObj, err := c.Get().NamespaceIfScoped(namespace, m.NamespaceScoped).Resource(m.Resource).Name(name).Do().Get()
		if err != nil {
			// The object does not exist, but we want it to be created
			return m.replaceResource(c, m.Resource, namespace, name, obj)
		}
		serverVersion, err := m.Versioner.ResourceVersion(serverObj)
		if err != nil {
			return nil, err
		}
		if err := m.Versioner.SetResourceVersion(obj, serverVersion); err != nil {
			return nil, err
		}
	}

	return m.replaceResource(c, m.Resource, namespace, name, obj)
}

func (m *Helper) replaceResource(c RESTClient, resource, namespace, name string, obj runtime.Object) (runtime.Object, error) {
	return c.Put().NamespaceIfScoped(namespace, m.NamespaceScoped).Resource(resource).Name(name).Body(obj).Do().Get()
}
