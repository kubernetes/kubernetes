/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubectl

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// RESTHelper provides methods for retrieving or mutating a RESTful
// resource.
type RESTHelper struct {
	Resource string
	// A RESTClient capable of mutating this resource
	RESTClient RESTClient
	// A codec for decoding and encoding objects of this resource type.
	Codec runtime.Codec
	// An interface for reading or writing the resource version of this
	// type.
	Versioner runtime.ResourceVersioner
}

// NewRESTHelper creates a RESTHelper from a ResourceMapping
func NewRESTHelper(client RESTClient, mapping *meta.RESTMapping) *RESTHelper {
	return &RESTHelper{
		RESTClient: client,
		Resource:   mapping.Resource,
		Codec:      mapping.Codec,
		Versioner:  mapping.MetadataAccessor,
	}
}

func (m *RESTHelper) Get(namespace, name string, selector labels.Selector) (runtime.Object, error) {
	return m.RESTClient.Get().Path(m.Resource).Namespace(namespace).Path(name).SelectorParam("labels", selector).Do().Get()
}

func (m *RESTHelper) Delete(namespace, name string) error {
	return m.RESTClient.Delete().Path(m.Resource).Namespace(namespace).Path(name).Do().Error()
}

func (m *RESTHelper) Create(namespace string, modify bool, data []byte) error {
	if modify {
		obj, err := m.Codec.Decode(data)
		if err != nil {
			// We don't know how to check a version on this object, but create it anyway
			return createResource(m.RESTClient, m.Resource, namespace, data)
		}

		// Attempt to version the object based on client logic.
		version, err := m.Versioner.ResourceVersion(obj)
		if err != nil {
			// We don't know how to clear the version on this object, so send it to the server as is
			return createResource(m.RESTClient, m.Resource, namespace, data)
		}
		if version != "" {
			if err := m.Versioner.SetResourceVersion(obj, ""); err != nil {
				return err
			}
			newData, err := m.Codec.Encode(obj)
			if err != nil {
				return err
			}
			data = newData
		}
	}

	return createResource(m.RESTClient, m.Resource, namespace, data)
}

func createResource(c RESTClient, resourcePath, namespace string, data []byte) error {
	return c.Post().Path(resourcePath).Namespace(namespace).Body(data).Do().Error()
}

func (m *RESTHelper) Update(namespace, name string, overwrite bool, data []byte) error {
	c := m.RESTClient

	obj, err := m.Codec.Decode(data)
	if err != nil {
		// We don't know how to handle this object, but update it anyway
		return updateResource(c, m.Resource, namespace, name, data)
	}

	// Attempt to version the object based on client logic.
	version, err := m.Versioner.ResourceVersion(obj)
	if err != nil {
		// We don't know how to version this object, so send it to the server as is
		return updateResource(c, m.Resource, namespace, name, data)
	}
	if version == "" && overwrite {
		// Retrieve the current version of the object to overwrite the server object
		serverObj, err := c.Get().Path(m.Resource).Path(name).Do().Get()
		if err != nil {
			// The object does not exist, but we want it to be created
			return updateResource(c, m.Resource, namespace, name, data)
		}
		serverVersion, err := m.Versioner.ResourceVersion(serverObj)
		if err != nil {
			return err
		}
		if err := m.Versioner.SetResourceVersion(obj, serverVersion); err != nil {
			return err
		}
		newData, err := m.Codec.Encode(obj)
		if err != nil {
			return err
		}
		data = newData
	}

	return updateResource(c, m.Resource, namespace, name, data)
}

func updateResource(c RESTClient, resourcePath, namespace, name string, data []byte) error {
	return c.Put().Path(resourcePath).Namespace(namespace).Path(name).Body(data).Do().Error()
}
