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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// RESTModifier provides methods for mutating a known or unknown
// RESTful resource.
type RESTModifier struct {
	Resource string
	// A RESTClient capable of mutating this resource
	RESTClient RESTClient
	// A codec for decoding and encoding objects of this resource type.
	Codec runtime.Codec
	// An interface for reading or writing the resource version of this
	// type.
	Versioner runtime.ResourceVersioner
}

// NewRESTModifier creates a RESTModifier from a RESTMapping
func NewRESTModifier(client RESTClient, mapping *meta.RESTMapping) *RESTModifier {
	return &RESTModifier{
		RESTClient: client,
		Resource:   mapping.Resource,
		Codec:      mapping.Codec,
		Versioner:  mapping.MetadataAccessor,
	}
}

func (m *RESTModifier) Delete(namespace, name string) error {
	return m.RESTClient.Delete().Path(m.Resource).Path(name).Do().Error()
}

func (m *RESTModifier) Create(namespace string, data []byte) error {
	return m.RESTClient.Post().Path(m.Resource).Body(data).Do().Error()
}

func (m *RESTModifier) Update(namespace, name string, overwrite bool, data []byte) error {
	c := m.RESTClient

	obj, err := m.Codec.Decode(data)
	if err != nil {
		// We don't know how to handle this object, but update it anyway
		return c.Put().Path(m.Resource).Path(name).Body(data).Do().Error()
	}

	// Attempt to version the object based on client logic.
	version, err := m.Versioner.ResourceVersion(obj)
	if err != nil {
		// We don't know how to version this object, so send it to the server as is
		return c.Put().Path(m.Resource).Path(name).Body(data).Do().Error()
	}
	if version == "" && overwrite {
		// Retrieve the current version of the object to overwrite the server object
		serverObj, err := c.Get().Path(m.Resource).Path(name).Do().Get()
		if err != nil {
			// The object does not exist, but we want it to be created
			return c.Put().Path(m.Resource).Path(name).Body(data).Do().Error()
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

	return c.Put().Path(m.Resource).Path(name).Body(data).Do().Error()
}
