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
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/registered"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/yaml"
)

// Mapper is a convenience struct for holding references to the three interfaces
// needed to create Info for arbitrary objects.
type Mapper struct {
	runtime.ObjectTyper
	meta.RESTMapper
	ClientMapper
}

// InfoForData creates an Info object for the given data. An error is returned
// if any of the decoding or client lookup steps fail. Name and namespace will be
// set into Info if the mapping's MetadataAccessor can retrieve them.
func (m *Mapper) InfoForData(data []byte, source string) (*Info, error) {
	json, err := yaml.ToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("unable to parse %q: %v", source, err)
	}
	data = json
	version, kind, err := runtime.UnstructuredJSONScheme.DataVersionAndKind(data)
	if err != nil {
		return nil, fmt.Errorf("unable to get type info from %q: %v", source, err)
	}
	if ok := registered.IsRegisteredAPIVersion(version); !ok {
		return nil, fmt.Errorf("API version %q in %q isn't supported, only supports API versions %q", version, source, registered.RegisteredVersions)
	}
	if kind == "" {
		return nil, fmt.Errorf("kind not set in %q", source)
	}
	mapping, err := m.RESTMapping(kind, version)
	if err != nil {
		return nil, fmt.Errorf("unable to recognize %q: %v", source, err)
	}
	obj, err := mapping.Codec.Decode(data)
	if err != nil {
		return nil, fmt.Errorf("unable to load %q: %v", source, err)
	}
	client, err := m.ClientForMapping(mapping)
	if err != nil {
		return nil, fmt.Errorf("unable to connect to a server to handle %q: %v", mapping.Resource, err)
	}

	name, _ := mapping.MetadataAccessor.Name(obj)
	namespace, _ := mapping.MetadataAccessor.Namespace(obj)
	resourceVersion, _ := mapping.MetadataAccessor.ResourceVersion(obj)

	var versionedObject interface{}

	if vo, _, _, err := api.Scheme.Raw().DecodeToVersionedObject(data); err == nil {
		versionedObject = vo
	}
	return &Info{
		Mapping:         mapping,
		Client:          client,
		Namespace:       namespace,
		Name:            name,
		Source:          source,
		VersionedObject: versionedObject,
		Object:          obj,
		ResourceVersion: resourceVersion,
	}, nil
}

// InfoForObject creates an Info object for the given Object. An error is returned
// if the object cannot be introspected. Name and namespace will be set into Info
// if the mapping's MetadataAccessor can retrieve them.
func (m *Mapper) InfoForObject(obj runtime.Object) (*Info, error) {
	version, kind, err := m.ObjectVersionAndKind(obj)
	if err != nil {
		return nil, fmt.Errorf("unable to get type info from the object %q: %v", reflect.TypeOf(obj), err)
	}
	mapping, err := m.RESTMapping(kind, version)
	if err != nil {
		return nil, fmt.Errorf("unable to recognize %q: %v", kind, err)
	}
	client, err := m.ClientForMapping(mapping)
	if err != nil {
		return nil, fmt.Errorf("unable to connect to a server to handle %q: %v", mapping.Resource, err)
	}
	name, _ := mapping.MetadataAccessor.Name(obj)
	namespace, _ := mapping.MetadataAccessor.Namespace(obj)
	resourceVersion, _ := mapping.MetadataAccessor.ResourceVersion(obj)
	return &Info{
		Mapping:   mapping,
		Client:    client,
		Namespace: namespace,
		Name:      name,

		Object:          obj,
		ResourceVersion: resourceVersion,
	}, nil
}
