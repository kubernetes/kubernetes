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
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Mapper is a convenience struct for holding references to the interfaces
// needed to create Info for arbitrary objects.
type Mapper struct {
	runtime.ObjectTyper
	meta.RESTMapper
	ClientMapper
	runtime.Decoder
}

// AcceptUnrecognizedObjects will return a mapper that will tolerate objects
// that are not recognized by the RESTMapper, returning mappings that can
// perform minimal transformation. Allows working in disconnected mode, or with
// objects that the server does not recognize. Returned resource.Info objects
// may have empty resource fields and nil clients.
func (m *Mapper) AcceptUnrecognizedObjects() *Mapper {
	copied := *m
	copied.RESTMapper = NewRelaxedRESTMapper(m.RESTMapper)
	copied.ClientMapper = NewRelaxedClientMapper(m.ClientMapper)
	return &copied
}

// InfoForData creates an Info object for the given data. An error is returned
// if any of the decoding or client lookup steps fail. Name and namespace will be
// set into Info if the mapping's MetadataAccessor can retrieve them.
func (m *Mapper) InfoForData(data []byte, source string) (*Info, error) {
	obj, gvk, err := m.Decode(data, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("unable to decode %q: %v", source, err)
	}

	mapping, err := m.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return nil, fmt.Errorf("unable to recognize %q: %v", source, err)
	}

	client, err := m.ClientForMapping(mapping)
	if err != nil {
		return nil, fmt.Errorf("unable to connect to a server to handle %q: %v", mapping.Resource, err)
	}

	name, _ := mapping.MetadataAccessor.Name(obj)
	namespace, _ := mapping.MetadataAccessor.Namespace(obj)
	resourceVersion, _ := mapping.MetadataAccessor.ResourceVersion(obj)

	return &Info{
		Client:  client,
		Mapping: mapping,

		Source:          source,
		Namespace:       namespace,
		Name:            name,
		ResourceVersion: resourceVersion,

		Object: obj,
	}, nil
}

// InfoForObject creates an Info object for the given Object. An error is returned
// if the object cannot be introspected. Name and namespace will be set into Info
// if the mapping's MetadataAccessor can retrieve them.
func (m *Mapper) InfoForObject(obj runtime.Object, preferredGVKs []schema.GroupVersionKind) (*Info, error) {
	groupVersionKinds, _, err := m.ObjectKinds(obj)
	if err != nil {
		return nil, fmt.Errorf("unable to get type info from the object %q: %v", reflect.TypeOf(obj), err)
	}

	groupVersionKind := groupVersionKinds[0]
	if len(groupVersionKinds) > 1 && len(preferredGVKs) > 0 {
		groupVersionKind = preferredObjectKind(groupVersionKinds, preferredGVKs)
	}

	mapping, err := m.RESTMapping(groupVersionKind.GroupKind(), groupVersionKind.Version)
	if err != nil {
		return nil, fmt.Errorf("unable to recognize %v: %v", groupVersionKind, err)
	}

	client, err := m.ClientForMapping(mapping)
	if err != nil {
		return nil, fmt.Errorf("unable to connect to a server to handle %q: %v", mapping.Resource, err)
	}
	name, _ := mapping.MetadataAccessor.Name(obj)
	namespace, _ := mapping.MetadataAccessor.Namespace(obj)
	resourceVersion, _ := mapping.MetadataAccessor.ResourceVersion(obj)
	return &Info{
		Client:  client,
		Mapping: mapping,

		Namespace:       namespace,
		Name:            name,
		ResourceVersion: resourceVersion,

		Object: obj,
	}, nil
}

// preferredObjectKind picks the possibility that most closely matches the priority list in this order:
// GroupVersionKind matches (exact match)
// GroupKind matches
// Group matches
func preferredObjectKind(possibilities []schema.GroupVersionKind, preferences []schema.GroupVersionKind) schema.GroupVersionKind {
	// Exact match
	for _, priority := range preferences {
		for _, possibility := range possibilities {
			if possibility == priority {
				return possibility
			}
		}
	}

	// GroupKind match
	for _, priority := range preferences {
		for _, possibility := range possibilities {
			if possibility.GroupKind() == priority.GroupKind() {
				return possibility
			}
		}
	}

	// Group match
	for _, priority := range preferences {
		for _, possibility := range possibilities {
			if possibility.Group == priority.Group {
				return possibility
			}
		}
	}

	// Just pick the first
	return possibilities[0]
}

// DisabledClientForMapping allows callers to avoid allowing remote calls when handling
// resources.
type DisabledClientForMapping struct {
	ClientMapper
}

func (f DisabledClientForMapping) ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error) {
	return nil, nil
}

// NewRelaxedClientMapper will return a nil mapping if the object is not a recognized resource.
func NewRelaxedClientMapper(mapper ClientMapper) ClientMapper {
	return relaxedClientMapper{mapper}
}

type relaxedClientMapper struct {
	ClientMapper
}

func (f relaxedClientMapper) ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error) {
	if len(mapping.Resource) == 0 {
		return nil, nil
	}
	return f.ClientMapper.ClientForMapping(mapping)
}

// NewRelaxedRESTMapper returns a RESTMapper that will tolerate mappings that don't exist in provided
// RESTMapper, returning a mapping that is a best effort against the current server. This allows objects
// that the server does not recognize to still be loaded.
func NewRelaxedRESTMapper(mapper meta.RESTMapper) meta.RESTMapper {
	return relaxedMapper{mapper}
}

type relaxedMapper struct {
	meta.RESTMapper
}

func (m relaxedMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	mapping, err := m.RESTMapper.RESTMapping(gk, versions...)
	if err != nil && meta.IsNoMatchError(err) && len(versions) > 0 {
		return &meta.RESTMapping{
			GroupVersionKind: gk.WithVersion(versions[0]),
			MetadataAccessor: meta.NewAccessor(),
			Scope:            meta.RESTScopeRoot,
			ObjectConvertor:  identityConvertor{},
		}, nil
	}
	return mapping, err
}
func (m relaxedMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	mappings, err := m.RESTMapper.RESTMappings(gk, versions...)
	if err != nil && meta.IsNoMatchError(err) && len(versions) > 0 {
		return []*meta.RESTMapping{
			{
				GroupVersionKind: gk.WithVersion(versions[0]),
				MetadataAccessor: meta.NewAccessor(),
				Scope:            meta.RESTScopeRoot,
				ObjectConvertor:  identityConvertor{},
			},
		}, nil
	}
	return mappings, err
}

type identityConvertor struct{}

var _ runtime.ObjectConvertor = identityConvertor{}

func (c identityConvertor) Convert(in interface{}, out interface{}, context interface{}) error {
	return fmt.Errorf("unable to convert objects across pointers")
}

func (c identityConvertor) ConvertToVersion(in runtime.Object, gv runtime.GroupVersioner) (out runtime.Object, err error) {
	return in, nil
}

func (c identityConvertor) ConvertFieldLabel(version string, kind string, label string, value string) (string, string, error) {
	return "", "", fmt.Errorf("unable to convert field labels")
}
