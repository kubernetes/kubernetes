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

package thirdpartyresourcedata

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/runtime"
)

type thirdPartyResourceDataMapper struct {
	mapper meta.RESTMapper
	kind   string
}

func (t *thirdPartyResourceDataMapper) GroupForResource(resource string) (string, error) {
	return t.mapper.GroupForResource(resource)
}

func (t *thirdPartyResourceDataMapper) RESTMapping(kind string, versions ...string) (*meta.RESTMapping, error) {
	mapping, err := t.mapper.RESTMapping(kind, versions...)
	if err != nil {
		return nil, err
	}
	mapping.Codec = NewCodec(mapping.Codec, t.kind)
	return mapping, nil
}

func (t *thirdPartyResourceDataMapper) AliasesForResource(resource string) ([]string, bool) {
	return t.mapper.AliasesForResource(resource)
}

func (t *thirdPartyResourceDataMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return t.mapper.ResourceSingularizer(resource)
}

func (t *thirdPartyResourceDataMapper) VersionAndKindForResource(resource string) (defaultVersion, kind string, err error) {
	return t.mapper.VersionAndKindForResource(resource)
}

func NewMapper(mapper meta.RESTMapper, kind string) meta.RESTMapper {
	return &thirdPartyResourceDataMapper{mapper, kind}
}

type thirdPartyResourceDataCodec struct {
	delegate runtime.Codec
	kind     string
}

func NewCodec(codec runtime.Codec, kind string) runtime.Codec {
	return &thirdPartyResourceDataCodec{codec, kind}
}

func (t *thirdPartyResourceDataCodec) populate(objIn *expapi.ThirdPartyResourceData, data []byte) error {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		fmt.Printf("Invalid JSON:\n%s\n", string(data))
		return err
	}
	mapObj, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	kind, ok := mapObj["kind"].(string)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	if kind != t.kind {
		return fmt.Errorf("unexpected kind: %s, expected: %s", kind, t.kind)
	}

	metadata, ok := mapObj["metadata"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}

	name, ok := metadata["name"].(string)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}

	objIn.Name = name
	objIn.Data = data
	return nil
}

func (t *thirdPartyResourceDataCodec) Decode(data []byte) (runtime.Object, error) {
	result := &expapi.ThirdPartyResourceData{}
	if err := t.populate(result, data); err != nil {
		return nil, err
	}
	return result, nil
}

func (t *thirdPartyResourceDataCodec) DecodeToVersion(data []byte, version string) (runtime.Object, error) {
	// TODO: this is hacky, there must be a better way...
	obj, err := t.Decode(data)
	if err != nil {
		return nil, err
	}
	objData, err := t.delegate.Encode(obj)
	if err != nil {
		return nil, err
	}
	return t.delegate.DecodeToVersion(objData, version)
}

func (t *thirdPartyResourceDataCodec) DecodeInto(data []byte, obj runtime.Object) error {
	thirdParty, ok := obj.(*expapi.ThirdPartyResourceData)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	return t.populate(thirdParty, data)
}

func (t *thirdPartyResourceDataCodec) DecodeIntoWithSpecifiedVersionKind(data []byte, obj runtime.Object, kind, version string) error {
	thirdParty, ok := obj.(*expapi.ThirdPartyResourceData)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	if err := t.populate(thirdParty, data); err != nil {
		return err
	}
	thirdParty.Kind = kind
	thirdParty.APIVersion = version
	return nil
}

const template = `{
  "kind": "%s",
  "items": [ %s ]
}`

func (t *thirdPartyResourceDataCodec) Encode(obj runtime.Object) (data []byte, err error) {
	switch obj := obj.(type) {
	case *expapi.ThirdPartyResourceData:
		return obj.Data, nil
	case *expapi.ThirdPartyResourceDataList:
		// TODO: There must be a better way to do this...
		buff := &bytes.Buffer{}
		dataStrings := make([]string, len(obj.Items))
		for ix := range obj.Items {
			dataStrings[ix] = string(obj.Items[ix].Data)
		}
		fmt.Fprintf(buff, template, t.kind+"List", strings.Join(dataStrings, ","))
		return buff.Bytes(), nil
	case *api.Status:
		return t.delegate.Encode(obj)
	default:
		return nil, fmt.Errorf("unexpected object to encode: %#v", obj)
	}
}
