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
	"io"
	"net/url"
	"strings"

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
)

type thirdPartyResourceDataMapper struct {
	mapper  meta.RESTMapper
	kind    string
	version string
	group   string
}

var _ meta.RESTMapper = &thirdPartyResourceDataMapper{}

func (t *thirdPartyResourceDataMapper) isThirdPartyResource(resource string) bool {
	return resource == strings.ToLower(t.kind)+"s"
}

func (t *thirdPartyResourceDataMapper) KindFor(resource string) (unversioned.GroupVersionKind, error) {
	if t.isThirdPartyResource(resource) {
		return unversioned.GroupVersionKind{Group: t.group, Version: t.version, Kind: t.kind}, nil
	}
	return t.mapper.KindFor(resource)
}

func (t *thirdPartyResourceDataMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	if len(versions) != 1 {
		return nil, fmt.Errorf("unexpected set of versions: %v", versions)
	}
	if gk.Group != t.group {
		return nil, fmt.Errorf("unknown group %q expected %s", gk.Group, t.group)
	}
	if gk.Kind != "ThirdPartyResourceData" {
		return nil, fmt.Errorf("unknown kind %s expected %s", gk.Kind, t.kind)
	}
	if versions[0] != t.version {
		return nil, fmt.Errorf("unknown version %q expected %q", versions[0], t.version)
	}

	// TODO figure out why we're doing this rewriting
	extensionGK := unversioned.GroupKind{Group: "extensions", Kind: "ThirdPartyResourceData"}

	mapping, err := t.mapper.RESTMapping(extensionGK, latest.GroupOrDie("extensions").GroupVersion.Version)
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

// ResourceIsValid takes a string (kind) and checks if it's a valid resource
func (t *thirdPartyResourceDataMapper) ResourceIsValid(resource string) bool {
	return t.isThirdPartyResource(resource) || t.mapper.ResourceIsValid(resource)
}

func NewMapper(mapper meta.RESTMapper, kind, version, group string) meta.RESTMapper {
	return &thirdPartyResourceDataMapper{
		mapper:  mapper,
		kind:    kind,
		version: version,
		group:   group,
	}
}

type thirdPartyResourceDataCodec struct {
	delegate runtime.Codec
	kind     string
}

func NewCodec(codec runtime.Codec, kind string) runtime.Codec {
	return &thirdPartyResourceDataCodec{codec, kind}
}

func (t *thirdPartyResourceDataCodec) populate(objIn *extensions.ThirdPartyResourceData, data []byte) error {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		fmt.Printf("Invalid JSON:\n%s\n", string(data))
		return err
	}
	mapObj, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	return t.populateFromObject(objIn, mapObj, data)
}

func (t *thirdPartyResourceDataCodec) populateFromObject(objIn *extensions.ThirdPartyResourceData, mapObj map[string]interface{}, data []byte) error {
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(data, &typeMeta); err != nil {
		return err
	}
	if typeMeta.Kind != t.kind {
		return fmt.Errorf("unexpected kind: %s, expected %s", typeMeta.Kind, t.kind)
	}

	metadata, ok := mapObj["metadata"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpected object for metadata: %#v", mapObj["metadata"])
	}

	metadataData, err := json.Marshal(metadata)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(metadataData, &objIn.ObjectMeta); err != nil {
		return err
	}

	objIn.Data = data
	return nil
}

func (t *thirdPartyResourceDataCodec) Decode(data []byte) (runtime.Object, error) {
	result := &extensions.ThirdPartyResourceData{}
	if err := t.populate(result, data); err != nil {
		return nil, err
	}
	return result, nil
}

func (t *thirdPartyResourceDataCodec) DecodeToVersion(data []byte, gv unversioned.GroupVersion) (runtime.Object, error) {
	// TODO: this is hacky, there must be a better way...
	obj, err := t.Decode(data)
	if err != nil {
		return nil, err
	}
	objData, err := t.delegate.Encode(obj)
	if err != nil {
		return nil, err
	}
	return t.delegate.DecodeToVersion(objData, gv)
}

func (t *thirdPartyResourceDataCodec) DecodeInto(data []byte, obj runtime.Object) error {
	thirdParty, ok := obj.(*extensions.ThirdPartyResourceData)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}
	return t.populate(thirdParty, data)
}

func (t *thirdPartyResourceDataCodec) DecodeIntoWithSpecifiedVersionKind(data []byte, obj runtime.Object, gvk unversioned.GroupVersionKind) error {
	thirdParty, ok := obj.(*extensions.ThirdPartyResourceData)
	if !ok {
		return fmt.Errorf("unexpected object: %#v", obj)
	}

	if gvk.Kind != "ThirdPartyResourceData" {
		return fmt.Errorf("unexpeceted kind: %s", gvk.Kind)
	}

	var dataObj interface{}
	if err := json.Unmarshal(data, &dataObj); err != nil {
		return err
	}
	mapObj, ok := dataObj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpcted object: %#v", dataObj)
	}
	if kindObj, found := mapObj["kind"]; !found {
		mapObj["kind"] = gvk.Kind
	} else {
		kindStr, ok := kindObj.(string)
		if !ok {
			return fmt.Errorf("unexpected object for 'kind': %v", kindObj)
		}
		if kindStr != t.kind {
			return fmt.Errorf("kind doesn't match, expecting: %s, got %s", gvk.Kind, kindStr)
		}
	}
	if versionObj, found := mapObj["apiVersion"]; !found {
		mapObj["apiVersion"] = gvk.GroupVersion().String()
	} else {
		versionStr, ok := versionObj.(string)
		if !ok {
			return fmt.Errorf("unexpected object for 'apiVersion': %v", versionObj)
		}
		if versionStr != gvk.GroupVersion().String() {
			return fmt.Errorf("version doesn't match, expecting: %v, got %s", gvk.GroupVersion(), versionStr)
		}
	}

	if err := t.populate(thirdParty, data); err != nil {
		return err
	}
	return nil
}

func (t *thirdPartyResourceDataCodec) DecodeParametersInto(parameters url.Values, obj runtime.Object) error {
	return t.delegate.DecodeParametersInto(parameters, obj)
}

const template = `{
  "kind": "%s",
  "items": [ %s ]
}`

func encodeToJSON(obj *extensions.ThirdPartyResourceData, stream io.Writer) error {
	var objOut interface{}
	if err := json.Unmarshal(obj.Data, &objOut); err != nil {
		return err
	}
	objMap, ok := objOut.(map[string]interface{})
	if !ok {
		return fmt.Errorf("unexpected type: %v", objOut)
	}
	objMap["metadata"] = obj.ObjectMeta
	encoder := json.NewEncoder(stream)
	return encoder.Encode(objMap)
}

func (t *thirdPartyResourceDataCodec) Encode(obj runtime.Object) ([]byte, error) {
	buff := &bytes.Buffer{}
	if err := t.EncodeToStream(obj, buff); err != nil {
		return nil, err
	}
	return buff.Bytes(), nil
}

func (t *thirdPartyResourceDataCodec) EncodeToStream(obj runtime.Object, stream io.Writer) (err error) {
	switch obj := obj.(type) {
	case *extensions.ThirdPartyResourceData:
		return encodeToJSON(obj, stream)
	case *extensions.ThirdPartyResourceDataList:
		// TODO: There must be a better way to do this...
		dataStrings := make([]string, len(obj.Items))
		for ix := range obj.Items {
			buff := &bytes.Buffer{}
			err := encodeToJSON(&obj.Items[ix], buff)
			if err != nil {
				return err
			}
			dataStrings[ix] = buff.String()
		}
		fmt.Fprintf(stream, template, t.kind+"List", strings.Join(dataStrings, ","))
		return nil
	case *unversioned.Status:
		return t.delegate.EncodeToStream(obj, stream)
	default:
		return fmt.Errorf("unexpected object to encode: %#v", obj)
	}
}

func NewObjectCreator(group, version string, delegate runtime.ObjectCreater) runtime.ObjectCreater {
	return &thirdPartyResourceDataCreator{group, version, delegate}
}

type thirdPartyResourceDataCreator struct {
	group    string
	version  string
	delegate runtime.ObjectCreater
}

func (t *thirdPartyResourceDataCreator) New(groupVersion, kind string) (out runtime.Object, err error) {
	switch kind {
	case "ThirdPartyResourceData":
		if apiutil.GetGroupVersion(t.group, t.version) != groupVersion {
			return nil, fmt.Errorf("unknown version %s for kind %s", groupVersion, kind)
		}
		return &extensions.ThirdPartyResourceData{}, nil
	case "ThirdPartyResourceDataList":
		if apiutil.GetGroupVersion(t.group, t.version) != groupVersion {
			return nil, fmt.Errorf("unknown version %s for kind %s", groupVersion, kind)
		}
		return &extensions.ThirdPartyResourceDataList{}, nil
	default:
		return t.delegate.New(groupVersion, kind)
	}
}
