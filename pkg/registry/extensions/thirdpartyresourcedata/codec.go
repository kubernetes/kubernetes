/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

type thirdPartyObjectConverter struct {
	converter runtime.ObjectConvertor
}

func (t *thirdPartyObjectConverter) ConvertToVersion(in runtime.Object, outVersion runtime.GroupVersioner) (out runtime.Object, err error) {
	switch in.(type) {
	// This seems weird, but in this case the ThirdPartyResourceData is really just a wrapper on the raw 3rd party data.
	// The actual thing printed/sent to server is the actual raw third party resource data, which only has one version.
	case *extensions.ThirdPartyResourceData:
		return in, nil
	default:
		return t.converter.ConvertToVersion(in, outVersion)
	}
}

func (t *thirdPartyObjectConverter) Convert(in, out, context interface{}) error {
	return t.converter.Convert(in, out, context)
}

func (t *thirdPartyObjectConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return t.converter.ConvertFieldLabel(version, kind, label, value)
}

func NewThirdPartyObjectConverter(converter runtime.ObjectConvertor) runtime.ObjectConvertor {
	return &thirdPartyObjectConverter{converter}
}

type thirdPartyResourceDataMapper struct {
	mapper  meta.RESTMapper
	kind    string
	version string
	group   string
}

var _ meta.RESTMapper = &thirdPartyResourceDataMapper{}

func (t *thirdPartyResourceDataMapper) getResource() unversioned.GroupVersionResource {
	plural, _ := meta.KindToResource(t.getKind())

	return plural
}

func (t *thirdPartyResourceDataMapper) getKind() unversioned.GroupVersionKind {
	return unversioned.GroupVersionKind{Group: t.group, Version: t.version, Kind: t.kind}
}

func (t *thirdPartyResourceDataMapper) isThirdPartyResource(partialResource unversioned.GroupVersionResource) bool {
	actualResource := t.getResource()
	if strings.ToLower(partialResource.Resource) != strings.ToLower(actualResource.Resource) {
		return false
	}
	if len(partialResource.Group) != 0 && partialResource.Group != actualResource.Group {
		return false
	}
	if len(partialResource.Version) != 0 && partialResource.Version != actualResource.Version {
		return false
	}

	return true
}

func (t *thirdPartyResourceDataMapper) ResourcesFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	if t.isThirdPartyResource(resource) {
		return []unversioned.GroupVersionResource{t.getResource()}, nil
	}
	return t.mapper.ResourcesFor(resource)
}

func (t *thirdPartyResourceDataMapper) KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error) {
	if t.isThirdPartyResource(resource) {
		return []unversioned.GroupVersionKind{t.getKind()}, nil
	}
	return t.mapper.KindsFor(resource)
}

func (t *thirdPartyResourceDataMapper) ResourceFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	if t.isThirdPartyResource(resource) {
		return t.getResource(), nil
	}
	return t.mapper.ResourceFor(resource)
}

func (t *thirdPartyResourceDataMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	if t.isThirdPartyResource(resource) {
		return t.getKind(), nil
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
	extensionGK := unversioned.GroupKind{Group: extensions.GroupName, Kind: "ThirdPartyResourceData"}

	mapping, err := t.mapper.RESTMapping(extensionGK, registered.GroupOrDie(extensions.GroupName).GroupVersion.Version)
	if err != nil {
		return nil, err
	}
	mapping.ObjectConvertor = &thirdPartyObjectConverter{mapping.ObjectConvertor}
	return mapping, nil
}

func (t *thirdPartyResourceDataMapper) RESTMappings(gk unversioned.GroupKind) ([]*meta.RESTMapping, error) {
	if gk.Group != t.group {
		return nil, fmt.Errorf("unknown group %q expected %s", gk.Group, t.group)
	}
	if gk.Kind != "ThirdPartyResourceData" {
		return nil, fmt.Errorf("unknown kind %s expected %s", gk.Kind, t.kind)
	}

	// TODO figure out why we're doing this rewriting
	extensionGK := unversioned.GroupKind{Group: extensions.GroupName, Kind: "ThirdPartyResourceData"}

	mappings, err := t.mapper.RESTMappings(extensionGK)
	if err != nil {
		return nil, err
	}
	for _, m := range mappings {
		m.ObjectConvertor = &thirdPartyObjectConverter{m.ObjectConvertor}
	}
	return mappings, nil
}

func (t *thirdPartyResourceDataMapper) AliasesForResource(resource string) ([]string, bool) {
	return t.mapper.AliasesForResource(resource)
}

func (t *thirdPartyResourceDataMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return t.mapper.ResourceSingularizer(resource)
}

func NewMapper(mapper meta.RESTMapper, kind, version, group string) meta.RESTMapper {
	return &thirdPartyResourceDataMapper{
		mapper:  mapper,
		kind:    kind,
		version: version,
		group:   group,
	}
}

type thirdPartyResourceDataCodecFactory struct {
	delegate runtime.NegotiatedSerializer
	kind     string
	encodeGV unversioned.GroupVersion
	decodeGV unversioned.GroupVersion
}

func NewNegotiatedSerializer(s runtime.NegotiatedSerializer, kind string, encodeGV, decodeGV unversioned.GroupVersion) runtime.NegotiatedSerializer {
	return &thirdPartyResourceDataCodecFactory{
		delegate: s,
		kind:     kind,
		encodeGV: encodeGV,
		decodeGV: decodeGV,
	}
}

func (t *thirdPartyResourceDataCodecFactory) SupportedMediaTypes() []runtime.SerializerInfo {
	for _, info := range t.delegate.SupportedMediaTypes() {
		if info.MediaType == runtime.ContentTypeJSON {
			return []runtime.SerializerInfo{info}
		}
	}
	return nil
}

func (t *thirdPartyResourceDataCodecFactory) EncoderForVersion(s runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return &thirdPartyResourceDataEncoder{delegate: t.delegate.EncoderForVersion(s, gv), gvk: t.encodeGV.WithKind(t.kind)}
}

func (t *thirdPartyResourceDataCodecFactory) DecoderToVersion(s runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return NewDecoder(t.delegate.DecoderToVersion(s, gv), t.kind)
}

func NewCodec(delegate runtime.Codec, gvk unversioned.GroupVersionKind) runtime.Codec {
	return runtime.NewCodec(NewEncoder(delegate, gvk), NewDecoder(delegate, gvk.Kind))
}

type thirdPartyResourceDataDecoder struct {
	delegate runtime.Decoder
	kind     string
}

func NewDecoder(delegate runtime.Decoder, kind string) runtime.Decoder {
	return &thirdPartyResourceDataDecoder{delegate: delegate, kind: kind}
}

var _ runtime.Decoder = &thirdPartyResourceDataDecoder{}

func parseObject(data []byte) (map[string]interface{}, error) {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, err
	}
	mapObj, ok := obj.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected object: %#v", obj)
	}
	return mapObj, nil
}

func (t *thirdPartyResourceDataDecoder) populate(data []byte) (runtime.Object, *unversioned.GroupVersionKind, error) {
	mapObj, err := parseObject(data)
	if err != nil {
		return nil, nil, err
	}
	return t.populateFromObject(mapObj, data)
}

func (t *thirdPartyResourceDataDecoder) populateFromObject(mapObj map[string]interface{}, data []byte) (runtime.Object, *unversioned.GroupVersionKind, error) {
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(data, &typeMeta); err != nil {
		return nil, nil, err
	}

	gv, err := unversioned.ParseGroupVersion(typeMeta.APIVersion)
	if err != nil {
		return nil, nil, err
	}
	gvk := gv.WithKind(typeMeta.Kind)

	isList := strings.HasSuffix(typeMeta.Kind, "List")
	switch {
	case !isList && (len(t.kind) == 0 || typeMeta.Kind == t.kind):
		result := &extensions.ThirdPartyResourceData{}
		if err := t.populateResource(result, mapObj, data); err != nil {
			return nil, nil, err
		}
		return result, &gvk, nil
	case isList && (len(t.kind) == 0 || typeMeta.Kind == t.kind+"List"):
		list := &extensions.ThirdPartyResourceDataList{}
		if err := t.populateListResource(list, mapObj); err != nil {
			return nil, nil, err
		}
		return list, &gvk, nil
	default:
		return nil, nil, fmt.Errorf("unexpected kind: %s, expected %s", typeMeta.Kind, t.kind)
	}
}

func (t *thirdPartyResourceDataDecoder) populateResource(objIn *extensions.ThirdPartyResourceData, mapObj map[string]interface{}, data []byte) error {
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
	// Override API Version with the ThirdPartyResourceData value
	// TODO: fix this hard code
	objIn.APIVersion = v1beta1.SchemeGroupVersion.String()

	objIn.Data = data
	return nil
}

func IsThirdPartyObject(rawData []byte, gvk *unversioned.GroupVersionKind) (isThirdParty bool, gvkOut *unversioned.GroupVersionKind, err error) {
	var gv unversioned.GroupVersion
	if gvk == nil {
		data, err := yaml.ToJSON(rawData)
		if err != nil {
			return false, nil, err
		}
		metadata := unversioned.TypeMeta{}
		if err = json.Unmarshal(data, &metadata); err != nil {
			return false, nil, err
		}
		gv, err = unversioned.ParseGroupVersion(metadata.APIVersion)
		if err != nil {
			return false, nil, err
		}
		gvkOut = &unversioned.GroupVersionKind{
			Group:   gv.Group,
			Version: gv.Version,
			Kind:    metadata.Kind,
		}
	} else {
		gv = gvk.GroupVersion()
		gvkOut = gvk
	}
	return registered.IsThirdPartyAPIGroupVersion(gv), gvkOut, nil
}

func (t *thirdPartyResourceDataDecoder) Decode(data []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	if into == nil {
		if gvk == nil || gvk.Kind != t.kind {
			if isThirdParty, _, err := IsThirdPartyObject(data, gvk); err != nil {
				return nil, nil, err
			} else if !isThirdParty {
				return t.delegate.Decode(data, gvk, into)
			}
		}
		return t.populate(data)
	}
	switch o := into.(type) {
	case *extensions.ThirdPartyResourceData:
		break
	case *runtime.VersionedObjects:
		// We're not sure that it's third party, we need to test
		if gvk == nil || gvk.Kind != t.kind {
			if isThirdParty, _, err := IsThirdPartyObject(data, gvk); err != nil {
				return nil, nil, err
			} else if !isThirdParty {
				return t.delegate.Decode(data, gvk, into)
			}
		}
		obj, outGVK, err := t.populate(data)
		if err != nil {
			return nil, nil, err
		}
		o.Objects = []runtime.Object{
			obj,
		}
		return o, outGVK, nil
	default:
		if gvk != nil && registered.IsThirdPartyAPIGroupVersion(gvk.GroupVersion()) {
			// delegate won't recognize a thirdparty group version
			gvk = nil
		}
		return t.delegate.Decode(data, gvk, into)
	}

	thirdParty := into.(*extensions.ThirdPartyResourceData)
	var dataObj interface{}
	if err := json.Unmarshal(data, &dataObj); err != nil {
		return nil, nil, err
	}
	mapObj, ok := dataObj.(map[string]interface{})
	if !ok {

		return nil, nil, fmt.Errorf("unexpected object: %#v", dataObj)
	}
	/*if gvk.Kind != "ThirdPartyResourceData" {
		return nil, nil, fmt.Errorf("unexpected kind: %s", gvk.Kind)
	}*/
	actual := &unversioned.GroupVersionKind{}
	if kindObj, found := mapObj["kind"]; !found {
		if gvk == nil {
			return nil, nil, runtime.NewMissingKindErr(string(data))
		}
		mapObj["kind"] = gvk.Kind
		actual.Kind = gvk.Kind
	} else {
		kindStr, ok := kindObj.(string)
		if !ok {
			return nil, nil, fmt.Errorf("unexpected object for 'kind': %v", kindObj)
		}
		if len(t.kind) > 0 && kindStr != t.kind {
			return nil, nil, fmt.Errorf("kind doesn't match, expecting: %s, got %s", t.kind, kindStr)
		}
		actual.Kind = kindStr
	}
	if versionObj, found := mapObj["apiVersion"]; !found {
		if gvk == nil {
			return nil, nil, runtime.NewMissingVersionErr(string(data))
		}
		mapObj["apiVersion"] = gvk.GroupVersion().String()
		actual.Group, actual.Version = gvk.Group, gvk.Version
	} else {
		versionStr, ok := versionObj.(string)
		if !ok {
			return nil, nil, fmt.Errorf("unexpected object for 'apiVersion': %v", versionObj)
		}
		if gvk != nil && versionStr != gvk.GroupVersion().String() {
			return nil, nil, fmt.Errorf("version doesn't match, expecting: %v, got %s", gvk.GroupVersion(), versionStr)
		}
		gv, err := unversioned.ParseGroupVersion(versionStr)
		if err != nil {
			return nil, nil, err
		}
		actual.Group, actual.Version = gv.Group, gv.Version
	}

	mapObj, err := parseObject(data)
	if err != nil {
		return nil, actual, err
	}
	if err := t.populateResource(thirdParty, mapObj, data); err != nil {
		return nil, actual, err
	}
	return thirdParty, actual, nil
}

func (t *thirdPartyResourceDataDecoder) populateListResource(objIn *extensions.ThirdPartyResourceDataList, mapObj map[string]interface{}) error {
	items, ok := mapObj["items"].([]interface{})
	if !ok {
		return fmt.Errorf("unexpected object for items: %#v", mapObj["items"])
	}
	objIn.Items = make([]extensions.ThirdPartyResourceData, len(items))
	for ix := range items {
		objData, err := json.Marshal(items[ix])
		if err != nil {
			return err
		}
		objMap, err := parseObject(objData)
		if err != nil {
			return err
		}
		if err := t.populateResource(&objIn.Items[ix], objMap, objData); err != nil {
			return err
		}
	}
	return nil
}

type thirdPartyResourceDataEncoder struct {
	delegate runtime.Encoder
	gvk      unversioned.GroupVersionKind
}

func NewEncoder(delegate runtime.Encoder, gvk unversioned.GroupVersionKind) runtime.Encoder {
	return &thirdPartyResourceDataEncoder{delegate: delegate, gvk: gvk}
}

var _ runtime.Encoder = &thirdPartyResourceDataEncoder{}

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

func (t *thirdPartyResourceDataEncoder) Encode(obj runtime.Object, stream io.Writer) (err error) {
	switch obj := obj.(type) {
	case *extensions.ThirdPartyResourceData:
		return encodeToJSON(obj, stream)
	case *extensions.ThirdPartyResourceDataList:
		// TODO: There are likely still better ways to do this...
		listItems := make([]json.RawMessage, len(obj.Items))

		for ix := range obj.Items {
			buff := &bytes.Buffer{}
			err := encodeToJSON(&obj.Items[ix], buff)
			if err != nil {
				return err
			}
			listItems[ix] = json.RawMessage(buff.Bytes())
		}

		if t.gvk.Empty() {
			return fmt.Errorf("thirdPartyResourceDataEncoder was not given a target version")
		}

		encMap := struct {
			// +optional
			Kind  string            `json:"kind,omitempty"`
			Items []json.RawMessage `json:"items"`
			// +optional
			Metadata unversioned.ListMeta `json:"metadata,omitempty"`
			// +optional
			APIVersion string `json:"apiVersion,omitempty"`
		}{
			Kind:       t.gvk.Kind + "List",
			Items:      listItems,
			Metadata:   obj.ListMeta,
			APIVersion: t.gvk.GroupVersion().String(),
		}

		encBytes, err := json.Marshal(encMap)
		if err != nil {
			return err
		}

		_, err = stream.Write(encBytes)
		return err
	case *versioned.InternalEvent:
		event := &versioned.Event{}
		err := versioned.Convert_versioned_InternalEvent_to_versioned_Event(obj, event, nil)
		if err != nil {
			return err
		}

		enc := json.NewEncoder(stream)
		err = enc.Encode(event)
		if err != nil {
			return err
		}

		return nil
	case *unversioned.Status, *unversioned.APIResourceList:
		return t.delegate.Encode(obj, stream)
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

func (t *thirdPartyResourceDataCreator) New(kind unversioned.GroupVersionKind) (out runtime.Object, err error) {
	switch kind.Kind {
	case "ThirdPartyResourceData":
		if apiutil.GetGroupVersion(t.group, t.version) != kind.GroupVersion().String() {
			return nil, fmt.Errorf("unknown kind %v", kind)
		}
		return &extensions.ThirdPartyResourceData{}, nil
	case "ThirdPartyResourceDataList":
		if apiutil.GetGroupVersion(t.group, t.version) != kind.GroupVersion().String() {
			return nil, fmt.Errorf("unknown kind %v", kind)
		}
		return &extensions.ThirdPartyResourceDataList{}, nil
	// TODO: this list needs to be formalized higher in the chain
	case "ListOptions", "WatchEvent":
		if apiutil.GetGroupVersion(t.group, t.version) == kind.GroupVersion().String() {
			// Translate third party group to external group.
			gvk := registered.EnabledVersionsForGroup(api.GroupName)[0].WithKind(kind.Kind)
			return t.delegate.New(gvk)
		}
		return t.delegate.New(kind)
	default:
		return t.delegate.New(kind)
	}
}

func NewThirdPartyParameterCodec(p runtime.ParameterCodec) runtime.ParameterCodec {
	return &thirdPartyParameterCodec{p}
}

type thirdPartyParameterCodec struct {
	delegate runtime.ParameterCodec
}

func (t *thirdPartyParameterCodec) DecodeParameters(parameters url.Values, from unversioned.GroupVersion, into runtime.Object) error {
	return t.delegate.DecodeParameters(parameters, v1.SchemeGroupVersion, into)
}

func (t *thirdPartyParameterCodec) EncodeParameters(obj runtime.Object, to unversioned.GroupVersion) (url.Values, error) {
	return t.delegate.EncodeParameters(obj, v1.SchemeGroupVersion)
}
