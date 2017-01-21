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

package schemaless

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
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

type schemalessObjectConverter struct {
	converter runtime.ObjectConvertor
}

func (t *schemalessObjectConverter) ConvertToVersion(in runtime.Object, outVersion runtime.GroupVersioner) (out runtime.Object, err error) {
	switch in.(type) {
	case *runtime.Unstructured:
		return in, nil
	default:
		return t.converter.ConvertToVersion(in, outVersion)
	}
}

func (t *schemalessObjectConverter) Convert(in, out, context interface{}) error {
	return t.converter.Convert(in, out, context)
}

func (t *schemalessObjectConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return t.converter.ConvertFieldLabel(version, kind, label, value)
}

func NewSchemalessObjectConverter(converter runtime.ObjectConvertor) runtime.ObjectConvertor {
	return &schemalessObjectConverter{converter}
}

type schemalessDataMapper struct {
	mapper  meta.RESTMapper
	kind    string
	version string
	group   string
}

var _ meta.RESTMapper = &schemalessDataMapper{}

func (t *schemalessDataMapper) getResource() unversioned.GroupVersionResource {
	plural, _ := meta.KindToResource(t.getKind())

	return plural
}

func (t *schemalessDataMapper) getKind() unversioned.GroupVersionKind {
	return unversioned.GroupVersionKind{Group: t.group, Version: t.version, Kind: t.kind}
}

func (t *schemalessDataMapper) matches(partialResource unversioned.GroupVersionResource) bool {
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

func (t *schemalessDataMapper) ResourcesFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	if t.matches(resource) {
		return []unversioned.GroupVersionResource{t.getResource()}, nil
	}
	return t.mapper.ResourcesFor(resource)
}

func (t *schemalessDataMapper) KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error) {
	if t.matches(resource) {
		return []unversioned.GroupVersionKind{t.getKind()}, nil
	}
	return t.mapper.KindsFor(resource)
}

func (t *schemalessDataMapper) ResourceFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	if t.matches(resource) {
		return t.getResource(), nil
	}
	return t.mapper.ResourceFor(resource)
}

func (t *schemalessDataMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	if t.matches(resource) {
		return t.getKind(), nil
	}
	return t.mapper.KindFor(resource)
}

func (t *schemalessDataMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	if len(versions) != 1 {
		return nil, fmt.Errorf("unexpected set of versions: %v", versions)
	}
	if gk.Group != t.group {
		return nil, fmt.Errorf("unknown group %q expected %s", gk.Group, t.group)
	}
	if gk.Kind != t.kind {
		return nil, fmt.Errorf("unknown kind %s expected %s", gk.Kind, t.kind)
	}
	if versions[0] != t.version {
		return nil, fmt.Errorf("unknown version %q expected %q", versions[0], t.version)
	}

	return &meta.RESTMapping{
		Resource:         t.getResource().Resource,
		GroupVersionKind: unversioned.GroupVersionKind{Group: t.group, Version: t.version, Kind: t.kind},
		Scope:            meta.RESTScopeNamespace, // TODO: not all unknown types are namespaced
		ObjectConvertor:  &schemalessObjectConverter{ /*t.mapping.ObjectConvertor*/ },
		MetadataAccessor: meta.NewAccessor(),
	}, nil
}

func (t *schemalessDataMapper) RESTMappings(gk unversioned.GroupKind) ([]*meta.RESTMapping, error) {
	if gk.Group != t.group {
		return nil, fmt.Errorf("unknown group %q expected %s", gk.Group, t.group)
	}
	if gk.Kind != t.kind {
		return nil, fmt.Errorf("unknown kind %s expected %s", gk.Kind, t.kind)
	}

	return []*meta.RESTMapping{{
		Resource:         t.getResource().Resource,
		GroupVersionKind: unversioned.GroupVersionKind{Group: t.group, Version: t.version, Kind: t.kind},
		Scope:            meta.RESTScopeNamespace, // TODO: not all unknown types are namespaced
		ObjectConvertor:  &schemalessObjectConverter{ /*mapping.ObjectConvertor*/ },
	}}, nil
}

func (t *schemalessDataMapper) AliasesForResource(resource string) ([]string, bool) {
	return t.mapper.AliasesForResource(resource)
}

func (t *schemalessDataMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return t.mapper.ResourceSingularizer(resource)
}

func NewMapper(mapper meta.RESTMapper, kind, version, group string) meta.RESTMapper {
	return &schemalessDataMapper{
		mapper:  mapper,
		kind:    kind,
		version: version,
		group:   group,
	}
}

type schemalessCodecFactory struct {
	delegate runtime.NegotiatedSerializer
	kind     string
	encodeGV unversioned.GroupVersion
	decodeGV unversioned.GroupVersion
}

func NewNegotiatedSerializer(s runtime.NegotiatedSerializer, kind string, encodeGV, decodeGV unversioned.GroupVersion) runtime.NegotiatedSerializer {
	return &schemalessCodecFactory{
		delegate: s,
		kind:     kind,
		encodeGV: encodeGV,
		decodeGV: decodeGV,
	}
}

func (t *schemalessCodecFactory) SupportedMediaTypes() []string {
	supported := sets.NewString(t.delegate.SupportedMediaTypes()...)
	return supported.Intersection(sets.NewString("application/json", "application/yaml")).List()
}

func (t *schemalessCodecFactory) SerializerForMediaType(mediaType string, params map[string]string) (runtime.SerializerInfo, bool) {
	switch mediaType {
	case "application/json", "application/yaml":
		return t.delegate.SerializerForMediaType(mediaType, params)
	default:
		return runtime.SerializerInfo{}, false
	}
}

func (t *schemalessCodecFactory) SupportedStreamingMediaTypes() []string {
	supported := sets.NewString(t.delegate.SupportedStreamingMediaTypes()...)
	return supported.Intersection(sets.NewString("application/json", "application/json;stream=watch")).List()
}

func (t *schemalessCodecFactory) StreamingSerializerForMediaType(mediaType string, params map[string]string) (runtime.StreamSerializerInfo, bool) {
	switch mediaType {
	case "application/json", "application/json;stream=watch":
		return t.delegate.StreamingSerializerForMediaType(mediaType, params)
	default:
		return runtime.StreamSerializerInfo{}, false
	}
}

func (t *schemalessCodecFactory) EncoderForVersion(s runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return &schemalessDataEncoder{delegate: t.delegate.EncoderForVersion(s, gv), gvk: t.encodeGV.WithKind(t.kind)}
}

func (t *schemalessCodecFactory) DecoderToVersion(s runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return NewDecoder(t.delegate.DecoderToVersion(s, gv), t.kind)
}

func NewCodec(delegate runtime.Codec, gvk unversioned.GroupVersionKind) runtime.Codec {
	return runtime.NewCodec(NewEncoder(delegate, gvk), NewDecoder(delegate, gvk.Kind))
}

type schemalessDataDecoder struct {
	delegate runtime.Decoder
	kind     string
}

func NewDecoder(delegate runtime.Decoder, kind string) runtime.Decoder {
	return &schemalessDataDecoder{delegate: delegate, kind: kind}
}

var _ runtime.Decoder = &schemalessDataDecoder{}

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

func (t *schemalessDataDecoder) populate(data []byte) (runtime.Object, *unversioned.GroupVersionKind, error) {
	mapObj, err := parseObject(data)
	if err != nil {
		return nil, nil, err
	}
	return t.populateFromObject(mapObj, data)
}

func (t *schemalessDataDecoder) populateFromObject(mapObj map[string]interface{}, data []byte) (runtime.Object, *unversioned.GroupVersionKind, error) {
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
		result := &runtime.Unstructured{}
		if err := t.populateResource(result, mapObj, data); err != nil {
			return nil, nil, err
		}
		return result, &gvk, nil
	case isList && (len(t.kind) == 0 || typeMeta.Kind == t.kind+"List"):
		list := &api.List{}
		if err := t.populateListResource(list, mapObj); err != nil {
			return nil, nil, err
		}
		return list, &gvk, nil
	default:
		return nil, nil, fmt.Errorf("unexpected kind: %s, expected %s", typeMeta.Kind, t.kind)
	}
}

func (t *schemalessDataDecoder) populateResource(objIn *runtime.Unstructured, mapObj map[string]interface{}, data []byte) error {
	objIn.Object = mapObj
	return nil
}

func isSchemaless(rawData []byte, gvk *unversioned.GroupVersionKind) (isThirdParty bool, gvkOut *unversioned.GroupVersionKind, err error) {
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
	// TODO: this holds all the types we don't have go structs for, not just third party resources
	return registered.IsThirdPartyAPIGroupVersion(gv), gvkOut, nil
}

func (t *schemalessDataDecoder) Decode(data []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	if into == nil {
		if gvk == nil || gvk.Kind != t.kind {
			if isThirdParty, _, err := isSchemaless(data, gvk); err != nil {
				return nil, nil, err
			} else if !isThirdParty {
				return t.delegate.Decode(data, gvk, into)
			}
		}
		return t.populate(data)
	}

	switch o := into.(type) {
	case *runtime.Unstructured:
		break
	case *runtime.VersionedObjects:
		// We're not sure that it's third party, we need to test
		if gvk == nil || gvk.Kind != t.kind {
			if isThirdParty, _, err := isSchemaless(data, gvk); err != nil {
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
		return t.delegate.Decode(data, gvk, into)
	}

	unstructured := into.(*runtime.Unstructured)
	var dataObj map[string]interface{}
	if err := json.Unmarshal(data, &dataObj); err != nil {
		return nil, nil, err
	}
	unstructured.Object = dataObj

	actual := &unversioned.GroupVersionKind{}
	if kindStr := unstructured.GetKind(); kindStr == "" {
		if gvk == nil {
			return nil, nil, runtime.NewMissingKindErr(string(data))
		}
		unstructured.SetKind(gvk.Kind)
		actual.Kind = gvk.Kind
	} else {
		if kindStr != t.kind {
			return nil, nil, fmt.Errorf("kind doesn't match, expecting: %s, got %s", gvk.Kind, kindStr)
		}
		actual.Kind = kindStr
	}

	if versionStr := unstructured.GetAPIVersion(); versionStr == "" {
		if gvk == nil {
			return nil, nil, runtime.NewMissingVersionErr(string(data))
		}
		unstructured.SetAPIVersion(gvk.GroupVersion().String())
		actual.Group, actual.Version = gvk.Group, gvk.Version
	} else {
		if gvk != nil && versionStr != gvk.GroupVersion().String() {
			return nil, nil, fmt.Errorf("version doesn't match, expecting: %v, got %s", gvk.GroupVersion(), versionStr)
		}
		gv, err := unversioned.ParseGroupVersion(versionStr)
		if err != nil {
			return nil, nil, err
		}
		actual.Group, actual.Version = gv.Group, gv.Version
	}

	return unstructured, actual, nil
}

func (t *schemalessDataDecoder) populateListResource(objIn *api.List, mapObj map[string]interface{}) error {
	items, ok := mapObj["items"].([]interface{})
	if !ok {
		return fmt.Errorf("unexpected object for items: %#v", mapObj["items"])
	}
	objIn.Items = make([]runtime.Object, len(items))
	for ix := range items {
		objIn.Items[ix] = &runtime.Unstructured{Object: items[ix].(map[string]interface{})}
	}
	return nil
}

type schemalessDataEncoder struct {
	delegate runtime.Encoder
	gvk      unversioned.GroupVersionKind
}

func NewEncoder(delegate runtime.Encoder, gvk unversioned.GroupVersionKind) runtime.Encoder {
	return &schemalessDataEncoder{delegate: delegate, gvk: gvk}
}

var _ runtime.Encoder = &schemalessDataEncoder{}

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

func (t *schemalessDataEncoder) Encode(obj runtime.Object, stream io.Writer) (err error) {
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
			return fmt.Errorf("schemalessDataEncoder was not given a target version")
		}

		encMap := struct {
			Kind       string               `json:"kind,omitempty"`
			Items      []json.RawMessage    `json:"items"`
			Metadata   unversioned.ListMeta `json:"metadata,omitempty"`
			APIVersion string               `json:"apiVersion,omitempty"`
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
