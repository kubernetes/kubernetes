/*
Copyright 2018 The Kubernetes Authors.

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

package dynamic

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/client-go/features"
)

var basicScheme = runtime.NewScheme()
var parameterScheme = runtime.NewScheme()
var dynamicParameterCodec = runtime.NewParameterCodec(parameterScheme)

var versionV1 = schema.GroupVersion{Version: "v1"}

func init() {
	metav1.AddToGroupVersion(basicScheme, versionV1)
	metav1.AddToGroupVersion(parameterScheme, versionV1)
}

func newBasicNegotiatedSerializer() basicNegotiatedSerializer {
	supportedMediaTypes := []runtime.SerializerInfo{
		{
			MediaType:        "application/json",
			MediaTypeType:    "application",
			MediaTypeSubType: "json",
			EncodesAsText:    true,
			Serializer:       json.NewSerializerWithOptions(json.DefaultMetaFactory, unstructuredCreater{basicScheme}, unstructuredTyper{basicScheme}, json.SerializerOptions{}),
			PrettySerializer: json.NewSerializerWithOptions(json.DefaultMetaFactory, unstructuredCreater{basicScheme}, unstructuredTyper{basicScheme}, json.SerializerOptions{Pretty: true}),
			StreamSerializer: &runtime.StreamSerializerInfo{
				EncodesAsText: true,
				Serializer:    json.NewSerializerWithOptions(json.DefaultMetaFactory, basicScheme, basicScheme, json.SerializerOptions{}),
				Framer:        json.Framer,
			},
		},
	}
	if features.FeatureGates().Enabled(features.ClientsAllowCBOR) {
		supportedMediaTypes = append(supportedMediaTypes, runtime.SerializerInfo{
			MediaType:        "application/cbor",
			MediaTypeType:    "application",
			MediaTypeSubType: "cbor",
			Serializer:       cbor.NewSerializer(unstructuredCreater{basicScheme}, unstructuredTyper{basicScheme}),
			StreamSerializer: &runtime.StreamSerializerInfo{
				Serializer: cbor.NewSerializer(basicScheme, basicScheme, cbor.Transcode(false)),
				Framer:     cbor.NewFramer(),
			},
		})
	}
	return basicNegotiatedSerializer{supportedMediaTypes: supportedMediaTypes}
}

type basicNegotiatedSerializer struct {
	supportedMediaTypes []runtime.SerializerInfo
}

func (s basicNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return s.supportedMediaTypes
}

func (s basicNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return runtime.WithVersionEncoder{
		Version:     gv,
		Encoder:     encoder,
		ObjectTyper: permissiveTyper{basicScheme},
	}
}

func (s basicNegotiatedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return decoder
}

type unstructuredCreater struct {
	nested runtime.ObjectCreater
}

func (c unstructuredCreater) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	out, err := c.nested.New(kind)
	if err == nil {
		return out, nil
	}
	out = &unstructured.Unstructured{}
	out.GetObjectKind().SetGroupVersionKind(kind)
	return out, nil
}

type unstructuredTyper struct {
	nested runtime.ObjectTyper
}

func (t unstructuredTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	kinds, unversioned, err := t.nested.ObjectKinds(obj)
	if err == nil {
		return kinds, unversioned, nil
	}
	if _, ok := obj.(runtime.Unstructured); ok && !obj.GetObjectKind().GroupVersionKind().Empty() {
		return []schema.GroupVersionKind{obj.GetObjectKind().GroupVersionKind()}, false, nil
	}
	return nil, false, err
}

func (t unstructuredTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return true
}

// The dynamic client has historically accepted Unstructured objects with missing or empty
// apiVersion and/or kind as arguments to its write request methods. This typer will return the type
// of a runtime.Unstructured with no error, even if the type is missing or empty.
type permissiveTyper struct {
	nested runtime.ObjectTyper
}

func (t permissiveTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	kinds, unversioned, err := t.nested.ObjectKinds(obj)
	if err == nil {
		return kinds, unversioned, nil
	}
	if _, ok := obj.(runtime.Unstructured); ok {
		return []schema.GroupVersionKind{obj.GetObjectKind().GroupVersionKind()}, false, nil
	}
	return nil, false, err
}

func (t permissiveTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return true
}
