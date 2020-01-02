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

package unstructuredscheme

import (
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

// NewUnstructuredNegotiatedSerializer returns a simple, negotiated serializer
func NewUnstructuredNegotiatedSerializer() runtime.NegotiatedSerializer {
	return unstructuredNegotiatedSerializer{
		scheme:  scheme,
		typer:   NewUnstructuredObjectTyper(),
		creator: NewUnstructuredCreator(),
	}
}

type unstructuredNegotiatedSerializer struct {
	scheme  *runtime.Scheme
	typer   runtime.ObjectTyper
	creator runtime.ObjectCreater
}

func (s unstructuredNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{
		{
			MediaType:        "application/json",
			MediaTypeType:    "application",
			MediaTypeSubType: "json",
			EncodesAsText:    true,
			Serializer:       json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, false),
			PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, true),
			StreamSerializer: &runtime.StreamSerializerInfo{
				EncodesAsText: true,
				Serializer:    json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, false),
				Framer:        json.Framer,
			},
		},
		{
			MediaType:        "application/yaml",
			MediaTypeType:    "application",
			MediaTypeSubType: "yaml",
			EncodesAsText:    true,
			Serializer:       json.NewYAMLSerializer(json.DefaultMetaFactory, s.creator, s.typer),
		},
	}
}

func (s unstructuredNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewDefaultingCodecForScheme(s.scheme, encoder, nil, gv, nil)
}

func (s unstructuredNegotiatedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return versioning.NewDefaultingCodecForScheme(s.scheme, nil, decoder, nil, gv)
}

type unstructuredObjectTyper struct {
}

// NewUnstructuredObjectTyper returns an object typer that can deal with unstructured things
func NewUnstructuredObjectTyper() runtime.ObjectTyper {
	return unstructuredObjectTyper{}
}

func (t unstructuredObjectTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	// Delegate for things other than Unstructured.
	if _, ok := obj.(runtime.Unstructured); !ok {
		return nil, false, fmt.Errorf("cannot type %T", obj)
	}
	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, false, runtime.NewMissingKindErr("object has no kind field ")
	}
	if len(gvk.Version) == 0 {
		return nil, false, runtime.NewMissingVersionErr("object has no apiVersion field")
	}

	return []schema.GroupVersionKind{obj.GetObjectKind().GroupVersionKind()}, false, nil
}

func (t unstructuredObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return true
}

type unstructuredCreator struct{}

// NewUnstructuredCreator returns a simple object creator that always returns an unstructured
func NewUnstructuredCreator() runtime.ObjectCreater {
	return unstructuredCreator{}
}

func (c unstructuredCreator) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	ret := &unstructured.Unstructured{}
	ret.SetGroupVersionKind(kind)
	return ret, nil
}

type unstructuredDefaulter struct {
}

// NewUnstructuredDefaulter returns defaulter suitable for unstructured types that doesn't default anything
func NewUnstructuredDefaulter() runtime.ObjectDefaulter {
	return unstructuredDefaulter{}
}

func (d unstructuredDefaulter) Default(in runtime.Object) {
}
