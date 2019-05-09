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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
)

var watchScheme = runtime.NewScheme()
var basicScheme = runtime.NewScheme()
var deleteScheme = runtime.NewScheme()
var parameterScheme = runtime.NewScheme()
var deleteOptionsCodec = serializer.NewCodecFactory(deleteScheme)
var dynamicParameterCodec = runtime.NewParameterCodec(parameterScheme)

var versionV1 = schema.GroupVersion{Version: "v1"}

func init() {
	metav1.AddToGroupVersion(watchScheme, versionV1)
	metav1.AddToGroupVersion(basicScheme, versionV1)
	metav1.AddToGroupVersion(parameterScheme, versionV1)
	metav1.AddToGroupVersion(deleteScheme, versionV1)
}

var watchJsonSerializerInfo = runtime.SerializerInfo{
	MediaType:        "application/json",
	MediaTypeType:    "application",
	MediaTypeSubType: "json",
	EncodesAsText:    true,
	Serializer:       json.NewSerializer(json.DefaultMetaFactory, watchScheme, watchScheme, false),
	PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, watchScheme, watchScheme, true),
	StreamSerializer: &runtime.StreamSerializerInfo{
		EncodesAsText: true,
		Serializer:    json.NewSerializer(json.DefaultMetaFactory, watchScheme, watchScheme, false),
		Framer:        json.Framer,
	},
}

// watchNegotiatedSerializer is used to read the wrapper of the watch stream
type watchNegotiatedSerializer struct{}

var watchNegotiatedSerializerInstance = watchNegotiatedSerializer{}

func (s watchNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{watchJsonSerializerInfo}
}

func (s watchNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewDefaultingCodecForScheme(watchScheme, encoder, nil, gv, nil)
}

func (s watchNegotiatedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return versioning.NewDefaultingCodecForScheme(watchScheme, nil, decoder, nil, gv)
}

// basicNegotiatedSerializer is used to handle discovery and error handling serialization
type basicNegotiatedSerializer struct{}

func (s basicNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{
		{
			MediaType:        "application/json",
			MediaTypeType:    "application",
			MediaTypeSubType: "json",
			EncodesAsText:    true,
			Serializer:       json.NewSerializer(json.DefaultMetaFactory, basicScheme, basicScheme, false),
			PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, basicScheme, basicScheme, true),
			StreamSerializer: &runtime.StreamSerializerInfo{
				EncodesAsText: true,
				Serializer:    json.NewSerializer(json.DefaultMetaFactory, basicScheme, basicScheme, false),
				Framer:        json.Framer,
			},
		},
	}
}

func (s basicNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewDefaultingCodecForScheme(watchScheme, encoder, nil, gv, nil)
}

func (s basicNegotiatedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return versioning.NewDefaultingCodecForScheme(watchScheme, nil, decoder, nil, gv)
}
