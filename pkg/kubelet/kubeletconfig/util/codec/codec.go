/*
Copyright 2017 The Kubernetes Authors.

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

package codec

import (
	"fmt"

	// ensure the core apis are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
)

// EncodeKubeletConfig encodes an internal KubeletConfiguration to an external YAML representation
func EncodeKubeletConfig(internal *kubeletconfig.KubeletConfiguration, targetVersion schema.GroupVersion) ([]byte, error) {
	encoder, err := NewKubeletconfigYAMLEncoder(targetVersion)
	if err != nil {
		return nil, err
	}
	// encoder will convert to external version
	data, err := runtime.Encode(encoder, internal)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// NewKubeletconfigYAMLEncoder returns an encoder that can write objects in the kubeletconfig API group to YAML
func NewKubeletconfigYAMLEncoder(targetVersion schema.GroupVersion) (runtime.Encoder, error) {
	_, codecs, err := scheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}
	return codecs.EncoderForVersion(info.Serializer, targetVersion), nil
}

// NewYAMLEncoder generates a new runtime.Encoder that encodes objects to YAML
func NewYAMLEncoder(groupName string) (runtime.Encoder, error) {
	// encode to YAML
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}

	versions := legacyscheme.Scheme.PrioritizedVersionsForGroup(groupName)
	if len(versions) == 0 {
		return nil, fmt.Errorf("no enabled versions for group %q", groupName)
	}

	// the "best" version supposedly comes first in the list returned from legacyscheme.Registry.EnabledVersionsForGroup
	return legacyscheme.Codecs.EncoderForVersion(info.Serializer, versions[0]), nil
}

// DecodeKubeletConfiguration decodes a serialized KubeletConfiguration to the internal type
func DecodeKubeletConfiguration(kubeletCodecs *serializer.CodecFactory, data []byte) (*kubeletconfig.KubeletConfiguration, error) {
	// the UniversalDecoder runs defaulting and returns the internal type by default
	obj, gvk, err := kubeletCodecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode, error: %v", err)
	}

	internalKC, ok := obj.(*kubeletconfig.KubeletConfiguration)
	if !ok {
		return nil, fmt.Errorf("failed to cast object to KubeletConfiguration, unexpected type: %v", gvk)
	}

	return internalKC, nil
}
