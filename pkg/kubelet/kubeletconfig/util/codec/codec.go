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
	_ "k8s.io/kubernetes/pkg/api/install"
	// ensure the kubeletconfig apis are installed
	_ "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/install"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
)

// TODO(mtaufen): allow an encoder to be injected into checkpoint objects at creation time? (then we could ultimately instantiate only one encoder)

// NewJSONEncoder generates a new runtime.Encoder that encodes objects to JSON
func NewJSONEncoder(groupName string) (runtime.Encoder, error) {
	// encode to json
	mediaType := "application/json"
	info, ok := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}

	versions := api.Registry.EnabledVersionsForGroup(groupName)
	if len(versions) == 0 {
		return nil, fmt.Errorf("no enabled versions for group %q", groupName)
	}

	// the "best" version supposedly comes first in the list returned from api.Registry.EnabledVersionsForGroup
	return api.Codecs.EncoderForVersion(info.Serializer, versions[0]), nil
}

// DecodeKubeletConfiguration decodes an encoded (v1alpha1) KubeletConfiguration object to the internal type
func DecodeKubeletConfiguration(data []byte) (*kubeletconfig.KubeletConfiguration, error) {
	// decode the object, note we use the external version scheme to decode, because users provide the external version
	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(kubeletconfigv1alpha1.SchemeGroupVersion), data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode, error: %v", err)
	}

	externalKC, ok := obj.(*kubeletconfigv1alpha1.KubeletConfiguration)
	if !ok {
		return nil, fmt.Errorf("failed to cast object to KubeletConfiguration, object: %#v", obj)
	}

	// TODO(mtaufen): confirm whether api.Codecs.UniversalDecoder runs the defaulting, which would make this redundant
	// run the defaulter on the decoded configuration before converting to internal type
	api.Scheme.Default(externalKC)

	// convert to internal type
	internalKC := &kubeletconfig.KubeletConfiguration{}
	err = api.Scheme.Convert(externalKC, internalKC, nil)
	if err != nil {
		return nil, err
	}
	return internalKC, nil
}
