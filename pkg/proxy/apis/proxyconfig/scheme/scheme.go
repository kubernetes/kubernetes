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

package scheme

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/proxy/apis/proxyconfig"
	"k8s.io/kubernetes/pkg/proxy/apis/proxyconfig/v1alpha1"
)

// Utility functions for the Kube-proxy's proxyconfig API group

// NewSchemeAndCodecs is a utility funciton that returns a Scheme and CodecFactory
// that understand the types in the proxyconfig API group.
func NewSchemeAndCodecs() (*runtime.Scheme, *serializer.CodecFactory, error) {
	scheme := runtime.NewScheme()
	if err := proxyconfig.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		return nil, nil, err
	}
	codecs := serializer.NewCodecFactory(scheme)
	return scheme, &codecs, nil
}
