/*
Copyright 2019 The Kubernetes Authors.

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

package crd

import (
	"fmt"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// RuntimeClassCRD returns the CRD defining the node-api.k8s.io/RuntimeClass resource.
func RuntimeClassCRD() (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	raw, err := pkgApisNodeCrdRuntimeclass_crdYamlBytes()
	if err != nil {
		return nil, err
	}

	return decodeCRD(raw)
}

func decodeCRD(raw []byte) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	crd := &apiextensionsv1beta1.CustomResourceDefinition{}

	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), raw, crd); err != nil {
		return nil, fmt.Errorf("failed to decode CRD: %v", err)
	}

	return crd, nil
}
