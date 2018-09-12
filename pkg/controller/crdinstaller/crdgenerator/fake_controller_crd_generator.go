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

package crdgenerator

import (
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

// NewFakeControllerCRDGenerator returns a fake new instance of
// ControllerCRDGenerator.
func NewFakeControllerCRDGenerator() ControllerCRDGenerator {
	return &fakeControllerCRDGenerator{}
}

var _ ControllerCRDGenerator = (*fakeControllerCRDGenerator)(nil)

type fakeControllerCRDGenerator struct {
}

// GetCRDs returns the CRDs required by the fake controller.
func (adcCRDGen *fakeControllerCRDGenerator) GetCRDs() []*apiextensionsv1beta1.CustomResourceDefinition {
	var fakeCRDs []*apiextensionsv1beta1.CustomResourceDefinition
	return fakeCRDs
}
