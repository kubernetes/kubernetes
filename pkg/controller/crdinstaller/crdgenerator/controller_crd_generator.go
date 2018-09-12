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

// Package crdgenerator defines the interface for generating CRDs.
// Implemented by Kubernetes controllers that require CRD installation.
// Consumed by the CRDInstallationController.
package crdgenerator

import (
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

// ControllerCRDGenerator defines the operations for generating CRDs.
type ControllerCRDGenerator interface {
	// GetCRDs returns the CRDs this component requires to install.
	GetCRDs() []*apiextensionsv1beta1.CustomResourceDefinition
}
