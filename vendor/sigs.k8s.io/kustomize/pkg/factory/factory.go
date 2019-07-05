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
// Package factory provides factories for kustomize.
package factory

import (
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/ifc/transformer"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
)

// KustFactory provides different factories for kustomize
type KustFactory struct {
	ResmapF      *resmap.Factory
	TransformerF transformer.Factory
	ValidatorF   ifc.Validator
	UnstructF    ifc.KunstructuredFactory
}

// NewKustFactory creats a KustFactory instance
func NewKustFactory(u ifc.KunstructuredFactory, v ifc.Validator, t transformer.Factory) *KustFactory {
	return &KustFactory{
		ResmapF:      resmap.NewFactory(resource.NewFactory(u)),
		TransformerF: t,
		ValidatorF:   v,
		UnstructF:    u,
	}
}
