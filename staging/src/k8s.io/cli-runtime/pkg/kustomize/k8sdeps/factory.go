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

// Package k8sdeps provides kustomize factory with k8s dependencies
package k8sdeps

import (
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kunstruct"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/transformer"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/validator"
	"sigs.k8s.io/kustomize/pkg/factory"
)

// NewFactory creates an instance of KustFactory using k8sdeps factories
func NewFactory() *factory.KustFactory {
	return factory.NewKustFactory(
		kunstruct.NewKunstructuredFactoryImpl(),
		validator.NewKustValidator(),
		transformer.NewFactoryImpl(),
	)
}
