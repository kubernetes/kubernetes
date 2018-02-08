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

package scheme

import (
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/install"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
)

var (
	// Scheme defines methods for serializing and deserializing API objects.
	Scheme = runtime.NewScheme()
	// Codecs provides methods for retrieving codecs and serializers for specific
	// versions and content types.
	Codecs = serializer.NewCodecFactory(Scheme)
	// groupFactoryRegistry is the APIGroupFactoryRegistry.
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	// Registry is an instance of an API registry.  This is an interim step to start removing the idea of a global
	// API registry.
	Registry = registered.NewOrDie("")
)

func init() {
	AddToScheme(Scheme)
	install.Install(groupFactoryRegistry, Registry, Scheme)
}

// AddToScheme adds the types of this group into the given scheme.
func AddToScheme(scheme *runtime.Scheme) {
	v1beta1.AddToScheme(scheme)
	v1.AddToScheme(scheme)
	apiregistration.AddToScheme(scheme)
}
