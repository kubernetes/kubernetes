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

package initializer

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
)

// WantsExternalKubeClientSet defines a function which sets external ClientSet for admission plugins that need it
type WantsExternalKubeClientSet interface {
	SetExternalKubeClientSet(kubernetes.Interface)
	admission.InitializationValidator
}

// WantsExternalKubeInformerFactory defines a function which sets InformerFactory for admission plugins that need it
type WantsExternalKubeInformerFactory interface {
	SetExternalKubeInformerFactory(informers.SharedInformerFactory)
	admission.InitializationValidator
}

// WantsAuthorizer defines a function which sets Authorizer for admission plugins that need it.
type WantsAuthorizer interface {
	SetAuthorizer(authorizer.Authorizer)
	admission.InitializationValidator
}

// WantsScheme defines a function that accepts runtime.Scheme for admission plugins that need it.
type WantsScheme interface {
	SetScheme(*runtime.Scheme)
	admission.InitializationValidator
}
