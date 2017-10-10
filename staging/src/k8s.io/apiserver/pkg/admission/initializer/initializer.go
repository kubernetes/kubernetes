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

type pluginInitializer struct {
	externalClient    kubernetes.Interface
	externalInformers informers.SharedInformerFactory
	authorizer        authorizer.Authorizer
	// serverIdentifyingClientCert used to provide identity when calling out to admission plugins
	serverIdentifyingClientCert []byte
	// serverIdentifyingClientKey private key for the client certificate used when calling out to admission plugins
	serverIdentifyingClientKey []byte
	scheme                     *runtime.Scheme
}

// New creates an instance of admission plugins initializer.
// TODO(p0lyn0mial): make the parameters public, this construction seems to be redundant.
func New(
	extClientset kubernetes.Interface,
	extInformers informers.SharedInformerFactory,
	authz authorizer.Authorizer,
	serverIdentifyingClientCert,
	serverIdentifyingClientKey []byte,
	scheme *runtime.Scheme,
) (pluginInitializer, error) {
	return pluginInitializer{
		externalClient:              extClientset,
		externalInformers:           extInformers,
		authorizer:                  authz,
		serverIdentifyingClientCert: serverIdentifyingClientCert,
		serverIdentifyingClientKey:  serverIdentifyingClientKey,
		scheme: scheme,
	}, nil
}

// Initialize checks the initialization interfaces implemented by a plugin
// and provide the appropriate initialization data
func (i pluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsExternalKubeClientSet); ok {
		wants.SetExternalKubeClientSet(i.externalClient)
	}

	if wants, ok := plugin.(WantsExternalKubeInformerFactory); ok {
		wants.SetExternalKubeInformerFactory(i.externalInformers)
	}

	if wants, ok := plugin.(WantsAuthorizer); ok {
		wants.SetAuthorizer(i.authorizer)
	}

	if wants, ok := plugin.(WantsClientCert); ok {
		wants.SetClientCert(i.serverIdentifyingClientCert, i.serverIdentifyingClientKey)
	}

	if wants, ok := plugin.(WantsScheme); ok {
		wants.SetScheme(i.scheme)
	}
}

var _ admission.PluginInitializer = pluginInitializer{}
