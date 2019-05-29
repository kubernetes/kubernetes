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
	"net/url"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/webhook"
)

// WantsServiceResolver defines a function that accepts a ServiceResolver for
// admission plugins that need to make calls to services.
type WantsServiceResolver interface {
	SetServiceResolver(webhook.ServiceResolver)
}

// ServiceResolver knows how to convert a service reference into an actual
// location.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string, port int32) (*url.URL, error)
}

// WantsAuthenticationInfoResolverWrapper defines a function that wraps the standard AuthenticationInfoResolver
// to allow the apiserver to control what is returned as auth info
type WantsAuthenticationInfoResolverWrapper interface {
	SetAuthenticationInfoResolverWrapper(wrapper webhook.AuthenticationInfoResolverWrapper)
	admission.InitializationValidator
}

// PluginInitializer is used for initialization of the webhook admission plugin.
type PluginInitializer struct {
	serviceResolver                   webhook.ServiceResolver
	authenticationInfoResolverWrapper webhook.AuthenticationInfoResolverWrapper
}

var _ admission.PluginInitializer = &PluginInitializer{}

// NewPluginInitializer constructs new instance of PluginInitializer
func NewPluginInitializer(
	authenticationInfoResolverWrapper webhook.AuthenticationInfoResolverWrapper,
	serviceResolver webhook.ServiceResolver,
) *PluginInitializer {
	return &PluginInitializer{
		authenticationInfoResolverWrapper: authenticationInfoResolverWrapper,
		serviceResolver:                   serviceResolver,
	}
}

// Initialize checks the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *PluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsServiceResolver); ok {
		wants.SetServiceResolver(i.serviceResolver)
	}

	if wants, ok := plugin.(WantsAuthenticationInfoResolverWrapper); ok {
		if i.authenticationInfoResolverWrapper != nil {
			wants.SetAuthenticationInfoResolverWrapper(i.authenticationInfoResolverWrapper)
		}
	}
}
