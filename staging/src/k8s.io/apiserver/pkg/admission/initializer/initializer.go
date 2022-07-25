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
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
)

type pluginInitializer struct {
	externalClient    kubernetes.Interface
	externalInformers informers.SharedInformerFactory
	authorizer        authorizer.Authorizer
	featureGates      featuregate.FeatureGate
	stopCh            <-chan struct{}
}

// New creates an instance of admission plugins initializer.
// This constructor is public with a long param list so that callers immediately know that new information can be expected
// during compilation when they update a level.
func New(
	extClientset kubernetes.Interface,
	extInformers informers.SharedInformerFactory,
	authz authorizer.Authorizer,
	featureGates featuregate.FeatureGate,
	stopCh <-chan struct{},
) pluginInitializer {
	return pluginInitializer{
		externalClient:    extClientset,
		externalInformers: extInformers,
		authorizer:        authz,
		featureGates:      featureGates,
		stopCh:            stopCh,
	}
}

// Initialize checks the initialization interfaces implemented by a plugin
// and provide the appropriate initialization data
func (i pluginInitializer) Initialize(plugin admission.Interface) {
	// First tell the plugin about drained notification, so it can pass it to further initializations.
	if wants, ok := plugin.(WantsDrainedNotification); ok {
		wants.SetDrainedNotification(i.stopCh)
	}

	// Second tell the plugin about enabled features, so it can decide whether to start informers or not
	if wants, ok := plugin.(WantsFeatures); ok {
		wants.InspectFeatureGates(i.featureGates)
	}

	if wants, ok := plugin.(WantsExternalKubeClientSet); ok {
		wants.SetExternalKubeClientSet(i.externalClient)
	}

	if wants, ok := plugin.(WantsExternalKubeInformerFactory); ok {
		wants.SetExternalKubeInformerFactory(i.externalInformers)
	}

	if wants, ok := plugin.(WantsAuthorizer); ok {
		wants.SetAuthorizer(i.authorizer)
	}
}

var _ admission.PluginInitializer = pluginInitializer{}
