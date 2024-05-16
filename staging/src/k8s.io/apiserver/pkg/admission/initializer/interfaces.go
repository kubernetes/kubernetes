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
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
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

// WantsQuotaConfiguration defines a function which sets quota configuration for admission plugins that need it.
type WantsQuotaConfiguration interface {
	SetQuotaConfiguration(quota.Configuration)
	admission.InitializationValidator
}

// WantsDrainedNotification defines a function which sets the notification of where the apiserver
// has already been drained for admission plugins that need it.
// After receiving that notification, Admit/Validate calls won't be called anymore.
type WantsDrainedNotification interface {
	SetDrainedNotification(<-chan struct{})
	admission.InitializationValidator
}

// WantsFeatureGate defines a function which passes the featureGates for inspection by an admission plugin.
// Admission plugins should not hold a reference to the featureGates.  Instead, they should query a particular one
// and assign it to a simple bool in the admission plugin struct.
//
//	func (a *admissionPlugin) InspectFeatureGates(features featuregate.FeatureGate){
//	    a.myFeatureIsOn = features.Enabled("my-feature")
//	}
type WantsFeatures interface {
	InspectFeatureGates(featuregate.FeatureGate)
	admission.InitializationValidator
}

type WantsDynamicClient interface {
	SetDynamicClient(dynamic.Interface)
	admission.InitializationValidator
}

// WantsRESTMapper defines a function which sets RESTMapper for admission plugins that need it.
type WantsRESTMapper interface {
	SetRESTMapper(meta.RESTMapper)
	admission.InitializationValidator
}

// WantsSchemaResolver defines a function which sets the SchemaResolver for
// an admission plugin that needs it.
type WantsSchemaResolver interface {
	SetSchemaResolver(resolver resolver.SchemaResolver)
	admission.InitializationValidator
}

// WantsExcludedAdmissionResources defines a function which sets the ExcludedAdmissionResources
// for an admission plugin that needs it.
type WantsExcludedAdmissionResources interface {
	SetExcludedAdmissionResources(excludedAdmissionResources []schema.GroupResource)
	admission.InitializationValidator
}
