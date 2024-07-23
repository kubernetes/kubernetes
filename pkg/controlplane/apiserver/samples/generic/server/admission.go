/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission/plugin/namespace/lifecycle"
	validatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	"k8s.io/apiserver/pkg/admission/plugin/resourcequota"
	mutatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating"
	validatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/validating"
	"k8s.io/kubernetes/pkg/kubeapiserver/options"
	certapproval "k8s.io/kubernetes/plugin/pkg/admission/certificates/approval"
	"k8s.io/kubernetes/plugin/pkg/admission/certificates/ctbattest"
	certsigning "k8s.io/kubernetes/plugin/pkg/admission/certificates/signing"
	certsubjectrestriction "k8s.io/kubernetes/plugin/pkg/admission/certificates/subjectrestriction"
	"k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds"
	"k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
)

// DefaultOffAdmissionPlugins get admission plugins off by default for kube-apiserver.
func DefaultOffAdmissionPlugins() sets.Set[string] {
	defaultOnPlugins := sets.New[string](
		lifecycle.PluginName,                 // NamespaceLifecycle
		serviceaccount.PluginName,            // ServiceAccount
		defaulttolerationseconds.PluginName,  // DefaultTolerationSeconds
		mutatingwebhook.PluginName,           // MutatingAdmissionWebhook
		validatingwebhook.PluginName,         // ValidatingAdmissionWebhook
		resourcequota.PluginName,             // ResourceQuota
		certapproval.PluginName,              // CertificateApproval
		certsigning.PluginName,               // CertificateSigning
		ctbattest.PluginName,                 // ClusterTrustBundleAttest
		certsubjectrestriction.PluginName,    // CertificateSubjectRestriction
		validatingadmissionpolicy.PluginName, // ValidatingAdmissionPolicy, only active when feature gate ValidatingAdmissionPolicy is enabled
	)

	return sets.New(options.AllOrderedPlugins...).Difference(defaultOnPlugins)
}
