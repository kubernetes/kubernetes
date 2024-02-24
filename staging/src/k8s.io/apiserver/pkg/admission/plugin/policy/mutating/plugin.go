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

package mutating

import (
	"context"
	"io"

	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
)

const (
	// PluginName indicates the name of admission plug-in
	PluginName = "MutatingAdmissionPolicy"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(configFile io.Reader) (admission.Interface, error) {
		return NewPlugin(configFile), nil
	})
}

// Plugin is an implementation of admission.Interface.
type Policy = v1alpha1.MutatingAdmissionPolicy
type PolicyBinding = v1alpha1.MutatingAdmissionPolicyBinding
type PolicyMutation = v1alpha1.Mutation
type PolicyHook = generic.PolicyHook[*Policy, *PolicyBinding, PolicyEvaluator]
type MutationEvaluationFunc func(
	ctx context.Context,
	matchedResource schema.GroupVersionResource,
	versionedAttr *admission.VersionedAttributes,
	o admission.ObjectInterfaces,
	versionedParams runtime.Object,
	namespace *corev1.Namespace,
	typeConverter managedfields.TypeConverter,
	runtimeCELCostBudget int64,
) (runtime.Object, error)
type PolicyEvaluator = []MutationEvaluationFunc

type Plugin struct {
	*generic.Plugin[PolicyHook]
}

var _ admission.Interface = &Plugin{}
var _ admission.MutationInterface = &Plugin{}

// NewMutatingWebhook returns a generic admission webhook plugin.
func NewPlugin(_ io.Reader) *Plugin {
	handler := admission.NewHandler(admission.Connect, admission.Create, admission.Delete, admission.Update)
	res := &Plugin{}
	res.Plugin = generic.NewPlugin(
		handler,
		func(f informers.SharedInformerFactory, client kubernetes.Interface, dynamicClient dynamic.Interface, restMapper meta.RESTMapper) generic.Source[PolicyHook] {
			return generic.NewPolicySource(
				f.Admissionregistration().V1alpha1().MutatingAdmissionPolicies().Informer(),
				f.Admissionregistration().V1alpha1().MutatingAdmissionPolicyBindings().Informer(),
				NewMutatingAdmissionPolicyAccessor,
				NewMutatingAdmissionPolicyBindingAccessor,
				compilePolicy,
				//!TODO: Create a way to share param informers between
				// mutating/validating plugins
				f,
				dynamicClient,
				restMapper,
			)
		},
		func(a authorizer.Authorizer, m *matching.Matcher, client kubernetes.Interface) generic.Dispatcher[PolicyHook] {
			return NewDispatcher(a, m, patch.NewTypeConverterManager(nil, client.Discovery().OpenAPIV3()))
		},
	)
	return res
}

// Admit makes an admission decision based on the request attributes.
func (a *Plugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	return a.Plugin.Dispatch(ctx, attr, o)
}

func (a *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	a.Plugin.SetEnabled(featureGates.Enabled(features.MutatingAdmissionPolicy))
}
