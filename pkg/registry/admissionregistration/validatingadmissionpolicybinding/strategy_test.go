/*
Copyright 2022 The Kubernetes Authors.

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

package validatingadmissionpolicybinding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"

	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func TestPolicyBindingStrategy(t *testing.T) {
	strategy := NewStrategy(nil, nil, replicaLimitsResolver)
	ctx := genericapirequest.NewDefaultContext()
	if strategy.NamespaceScoped() {
		t.Error("PolicyBinding strategy must be cluster scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("PolicyBinding should not allow create on update")
	}

	for _, configuration := range validPolicyBindings() {
		strategy.PrepareForCreate(ctx, configuration)
		errs := strategy.Validate(ctx, configuration)
		if len(errs) != 0 {
			t.Errorf("Unexpected error validating %v", errs)
		}
		invalidConfiguration := &admissionregistration.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: ""},
		}
		strategy.PrepareForUpdate(ctx, invalidConfiguration, configuration)
		errs = strategy.ValidateUpdate(ctx, invalidConfiguration, configuration)
		if len(errs) == 0 {
			t.Errorf("Expected a validation error")
		}
	}
}

var replicaLimitsResolver resolver.ResourceResolverFunc = func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	return schema.GroupVersionResource{
		Group:    "rules.example.com",
		Version:  "v1",
		Resource: "replicalimits",
	}, nil
}

func validPolicyBindings() []*admissionregistration.ValidatingAdmissionPolicyBinding {
	denyAction := admissionregistration.DenyAction
	return []*admissionregistration.ValidatingAdmissionPolicyBinding{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "replica-limit-test.example.com",
					ParameterNotFoundAction: &denyAction,
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-clusterwide",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "replica-limit-test.example.com",
					Namespace:               "default",
					ParameterNotFoundAction: &denyAction,
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-selector",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: &denyAction,
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-selector-clusterwide",
			},
			Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Namespace: "mynamespace",
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: &denyAction,
				},
				ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
			},
		},
	}
}

func validPolicyBinding() *admissionregistration.ValidatingAdmissionPolicyBinding {
	return validPolicyBindings()[0]
}

func TestStaticSuffixWarningsAndValidation(t *testing.T) {
	strategy := NewStrategy(nil, nil, replicaLimitsResolver)
	ctx := genericapirequest.NewDefaultContext()
	staticName := "my-binding.static.k8s.io"

	makeConfig := func(name string) *admissionregistration.ValidatingAdmissionPolicyBinding {
		cfg := validPolicyBinding()
		cfg.Name = name
		return cfg
	}

	t.Run("feature gate disabled warns on create", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := strategy.WarningsOnCreate(ctx, makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix when feature gate is disabled")
		}
	})

	t.Run("feature gate disabled warns on update", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := strategy.WarningsOnUpdate(ctx, makeConfig(staticName), makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix when feature gate is disabled")
		}
	})

	t.Run("feature gate disabled no warning for normal name", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := strategy.WarningsOnCreate(ctx, makeConfig("normal-binding"))
		if len(warnings) != 0 {
			t.Errorf("Expected no warnings for normal name, got: %v", warnings)
		}
	})

	t.Run("feature gate enabled rejects static suffix on create", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
		errs := strategy.Validate(ctx, makeConfig(staticName))
		found := false
		for _, e := range errs {
			if e.Field == "metadata.name" {
				found = true
				break
			}
		}
		if !found {
			t.Error("Expected validation error for .static.k8s.io suffix when feature gate is enabled")
		}
	})

	t.Run("feature gate enabled no warning on create", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
		warnings := strategy.WarningsOnCreate(ctx, makeConfig(staticName))
		if len(warnings) != 0 {
			t.Errorf("Expected no warnings when feature gate is enabled (validation handles it), got: %v", warnings)
		}
	})

	t.Run("feature gate enabled warns on update", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
		warnings := strategy.WarningsOnUpdate(ctx, makeConfig(staticName), makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix on update even when feature gate is enabled")
		}
	})
}
