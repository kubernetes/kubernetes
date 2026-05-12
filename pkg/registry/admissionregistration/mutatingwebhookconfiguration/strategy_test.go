/*
Copyright 2021 The Kubernetes Authors.

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

package mutatingwebhookconfiguration

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func validMutatingWebhookConfiguration() *admissionregistration.MutatingWebhookConfiguration {
	ignore := admissionregistration.Ignore
	exact := admissionregistration.Exact
	thirty := int32(30)
	none := admissionregistration.SideEffectClassNone
	servicePath := "/"
	return &admissionregistration.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Webhooks: []admissionregistration.MutatingWebhook{{
			Name: "foo.example.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Name:      "foo",
					Namespace: "bar",
					Path:      &servicePath,
					Port:      443,
				},
			},
			FailurePolicy:           &ignore,
			MatchPolicy:             &exact,
			TimeoutSeconds:          &thirty,
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			SideEffects:             &none,
			AdmissionReviewVersions: []string{"v1beta1"},
		}},
	}
}

func TestStaticSuffixWarningsAndValidation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	staticName := "my-webhook.static.k8s.io"

	makeConfig := func(name string) *admissionregistration.MutatingWebhookConfiguration {
		cfg := validMutatingWebhookConfiguration()
		cfg.Name = name
		return cfg
	}

	t.Run("feature gate disabled warns on create", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := Strategy.WarningsOnCreate(ctx, makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix when feature gate is disabled")
		}
	})

	t.Run("feature gate disabled warns on update", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := Strategy.WarningsOnUpdate(ctx, makeConfig(staticName), makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix when feature gate is disabled")
		}
	})

	t.Run("feature gate disabled no warning for normal name", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, false)
		warnings := Strategy.WarningsOnCreate(ctx, makeConfig("normal-webhook"))
		if len(warnings) != 0 {
			t.Errorf("Expected no warnings for normal name, got: %v", warnings)
		}
	})

	t.Run("feature gate enabled rejects static suffix on create", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
		errs := Strategy.Validate(ctx, makeConfig(staticName))
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
		warnings := Strategy.WarningsOnCreate(ctx, makeConfig(staticName))
		if len(warnings) != 0 {
			t.Errorf("Expected no warnings when feature gate is enabled (validation handles it), got: %v", warnings)
		}
	})

	t.Run("feature gate enabled warns on update", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
		warnings := Strategy.WarningsOnUpdate(ctx, makeConfig(staticName), makeConfig(staticName))
		if len(warnings) == 0 {
			t.Error("Expected warning for .static.k8s.io suffix on update even when feature gate is enabled")
		}
	})
}
