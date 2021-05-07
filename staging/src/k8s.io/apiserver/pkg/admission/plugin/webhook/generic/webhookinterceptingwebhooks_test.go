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

package generic

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
	"k8s.io/apiserver/pkg/authentication/user"
)

type mockDispatcher struct {
	dispatchHooks []webhook.WebhookAccessor
	*admission.Handler
}

func (m *mockDispatcher) Dispatch(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces, hooks []webhook.WebhookAccessor) error {
	m.dispatchHooks = hooks
	return nil
}

var _ Dispatcher = &mockDispatcher{}

type mockAttribute struct {
	name string
	gvk  schema.GroupVersionKind
	admission.Attributes
	userInfo user.Info
}

func (a *mockAttribute) GetName() string {
	return a.name
}
func (a *mockAttribute) GetKind() schema.GroupVersionKind {
	return a.gvk
}
func (a *mockAttribute) GetUserInfo() user.Info {
	return a.userInfo
}

var _ admission.Attributes = &mockAttribute{}

type mockSource struct {
	hooks []webhook.WebhookAccessor
	Source
}

func (m *mockSource) Webhooks() []webhook.WebhookAccessor {
	return m.hooks
}

var _ Source = &mockSource{}

type mockUserInfo struct {
	user.Info
	name   string
	groups []string
}

func (u *mockUserInfo) GetName() string     { return u.name }
func (u *mockUserInfo) GetGroups() []string { return u.groups }

var _ user.Info = &mockUserInfo{}

type mockWebhookAccessor struct {
	Name       string
	ConfName   string
	IsMutating bool
	webhook.WebhookAccessor
}

func (m *mockWebhookAccessor) GetName() string {
	return m.Name
}

func (m *mockWebhookAccessor) GetMutatingWebhook() (*v1.MutatingWebhook, bool) {
	return nil, m.IsMutating
}

func (m *mockWebhookAccessor) GetConfigurationName() string {
	return m.ConfName
}

var _ webhook.WebhookAccessor = &mockWebhookAccessor{}

func TestPrecomputeWebhookInterceptingWebhooksConfig(t *testing.T) {
	for name, tc := range map[string]struct {
		webhookConfig  *webhookadmission.WebhookAdmission
		expectedConfig *webhookInterceptingWebhooksPrecomputedConfig
	}{
		"ValidConfig_ReturnsExpected": {
			webhookConfig: &webhookadmission.WebhookAdmission{
				WebhookInterceptingWebhooks: &webhookadmission.WebhookInterceptingWebhooks{
					Identifiers: []webhookadmission.WebhookIdentifier{
						{ConfigurationName: "config-a", Name: "webhook-a", Type: webhookadmission.Validating},
						{ConfigurationName: "config-b", Name: "webhook-a", Type: webhookadmission.Mutating},
					},
					Maintainers: webhookadmission.WebhookMaintainers{
						Users:  []string{"user-a", "user-b"},
						Groups: []string{"group-x", "group-y"},
					},
				}},
			expectedConfig: &webhookInterceptingWebhooksPrecomputedConfig{
				maintainerUsersSet:  sets.NewString("user-a", "user-b"),
				maintainerGroupsSet: sets.NewString("group-x", "group-y"),
				idSet: sets.NewString(
					`Validating\0config-a\0webhook-a`,
					`Mutating\0config-b\0webhook-a`),
			},
		},
		"ValidConfigEmptyWebhookInterceptingWebhooks_ReturnsEmpty": {
			webhookConfig: &webhookadmission.WebhookAdmission{
				WebhookInterceptingWebhooks: &webhookadmission.WebhookInterceptingWebhooks{},
			},
			expectedConfig: &webhookInterceptingWebhooksPrecomputedConfig{},
		},
		"ValidConfigNilWebhookInterceptingWebhooks_ReturnsNil": {
			webhookConfig:  &webhookadmission.WebhookAdmission{},
			expectedConfig: nil,
		},
		"NilConfig_ReturnsNil": {
			webhookConfig:  nil,
			expectedConfig: nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			observedConfig := precomputeWebhookInterceptingWebhooksConfig(tc.webhookConfig)
			if diff := cmp.Diff(tc.expectedConfig, observedConfig, cmpopts.IgnoreUnexported(webhookInterceptingWebhooksPrecomputedConfig{})); diff != "" {
				t.Errorf("precomputeWebhookInterceptingWebhooksConfig (...): -want set, +got set:\n%s", diff)
			}
			if tc.expectedConfig != nil && observedConfig != nil {
				if diff := cmp.Diff(tc.expectedConfig.maintainerUsersSet, observedConfig.maintainerUsersSet); diff != "" {
					t.Errorf("precomputeWebhookInterceptingWebhooksConfig (...): -want maintainerUsersSet set, +got maintainerUsersSet set:\n%s", diff)
				}
				if diff := cmp.Diff(tc.expectedConfig.maintainerGroupsSet, observedConfig.maintainerGroupsSet); diff != "" {
					t.Errorf("precomputeWebhookInterceptingWebhooksConfig (...): -want maintainerGroupsSet set, +got maintainerGroupsSet set:\n%s", diff)
				}
				if diff := cmp.Diff(tc.expectedConfig.idSet, observedConfig.idSet); diff != "" {
					t.Errorf("precomputeWebhookInterceptingWebhooksConfig (...): -want idSet set, +got idSet set:\n%s", diff)
				}
			}
		})
	}
}

func TestWebhookInterceptingWebhooks(t *testing.T) {
	for name, tc := range map[string]struct {
		registeredHooks []webhook.WebhookAccessor
		config          *webhookInterceptingWebhooksPrecomputedConfig
		expectedHooks   []webhook.WebhookAccessor
	}{
		"WebhookInterceptingWebhooksConfigNil_ReturnsNil": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "m-webhook-a"},
			},
			config:        nil,
			expectedHooks: nil,
		},
		"WebhookInterceptingWebhooksIdentifiersEmpty_ReturnsNil": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "m-webhook-a"},
			},
			config:        &webhookInterceptingWebhooksPrecomputedConfig{},
			expectedHooks: nil,
		},
		"WebhookInterceptingWebhooksIdentifiersNonEmpty_UserIsMaintainer_ReturnsNil": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-b", Name: "webhook-c"},
			},
			config: &webhookInterceptingWebhooksPrecomputedConfig{
				idSet:              sets.NewString(`Mutating\0config-a\0webhook-a`, `Validating\0config-b\0webhook-c`),
				maintainerUsersSet: sets.NewString("user-a", "user-b"),
			},
			expectedHooks: nil,
		},
		"WebhookInterceptingWebhooksIdentifiersNonEmpty_GroupIsMaintainer_ReturnsNil": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-b", Name: "webhook-c"},
			},
			config: &webhookInterceptingWebhooksPrecomputedConfig{
				idSet:               sets.NewString(`Mutating\0config-a\0webhook-a`, `Validating\0config-b\0webhook-c`),
				maintainerGroupsSet: sets.NewString("group-b", "group-f", "group-n"),
			},
			expectedHooks: nil,
		},
		"WebhookInterceptingWebhooksIdentifiersNonEmpty_MatchingRegisteredHooks_ReturnsExpected": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},  // (1)
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-b", Name: "webhook-c"},  // (2)
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-b", Name: "webhook-b"},  // (3)
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-x", Name: "webhook-d"}, // (4)
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-y", Name: "webhook-e"},  // (5)
			},
			config: &webhookInterceptingWebhooksPrecomputedConfig{
				idSet: sets.NewString(
					`Mutating\0config-a\0webhook-c`,   // type and config matches (1), but name doesn't.
					`Mutating\0config-a\0webhook-c`,   // type and name matches (2), but config doesn't.
					`Validating\0config-b\0webhook-b`, // config and name matches (3), but type doesn't.
					`Validating\0config-x\0webhook-d`, // matches (4).
					`Mutating\0config-y\0webhook-e`,   // matches (5).
				),
			},
			expectedHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-x", Name: "webhook-d"},
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-y", Name: "webhook-e"},
			},
		},
		"WebhookInterceptingWebhooksIdentifiersNonEmpty_NoMatchingRegisteredHooks_ReturnsNil": {
			registeredHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-b", Name: "webhook-b"},
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-c", Name: "webhook-b"},
			},
			config: &webhookInterceptingWebhooksPrecomputedConfig{
				idSet: sets.NewString(`Validating\0config-a\0webhook-a`, `Mutating\0config-x\0webhook-a`, `Mutating\0config-a\0webhook-x`),
			},
			expectedHooks: nil,
		},
		"NoRegisteredHooks_ReturnsNil": {
			registeredHooks: nil,
			config: &webhookInterceptingWebhooksPrecomputedConfig{
				idSet: sets.NewString(`Mutating\0config-a\0webhook-c`, `Validating\0config-b\0webhook-c`),
			},
			expectedHooks: nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			mockWebhook := &Webhook{
				webhookInterceptingWebhooksConfig: tc.config,
			}

			admissionRequestUserInfo := &mockUserInfo{
				name:   "user-a",
				groups: []string{"group-a", "group-b"},
			}

			observedFilter := mockWebhook.selectWebhookInterceptingWebhooks(&mockAttribute{userInfo: admissionRequestUserInfo})
			if observedFilter != nil {
				observedHooks := observedFilter(tc.registeredHooks)
				if diff := cmp.Diff(tc.expectedHooks, observedHooks); diff != "" {
					t.Errorf("selectWebhookInterceptingWebhooks (...)(registeredHooks): -want hooks, +got hooks:\n%s", diff)
				}
			} else {
				if tc.expectedHooks != nil {
					t.Errorf("selectWebhookInterceptingWebhooks (...): want nil, got non-nil")
				}
			}
		})
	}
}

func TestDispatch(t *testing.T) {
	sampleWebhookAttr := &mockAttribute{
		gvk: schema.GroupVersionKind{
			Group: "admissionregistration.k8s.io",
			Kind:  "ValidatingWebhookConfiguration",
		},
		userInfo: &mockUserInfo{
			name:   "a-user",
			groups: []string{"gr1", "gr2"},
		},
	}

	sampleRegisteredHooks := []webhook.WebhookAccessor{
		&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},
		&mockWebhookAccessor{IsMutating: true, ConfName: "config-b", Name: "webhook-a"},
		&mockWebhookAccessor{IsMutating: false, ConfName: "config-b", Name: "webhook-b"},
	}

	for name, tc := range map[string]struct {
		incomingAttr            *mockAttribute
		webhookIDs              sets.String
		registeredHooks         []webhook.WebhookAccessor
		expectedDispatchedHooks []webhook.WebhookAccessor
	}{
		"InterceptedObjectIsNotWebhook_DispatchCalled": {
			webhookIDs: sets.NewString(`Mutating\0config-a\0webhook-a`, `Validating\0config-b\0webhook-b`),
			incomingAttr: &mockAttribute{
				gvk: schema.GroupVersionKind{
					Group: "apps",
					Kind:  "Deployment",
				},
			},
			registeredHooks:         sampleRegisteredHooks,
			expectedDispatchedHooks: sampleRegisteredHooks,
		},
		"InterceptedObjectIsWebhook_EmptyWebhookInterceptingWebhooksIdentifiers_DispatchNotCalled": {
			webhookIDs:              sets.NewString(),
			incomingAttr:            sampleWebhookAttr,
			registeredHooks:         sampleRegisteredHooks,
			expectedDispatchedHooks: nil,
		},
		"InterceptedObjectIsWebhook_MatchingWebhookInterceptingWebhooksIdentifiers_DispatchCalled": {
			webhookIDs:      sets.NewString(`Mutating\0config-a\0webhook-a`, `Validating\0config-b\0webhook-b`),
			incomingAttr:    sampleWebhookAttr,
			registeredHooks: sampleRegisteredHooks,
			expectedDispatchedHooks: []webhook.WebhookAccessor{
				&mockWebhookAccessor{IsMutating: true, ConfName: "config-a", Name: "webhook-a"},
				&mockWebhookAccessor{IsMutating: false, ConfName: "config-b", Name: "webhook-b"},
			},
		},
		"InterceptedObjectIsWebhook_NonMatchingWebhookInterceptingWebhooksIdentifiers_DispatchNotCalled": {
			webhookIDs:              sets.NewString(`Validating\0config-a\0webhook-a`, `Mutating\0config-x\0webhook-a`),
			incomingAttr:            sampleWebhookAttr,
			registeredHooks:         sampleRegisteredHooks,
			expectedDispatchedHooks: nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			md := &mockDispatcher{}
			mockWebhook := &Webhook{
				dispatcher: md,
				hookSource: &mockSource{
					hooks: tc.registeredHooks,
				},
				Handler: &admission.Handler{},
				webhookInterceptingWebhooksConfig: &webhookInterceptingWebhooksPrecomputedConfig{
					idSet: tc.webhookIDs,
				},
			}

			mockWebhook.Dispatch(context.TODO(), tc.incomingAttr, nil)
			if diff := cmp.Diff(tc.expectedDispatchedHooks, md.dispatchHooks); diff != "" {
				t.Errorf("genericwebhook.Dispatch (...): -want hooks, +got hooks:\n%s", diff)
			}
		})
	}
}
