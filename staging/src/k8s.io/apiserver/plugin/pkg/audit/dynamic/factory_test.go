/*
Copyright 2018 The Kubernetes Authors.

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

package dynamic

import (
	"testing"

	"github.com/stretchr/testify/require"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	utilpointer "k8s.io/utils/pointer"
)

func TestToDelegate(t *testing.T) {
	config, _ := defaultTestConfig()
	defaultPolicy := auditregv1alpha1.Policy{
		Level: auditregv1alpha1.LevelMetadata,
	}
	u := "http://localhost:4444"
	for _, tc := range []struct {
		name            string
		auditConfig     *auditregv1alpha1.AuditSink
		throttleConfig  *auditregv1alpha1.WebhookThrottleConfig
		expectedBackend string
	}{
		{
			name: "build full",
			auditConfig: &auditregv1alpha1.AuditSink{
				Spec: auditregv1alpha1.AuditSinkSpec{
					Policy: defaultPolicy,
					Webhook: auditregv1alpha1.Webhook{
						Throttle: &auditregv1alpha1.WebhookThrottleConfig{
							QPS:   utilpointer.Int64Ptr(10),
							Burst: utilpointer.Int64Ptr(5),
						},
						ClientConfig: auditregv1alpha1.WebhookClientConfig{
							URL: &u,
						},
					},
				},
			},
			expectedBackend: "buffered<enforced<dynamic_webhook>>",
		},
		{
			name: "build no throttle",
			auditConfig: &auditregv1alpha1.AuditSink{
				Spec: auditregv1alpha1.AuditSinkSpec{
					Policy: defaultPolicy,
					Webhook: auditregv1alpha1.Webhook{
						ClientConfig: auditregv1alpha1.WebhookClientConfig{
							URL: &u,
						},
					},
				},
			},
			expectedBackend: "buffered<enforced<dynamic_webhook>>",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := NewBackend(config)
			require.NoError(t, err)
			c := factory{
				config:               b.(*backend).config,
				webhookClientManager: b.(*backend).webhookClientManager,
				sink:                 tc.auditConfig,
			}
			d, err := c.BuildDelegate()
			require.NoError(t, err)
			require.Equal(t, tc.expectedBackend, d.String())
		})
	}
}

func TestBuildWebhookBackend(t *testing.T) {
	defaultPolicy := auditregv1alpha1.Policy{
		Level: auditregv1alpha1.LevelMetadata,
	}
	config, _ := defaultTestConfig()
	b, err := NewBackend(config)
	require.NoError(t, err)
	d := b.(*backend)
	u := "http://localhost:4444"
	for _, tc := range []struct {
		name            string
		auditConfig     *auditregv1alpha1.AuditSink
		shouldErr       bool
		expectedBackend string
	}{
		{
			name: "build full",
			auditConfig: &auditregv1alpha1.AuditSink{
				Spec: auditregv1alpha1.AuditSinkSpec{
					Policy: defaultPolicy,
					Webhook: auditregv1alpha1.Webhook{
						ClientConfig: auditregv1alpha1.WebhookClientConfig{
							URL: &u,
						},
					},
				},
			},
			expectedBackend: "dynamic_webhook",
			shouldErr:       false,
		},
		{
			name: "fail missing url",
			auditConfig: &auditregv1alpha1.AuditSink{
				Spec: auditregv1alpha1.AuditSinkSpec{
					Policy: defaultPolicy,
					Webhook: auditregv1alpha1.Webhook{
						ClientConfig: auditregv1alpha1.WebhookClientConfig{},
					},
				},
			},
			shouldErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &factory{
				config:               config,
				webhookClientManager: d.webhookClientManager,
				sink:                 tc.auditConfig,
			}
			ab, err := c.buildWebhookBackend()
			if tc.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedBackend, ab.String())
		})
	}
}
