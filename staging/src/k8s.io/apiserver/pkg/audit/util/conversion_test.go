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

package util

import (
	"testing"

	"github.com/stretchr/testify/require"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/util/webhook"
)

func TestHookClientConfigForSink(t *testing.T) {
	testURL := "http://localhost"
	path := "/path"
	for _, tc := range []struct {
		desc         string
		sink         *auditregv1alpha1.AuditSink
		clientConfig webhook.ClientConfig
	}{
		{
			desc: "build full",
			sink: &auditregv1alpha1.AuditSink{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: auditregv1alpha1.AuditSinkSpec{
					Webhook: auditregv1alpha1.Webhook{
						ClientConfig: auditregv1alpha1.WebhookClientConfig{
							URL: &testURL,
							Service: &auditregv1alpha1.ServiceReference{
								Name:      "test",
								Path:      &path,
								Namespace: "test",
							},
						},
					},
				},
			},
			clientConfig: webhook.ClientConfig{
				Name: "test",
				URL:  testURL,
				Service: &webhook.ClientConfigService{
					Name:      "test",
					Namespace: "test",
					Path:      path,
				},
			},
		},
		{
			desc: "build empty client config",
			sink: &auditregv1alpha1.AuditSink{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: auditregv1alpha1.AuditSinkSpec{
					Webhook: auditregv1alpha1.Webhook{
						ClientConfig: auditregv1alpha1.WebhookClientConfig{},
					},
				},
			},
			clientConfig: webhook.ClientConfig{
				Name: "test",
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			ret := HookClientConfigForSink(tc.sink)
			require.Equal(t, tc.clientConfig, ret)
		})
	}
}
