/*
Copyright 2023 The Kubernetes Authors.

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

package app

import (
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/cloud-provider/app/config"
	cpconfig "k8s.io/cloud-provider/config"
	"k8s.io/cloud-provider/fake"
)

func TestWebhookEnableDisable(t *testing.T) {
	var (
		cloud                = &fake.Cloud{}
		noOpAdmissionHandler = func(req *admissionv1.AdmissionRequest) (*admissionv1.AdmissionResponse, error) {
			return &admissionv1.AdmissionResponse{}, nil
		}
	)

	cases := []struct {
		desc            string
		webhookConfigs  map[string]WebhookConfig
		completedConfig *config.CompletedConfig
		expected        map[string]WebhookHandler
	}{
		{
			"Webhooks Enabled",
			map[string]WebhookConfig{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
			newConfig(cpconfig.WebhookConfiguration{Webhooks: []string{"webhook-a", "webhook-b"}}),
			map[string]WebhookHandler{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
		},
		{
			"Webhook Not Enabled",
			map[string]WebhookConfig{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
			newConfig(cpconfig.WebhookConfiguration{Webhooks: []string{"webhook-a"}}),
			map[string]WebhookHandler{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
			},
		},
		{
			"Webhook Disabled (1)",
			map[string]WebhookConfig{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
			newConfig(cpconfig.WebhookConfiguration{Webhooks: []string{"webhook-a", "-webhook-b"}}),
			map[string]WebhookHandler{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
			},
		},
		{
			"Webhook Disabled (2)",
			map[string]WebhookConfig{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
			newConfig(cpconfig.WebhookConfiguration{Webhooks: []string{"-webhook-b"}}),
			map[string]WebhookHandler{},
		},
		{
			"Webhooks Enabled Glob",
			map[string]WebhookConfig{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
			newConfig(cpconfig.WebhookConfiguration{Webhooks: []string{"*"}}),
			map[string]WebhookHandler{
				"webhook-a": {Path: "/path/a", AdmissionHandler: noOpAdmissionHandler},
				"webhook-b": {Path: "/path/b", AdmissionHandler: noOpAdmissionHandler},
			},
		},
	}
	for _, tc := range cases {
		t.Logf("Running %q", tc.desc)
		actual := NewWebhookHandlers(tc.webhookConfigs, tc.completedConfig, cloud)
		if !webhookHandlersEqual(actual, tc.expected) {
			t.Fatalf(
				"FAILED: %q\n---\nActual:\n%s\nExpected:\n%s\ntc.webhookConfigs:\n%s\ntc.completedConfig:\n%s\n",
				tc.desc,
				dump.Pretty(actual),
				dump.Pretty(tc.expected),
				dump.Pretty(tc.webhookConfigs),
				dump.Pretty(tc.completedConfig),
			)
		}
	}
}

func newConfig(webhookConfig cpconfig.WebhookConfiguration) *config.CompletedConfig {
	cfg := &config.Config{
		ComponentConfig: cpconfig.CloudControllerManagerConfiguration{
			Webhook: webhookConfig,
		},
	}
	return cfg.Complete()
}

func webhookHandlersEqual(actual, expected map[string]WebhookHandler) bool {
	if len(actual) != len(expected) {
		return false
	}
	for k := range expected {
		if _, ok := actual[k]; !ok {
			return false
		}
	}
	return true
}
