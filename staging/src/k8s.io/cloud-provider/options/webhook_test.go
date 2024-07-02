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

package options

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestWebhookOptions_Validate(t *testing.T) {
	type args struct {
		validatingWebhooks        []string
		mutatingWebhooks          []string
		disabledByDefaultWebhooks []string
	}
	tests := []struct {
		name  string
		input *WebhookOptions
		args  args
		want  []error
	}{
		{
			name: "Unknown Validating Webhook",
			input: &WebhookOptions{
				Webhooks:                        []string{"test.ccm.io"},
				ValidatingWebhookConfigFilePath: "test.txt",
			},
			args: args{
				validatingWebhooks:        []string{"test1.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{fmt.Errorf("%q is not in the list of known webhooks", "test.ccm.io")},
		},
		{
			name: "Validating Webhook Configuration Missing",
			input: &WebhookOptions{
				Webhooks: []string{"test.ccm.io"},
			},
			args: args{
				validatingWebhooks:        []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhooks [test.ccm.io] are enabled but the validating webhook configuration path is empty")},
		},
		{
			name: "Missing Validating Webhook Configuration",
			input: &WebhookOptions{
				Webhooks:                        []string{"test.ccm.io"},
				ValidatingWebhookConfigFilePath: "test.txt",
				ValidatingWebhookConfiguration: &admissionregistrationv1.ValidatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test",
					},
					Webhooks: []admissionregistrationv1.ValidatingWebhook{},
				},
			},
			args: args{
				validatingWebhooks:        []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhook test.ccm.io is enabled but is not present in the webhook configuration")},
		},
		{
			name: "Extra Validating Webhook Configuration",
			input: &WebhookOptions{
				Webhooks:                        []string{"test.ccm.io"},
				ValidatingWebhookConfigFilePath: "test.txt",
				ValidatingWebhookConfiguration: &admissionregistrationv1.ValidatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test",
					},
					Webhooks: []admissionregistrationv1.ValidatingWebhook{
						{
							Name: "test.ccm.io",
						},
						{
							Name: "test1.ccm.io",
						},
					},
				},
			},
			args: args{
				validatingWebhooks:        []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhook configuration is present for webhooks map[test1.ccm.io:{}] but the webhooks are not present/disabled")},
		},
		{
			name: "Unknown Mutating Webhook",
			input: &WebhookOptions{
				Webhooks:                      []string{"test.ccm.io"},
				MutatingWebhookConfigFilePath: "test.txt",
			},
			args: args{
				mutatingWebhooks:          []string{"test1.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{fmt.Errorf("%q is not in the list of known webhooks", "test.ccm.io")},
		},
		{
			name: "Mutating Webhook Configuration Missing",
			input: &WebhookOptions{
				Webhooks: []string{"test.ccm.io"},
			},
			args: args{
				mutatingWebhooks:          []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhooks [test.ccm.io] are enabled but the mutating webhook configuration path is empty")},
		},
		{
			name: "Missing Mutating Webhook Configuration",
			input: &WebhookOptions{
				Webhooks:                      []string{"test.ccm.io"},
				MutatingWebhookConfigFilePath: "test.txt",
				MutatingWebhookConfiguration: &admissionregistrationv1.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test",
					},
					Webhooks: []admissionregistrationv1.MutatingWebhook{},
				},
			},
			args: args{
				mutatingWebhooks:          []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhook test.ccm.io is enabled but is not present in the webhook configuration")},
		},
		{
			name: "Extra Mutating Webhook Configuration",
			input: &WebhookOptions{
				Webhooks:                      []string{"test.ccm.io"},
				MutatingWebhookConfigFilePath: "test.txt",
				MutatingWebhookConfiguration: &admissionregistrationv1.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test",
					},
					Webhooks: []admissionregistrationv1.MutatingWebhook{
						{
							Name: "test.ccm.io",
						},
						{
							Name: "test1.ccm.io",
						},
					},
				},
			},
			args: args{
				mutatingWebhooks:          []string{"test.ccm.io"},
				disabledByDefaultWebhooks: []string{},
			},
			want: []error{errors.New("webhook configuration is present for webhooks map[test1.ccm.io:{}] but the webhooks are not present/disabled")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.input.Validate(tt.args.validatingWebhooks, tt.args.mutatingWebhooks, tt.args.disabledByDefaultWebhooks); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("WebhookOptions.Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}
