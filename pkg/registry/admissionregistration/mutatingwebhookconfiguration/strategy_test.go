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

package mutatingwebhookconfiguration

import (
	"context"
	"math"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func Test_mutatingWebhookConfigurationStrategy_PrepareForCreate(t *testing.T) {
	type args struct {
		ctx                context.Context
		obj                runtime.Object
		expectedGeneration int64
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "generation is set to 1 when the configuration is prepared for creation",
			args: args{
				ctx:                context.TODO(),
				obj:                &admissionregistration.MutatingWebhookConfiguration{},
				expectedGeneration: 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mutatingWebhookConfigurationStrategy{
				ObjectTyper:   nil,
				NameGenerator: nil,
			}
			m.PrepareForCreate(tt.args.ctx, tt.args.obj)
			v, ok := tt.args.obj.(*admissionregistration.MutatingWebhookConfiguration)
			if !ok {
				t.Fatalf("error casting runtime.Object to (*admissionregistration.MutatingWebhookConfiguration)")
			}
			if v.Generation != tt.args.expectedGeneration {
				t.Fatalf("generation number not matching, got: %d, want: %d", v.Generation, tt.args.expectedGeneration)
			}
		})
	}
}

func Test_mutatingWebhookConfigurationStrategy_PrepareForUpdate(t *testing.T) {
	type args struct {
		ctx                context.Context
		obj                runtime.Object
		old                runtime.Object
		expectedGeneration int64
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "when the previous webhooks are different than new webhooks, increment generation number",
			args: args{
				ctx: context.TODO(),
				obj: &admissionregistration.MutatingWebhookConfiguration{
					Webhooks: []admissionregistration.Webhook{
						{
							Name: "webhookfoo",
						},
					},
				},
				old:                &admissionregistration.MutatingWebhookConfiguration{},
				expectedGeneration: 1,
			},
		},
		{
			name: "when the new webhooks are different than the old ones, increment the generation number",
			args: args{
				ctx: context.TODO(),
				obj: &admissionregistration.MutatingWebhookConfiguration{},
				old: &admissionregistration.MutatingWebhookConfiguration{
					Webhooks: []admissionregistration.Webhook{
						{
							Name: "webhookfoo",
						},
						{
							Name: "webhookbar",
						},
						{
							Name: "webhookbaz",
						},
					},
				},
				expectedGeneration: 1,
			},
		},
		{
			name: "generation can be increased even if the old generation is higher than zero",
			args: args{
				ctx: context.TODO(),
				obj: &admissionregistration.MutatingWebhookConfiguration{
					Webhooks: []admissionregistration.Webhook{
						{
							Name: "webhookfoo",
						},
					},
				},
				old: &admissionregistration.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 10000003432,
					},
				},
				expectedGeneration: 10000003433,
			},
		},
		{
			name: "generation on the new configuration is not considered when increasing the generation number on update",
			args: args{
				ctx: context.TODO(),
				obj: &admissionregistration.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 100,
					},
					Webhooks: []admissionregistration.Webhook{
						{
							Name: "webhookfoo",
						},
					},
				},
				old: &admissionregistration.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 200,
					},
				},
				expectedGeneration: 201,
			},
		},
		{
			name: "when there's no update regarding webhooks do not increment generation number",
			args: args{
				ctx:                context.TODO(),
				obj:                &admissionregistration.MutatingWebhookConfiguration{},
				old:                &admissionregistration.MutatingWebhookConfiguration{},
				expectedGeneration: 0,
			},
		},
		{
			name: "overflow in generation number increase lead to int64 overflow",
			args: args{
				ctx: context.TODO(),
				obj: &admissionregistration.MutatingWebhookConfiguration{
					Webhooks: []admissionregistration.Webhook{
						{
							Name: "webhookfoo",
						},
					},
				},
				old: &admissionregistration.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{
						Generation: math.MaxInt64,
					},
				},
				expectedGeneration: -9223372036854775808,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mutatingWebhookConfigurationStrategy{
				ObjectTyper:   nil,
				NameGenerator: nil,
			}
			m.PrepareForUpdate(tt.args.ctx, tt.args.obj, tt.args.old)
			v, ok := tt.args.obj.(*admissionregistration.MutatingWebhookConfiguration)
			if !ok {
				t.Fatalf("error casting runtime.Object to (*admissionregistration.MutatingWebhookConfiguration)")
			}
			if v.Generation != tt.args.expectedGeneration {
				t.Fatalf("generation number not matching, got: %d, want: %d", v.Generation, tt.args.expectedGeneration)
			}
		})
	}
}
