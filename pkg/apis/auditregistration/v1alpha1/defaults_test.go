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

package v1alpha1_test

import (
	"reflect"
	"testing"

	auditregistrationv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/auditregistration/install"
	. "k8s.io/kubernetes/pkg/apis/auditregistration/v1alpha1"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultAuditSink(t *testing.T) {
	defaultURL := "http://test"
	tests := []struct {
		original *auditregistrationv1alpha1.AuditSink
		expected *auditregistrationv1alpha1.AuditSink
	}{
		{ // Missing Throttle
			original: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
			expected: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						Throttle: DefaultThrottle(),
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
		},
		{ // Missing QPS
			original: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						Throttle: &auditregistrationv1alpha1.WebhookThrottleConfig{
							Burst: utilpointer.Int64Ptr(1),
						},
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
			expected: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						Throttle: &auditregistrationv1alpha1.WebhookThrottleConfig{
							QPS:   DefaultThrottle().QPS,
							Burst: utilpointer.Int64Ptr(1),
						},
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
		},
		{ // Missing Burst
			original: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						Throttle: &auditregistrationv1alpha1.WebhookThrottleConfig{
							QPS: utilpointer.Int64Ptr(1),
						},
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
			expected: &auditregistrationv1alpha1.AuditSink{
				Spec: auditregistrationv1alpha1.AuditSinkSpec{
					Policy: auditregistrationv1alpha1.Policy{
						Level: auditregistrationv1alpha1.LevelMetadata,
					},
					Webhook: auditregistrationv1alpha1.Webhook{
						Throttle: &auditregistrationv1alpha1.WebhookThrottleConfig{
							QPS:   utilpointer.Int64Ptr(1),
							Burst: DefaultThrottle().Burst,
						},
						ClientConfig: auditregistrationv1alpha1.WebhookClientConfig{
							URL: &defaultURL,
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*auditregistrationv1alpha1.AuditSink)
		if !ok {
			t.Fatalf("(%d) unexpected object: %v", i, obj2)
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("(%d) got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", i, got.Spec, expected.Spec)
		}
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}
