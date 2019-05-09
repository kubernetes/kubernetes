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

// TODO(aaron-prindle) add back defaults_test.go
// import (
// 	"reflect"
// 	"testing"

// 	flowregistrationv1alpha1 "k8s.io/api/flowregistration/v1alpha1"
// 	apiequality "k8s.io/apimachinery/pkg/api/equality"
// 	"k8s.io/apimachinery/pkg/runtime"
// 	"k8s.io/kubernetes/pkg/api/legacyscheme"
// 	_ "k8s.io/kubernetes/pkg/apis/flowregistration/install"
// 	. "k8s.io/kubernetes/pkg/apis/flowregistration/v1alpha1"
// 	utilpointer "k8s.io/utils/pointer"
// )

// func TestSetDefaultFlowSchema(t *testing.T) {
// 	defaultURL := "http://test"
// 	tests := []struct {
// 		original *flowregistrationv1alpha1.FlowSchema
// 		expected *flowregistrationv1alpha1.FlowSchema
// 	}{
// 		{ // Missing Throttle
// 			original: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 			expected: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						Throttle: DefaultThrottle(),
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 		},
// 		{ // Missing QPS
// 			original: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						Throttle: &flowregistrationv1alpha1.WebhookThrottleConfig{
// 							Burst: utilpointer.Int64Ptr(1),
// 						},
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 			expected: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						Throttle: &flowregistrationv1alpha1.WebhookThrottleConfig{
// 							QPS:   DefaultThrottle().QPS,
// 							Burst: utilpointer.Int64Ptr(1),
// 						},
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 		},
// 		{ // Missing Burst
// 			original: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						Throttle: &flowregistrationv1alpha1.WebhookThrottleConfig{
// 							QPS: utilpointer.Int64Ptr(1),
// 						},
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 			expected: &flowregistrationv1alpha1.FlowSchema{
// 				Spec: flowregistrationv1alpha1.FlowSchemaSpec{
// 					Policy: flowregistrationv1alpha1.Policy{
// 						Level: flowregistrationv1alpha1.LevelMetadata,
// 					},
// 					Webhook: flowregistrationv1alpha1.Webhook{
// 						Throttle: &flowregistrationv1alpha1.WebhookThrottleConfig{
// 							QPS:   utilpointer.Int64Ptr(1),
// 							Burst: DefaultThrottle().Burst,
// 						},
// 						ClientConfig: flowregistrationv1alpha1.WebhookClientConfig{
// 							URL: &defaultURL,
// 						},
// 					},
// 				},
// 			},
// 		},
// 	}

// 	for i, test := range tests {
// 		original := test.original
// 		expected := test.expected
// 		obj2 := roundTrip(t, runtime.Object(original))
// 		got, ok := obj2.(*flowregistrationv1alpha1.FlowSchema)
// 		if !ok {
// 			t.Fatalf("(%d) unexpected object: %v", i, obj2)
// 		}
// 		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
// 			t.Errorf("(%d) got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", i, got.Spec, expected.Spec)
// 		}
// 	}
// }

// func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
// 	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
// 	if err != nil {
// 		t.Errorf("%v\n %#v", err, obj)
// 		return nil
// 	}
// 	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
// 	if err != nil {
// 		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
// 		return nil
// 	}
// 	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
// 	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
// 	if err != nil {
// 		t.Errorf("%v\nSource: %#v", err, obj2)
// 		return nil
// 	}
// 	return obj3
// }
