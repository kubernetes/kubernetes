/*
Copyright 2020 The Kubernetes Authors.

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

package v1beta1_test

import (
	"reflect"
	"testing"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	. "k8s.io/kubernetes/pkg/apis/networking/v1beta1"
)

func TestSetIngressPathDefaults(t *testing.T) {
	pathTypeImplementationSpecific := networkingv1beta1.PathTypeImplementationSpecific
	pathTypeExact := networkingv1beta1.PathTypeExact

	testCases := map[string]struct {
		original *networkingv1beta1.HTTPIngressPath
		expected *networkingv1beta1.HTTPIngressPath
	}{
		"empty pathType should default to ImplementationSpecific": {
			original: &networkingv1beta1.HTTPIngressPath{},
			expected: &networkingv1beta1.HTTPIngressPath{PathType: &pathTypeImplementationSpecific},
		},
		"ImplementationSpecific pathType should not change": {
			original: &networkingv1beta1.HTTPIngressPath{PathType: &pathTypeImplementationSpecific},
			expected: &networkingv1beta1.HTTPIngressPath{PathType: &pathTypeImplementationSpecific},
		},
		"Exact pathType should not change": {
			original: &networkingv1beta1.HTTPIngressPath{PathType: &pathTypeExact},
			expected: &networkingv1beta1.HTTPIngressPath{PathType: &pathTypeExact},
		},
	}
	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			ingressOriginal := wrapIngressPath(testCase.original)
			ingressExpected := wrapIngressPath(testCase.expected)
			runtimeObj := roundTrip(t, runtime.Object(ingressOriginal))
			ingressActual, ok := runtimeObj.(*networkingv1beta1.Ingress)
			if !ok {
				t.Fatalf("Unexpected object: %v", ingressActual)
			}
			if !apiequality.Semantic.DeepEqual(ingressActual.Spec, ingressExpected.Spec) {
				t.Errorf("Expected: %+v, got: %+v", ingressExpected.Spec, ingressActual.Spec)
			}
		})
	}
}

func wrapIngressPath(path *networkingv1beta1.HTTPIngressPath) *networkingv1beta1.Ingress {
	return &networkingv1beta1.Ingress{
		Spec: networkingv1beta1.IngressSpec{
			Rules: []networkingv1beta1.IngressRule{{
				IngressRuleValue: networkingv1beta1.IngressRuleValue{
					HTTP: &networkingv1beta1.HTTPIngressRuleValue{
						Paths: []networkingv1beta1.HTTPIngressPath{*path},
					},
				},
			}},
		},
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	t.Helper()
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
