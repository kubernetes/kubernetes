/*
Copyright 2017 The Kubernetes Authors.

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

package v2beta2_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	autoscalingv2 "k8s.io/api/autoscaling/v2beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	. "k8s.io/kubernetes/pkg/apis/autoscaling/v2beta2"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestHorizontalPodAutoscalerAnnotations(t *testing.T) {
	tests := []struct {
		hpa  autoscalingv2.HorizontalPodAutoscaler
		test string
	}{
		{
			hpa: autoscalingv2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						autoscaling.HorizontalPodAutoscalerConditionsAnnotation: "",
						autoscaling.MetricSpecsAnnotation:                       "",
						autoscaling.MetricStatusesAnnotation:                    "",
					},
				},
			},
			test: "test empty value for Annotations",
		},
		{
			hpa: autoscalingv2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						autoscaling.HorizontalPodAutoscalerConditionsAnnotation: "abc",
						autoscaling.MetricSpecsAnnotation:                       "abc",
						autoscaling.MetricStatusesAnnotation:                    "abc",
					},
				},
			},
			test: "test random value for Annotations",
		},
		{
			hpa: autoscalingv2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						autoscaling.HorizontalPodAutoscalerConditionsAnnotation: "[]",
						autoscaling.MetricSpecsAnnotation:                       "[]",
						autoscaling.MetricStatusesAnnotation:                    "[]",
					},
				},
			},
			test: "test empty array value for Annotations",
		},
	}

	for _, test := range tests {
		hpa := &test.hpa
		hpaBeforeMuatate := *hpa.DeepCopy()
		obj := roundTrip(t, runtime.Object(hpa))
		final_obj, ok := obj.(*autoscalingv2.HorizontalPodAutoscaler)
		if !ok {
			t.Fatalf("unexpected object: %v", obj)
		}
		if !reflect.DeepEqual(*hpa, hpaBeforeMuatate) {
			t.Errorf("diff: %v", diff.ObjectDiff(*hpa, hpaBeforeMuatate))
			t.Errorf("expected: %#v\n actual:   %#v", *hpa, hpaBeforeMuatate)
		}

		if len(final_obj.ObjectMeta.Annotations) != 0 {
			t.Fatalf("unexpected annotations: %v", final_obj.ObjectMeta.Annotations)
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
