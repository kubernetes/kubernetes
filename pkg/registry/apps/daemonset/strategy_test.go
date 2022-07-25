/*
Copyright 2015 The Kubernetes Authors.

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

package daemonset

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	fakeImageName = "fake-name"
	fakeImage     = "fakeimage"
	daemonsetName = "test-daemonset"
	namespace     = "test-namespace"
)

func TestSelectorImmutability(t *testing.T) {
	tests := []struct {
		requestInfo       genericapirequest.RequestInfo
		oldSelectorLabels map[string]string
		newSelectorLabels map[string]string
		expectedErrorList field.ErrorList
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1",
				Resource:   "daemonsets",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "d"},
			field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: field.NewPath("spec").Child("selector").String(),
					BadValue: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"c": "d"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
					Detail: "field is immutable",
				},
			},
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta2",
				Resource:   "daemonsets",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "d"},
			field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: field.NewPath("spec").Child("selector").String(),
					BadValue: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"c": "d"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
					Detail: "field is immutable",
				},
			},
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "extensions",
				APIVersion: "v1beta1",
				Resource:   "daemonsets",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "d"},
			nil,
		},
	}

	for _, test := range tests {
		oldDaemonSet := newDaemonSetWithSelectorLabels(test.oldSelectorLabels, 1)
		newDaemonSet := newDaemonSetWithSelectorLabels(test.newSelectorLabels, 2)
		context := genericapirequest.NewContext()
		context = genericapirequest.WithRequestInfo(context, &test.requestInfo)
		errorList := daemonSetStrategy{}.ValidateUpdate(context, newDaemonSet, oldDaemonSet)
		if !reflect.DeepEqual(test.expectedErrorList, errorList) {
			t.Errorf("Unexpected error list, expected: %v, actual: %v", test.expectedErrorList, errorList)
		}
	}
}

func newDaemonSetWithSelectorLabels(selectorLabels map[string]string, templateGeneration int64) *apps.DaemonSet {
	return &apps.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:            daemonsetName,
			Namespace:       namespace,
			ResourceVersion: "1",
		},
		Spec: apps.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels:      selectorLabels,
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.OnDeleteDaemonSetStrategyType,
			},
			TemplateGeneration: templateGeneration,
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: selectorLabels,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers:    []api.Container{{Name: fakeImageName, Image: fakeImage, ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
				},
			},
		},
	}
}

func makeDaemonSetWithSurge(unavailable intstr.IntOrString, surge intstr.IntOrString) *apps.DaemonSet {
	return &apps.DaemonSet{
		Spec: apps.DaemonSetSpec{
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.RollingUpdateDaemonSetStrategyType,
				RollingUpdate: &apps.RollingUpdateDaemonSet{
					MaxUnavailable: unavailable,
					MaxSurge:       surge,
				},
			},
		},
	}
}

func TestDropDisabledField(t *testing.T) {
	testCases := []struct {
		name        string
		enableSurge bool
		ds          *apps.DaemonSet
		old         *apps.DaemonSet
		expect      *apps.DaemonSet
	}{
		{
			name:        "not surge, no update",
			enableSurge: false,
			ds:          &apps.DaemonSet{},
			old:         nil,
			expect:      &apps.DaemonSet{},
		},
		{
			name:        "not surge, field not used",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			old:         nil,
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
		},
		{
			name:        "not surge, field not used in old and new",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
		},
		{
			name:        "not surge, field used",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
		},
		{
			name:        "not surge, field used, percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
		},
		{
			name:        "not surge, field used and cleared",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
		},
		{
			name:        "not surge, field used and cleared, percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
		},
		{
			name:        "surge, field not used",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			old:         nil,
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
		},
		{
			name:        "surge, field not used in old and new",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
		},
		{
			name:        "surge, field used",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			old:         nil,
			expect:      makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
		},
		{
			name:        "surge, field used, percent",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromString("1%")),
		},
		{
			name:        "surge, field used in old and new",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			old:         makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
		},
		{
			name:        "surge, allows both fields (validation must catch)",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.FromInt(1)),
		},
		{
			name:        "surge, allows change from unavailable to surge",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
		},
		{
			name:        "surge, allows change from surge to unvailable",
			enableSurge: true,
			ds:          makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			expect:      makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
		},
		{
			name:        "not surge, allows change from unavailable to surge",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
		},
		{
			name:        "not surge, allows change from surge to unvailable",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromInt(1)),
			old:         makeDaemonSetWithSurge(intstr.FromInt(2), intstr.IntOrString{}),
			expect:      makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.IntOrString{}),
		},
		{
			name:        "not surge, allows change from unavailable to surge, percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromString("2%"), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromString("2%"), intstr.IntOrString{}),
		},
		{
			name:        "not surge, allows change from surge to unvailable, percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.FromString("1%")),
			old:         makeDaemonSetWithSurge(intstr.FromString("2%"), intstr.IntOrString{}),
			expect:      makeDaemonSetWithSurge(intstr.IntOrString{}, intstr.IntOrString{}),
		},
		{
			name:        "not surge, resets zero percent, one percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.FromString("1%")),
			old:         makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.FromString("1%")),
		},
		{
			name:        "not surge, resets and clears when zero percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.IntOrString{}),
			old:         makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.FromString("1%")),
			expect:      makeDaemonSetWithSurge(intstr.FromInt(1), intstr.IntOrString{}),
		},
		{
			name:        "not surge, sets zero percent, one percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.FromString("1%")),
			old:         nil,
			expect:      makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.IntOrString{}),
		},
		{
			name:        "not surge, sets and clears zero percent",
			enableSurge: false,
			ds:          makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.IntOrString{}),
			old:         nil,
			expect:      makeDaemonSetWithSurge(intstr.FromString("0%"), intstr.IntOrString{}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DaemonSetUpdateSurge, tc.enableSurge)()
			old := tc.old.DeepCopy()

			dropDaemonSetDisabledFields(tc.ds, tc.old)

			// old obj should never be changed
			if !reflect.DeepEqual(tc.old, old) {
				t.Fatalf("old ds changed: %v", diff.ObjectReflectDiff(tc.old, old))
			}

			if !reflect.DeepEqual(tc.ds, tc.expect) {
				t.Fatalf("unexpected ds spec: %v", diff.ObjectReflectDiff(tc.expect, tc.ds))
			}
		})
	}

}
