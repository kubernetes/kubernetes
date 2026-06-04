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
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
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

func TestValidateToleratingBadLabels(t *testing.T) {
	oldObj := newDaemonSetWithSelectorLabels(map[string]string{"a": "b"}, 1)
	oldObj.Spec.Selector.MatchExpressions = []metav1.LabelSelectorRequirement{{Key: "key", Operator: metav1.LabelSelectorOpNotIn, Values: []string{"bad value"}}}
	newObj := newDaemonSetWithSelectorLabels(map[string]string{"a": "b"}, 1)
	newObj.Spec.Selector.MatchExpressions = []metav1.LabelSelectorRequirement{{Key: "key", Operator: metav1.LabelSelectorOpNotIn, Values: []string{"bad value"}}}

	context := genericapirequest.NewContext()
	errorList := daemonSetStrategy{}.ValidateUpdate(context, newObj, oldObj)
	if len(errorList) > 0 {
		t.Errorf("Unexpected error list with no-op update of bad object: %v", errorList)
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
				Spec: podtest.MakePodSpec(),
			},
		},
	}
}
