/*
Copyright 2016 The Kubernetes Authors.

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

package replicaset

import (
	"k8s.io/utils/ptr"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	fakeImageName  = "fake-name"
	fakeImage      = "fakeimage"
	replicasetName = "test-replicaset"
	namespace      = "test-namespace"
)

func TestReplicaSetStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("ReplicaSet must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ReplicaSet should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: podtest.MakePodSpec(),
		},
	}
	rs := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: validSelector},
			Template: validPodTemplate.Template,
		},
		Status: apps.ReplicaSetStatus{
			Replicas:           1,
			ObservedGeneration: int64(10),
		},
	}

	Strategy.PrepareForCreate(ctx, rs)
	if rs.Status.Replicas != 0 {
		t.Error("ReplicaSet should not allow setting status.replicas on create")
	}
	if rs.Status.ObservedGeneration != int64(0) {
		t.Error("ReplicaSet should not allow setting status.observedGeneration on create")
	}
	errs := Strategy.Validate(ctx, rs)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	invalidRc := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
	}
	Strategy.PrepareForUpdate(ctx, invalidRc, rs)
	errs = Strategy.ValidateUpdate(ctx, invalidRc, rs)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidRc.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestReplicaSetStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("ReplicaSet must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("ReplicaSet should not allow create on update")
	}
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	oldRS := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "10"},
		Spec: apps.ReplicaSetSpec{
			Replicas: 3,
			Selector: &metav1.LabelSelector{MatchLabels: validSelector},
			Template: validPodTemplate.Template,
		},
		Status: apps.ReplicaSetStatus{
			Replicas:           1,
			ObservedGeneration: int64(10),
		},
	}
	newRS := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "9"},
		Spec: apps.ReplicaSetSpec{
			Replicas: 1,
			Selector: &metav1.LabelSelector{MatchLabels: validSelector},
			Template: validPodTemplate.Template,
		},
		Status: apps.ReplicaSetStatus{
			Replicas:           3,
			ObservedGeneration: int64(11),
		},
	}
	StatusStrategy.PrepareForUpdate(ctx, newRS, oldRS)
	if newRS.Status.Replicas != 3 {
		t.Errorf("ReplicaSet status updates should allow change of replicas: %v", newRS.Status.Replicas)
	}
	if newRS.Spec.Replicas != 3 {
		t.Errorf("PrepareForUpdate should have preferred spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newRS, oldRS)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
}

func TestReplicaSetStatusStrategyWithDeploymentPodReplacementPolicy(t *testing.T) {
	tests := []struct {
		name                                 string
		enableDeploymentPodReplacementPolicy bool
		terminatingReplicas                  *int32
		terminatingReplicasUpdate            *int32
		expectedTerminatingReplicas          *int32
	}{
		{
			name:                                 "should not allow updates when feature gate is disabled",
			enableDeploymentPodReplacementPolicy: false,
			terminatingReplicas:                  nil,
			terminatingReplicasUpdate:            ptr.To[int32](2),
			expectedTerminatingReplicas:          nil,
		},
		{
			name:                                 "should allow update when the field is in use when feature gate is disabled",
			enableDeploymentPodReplacementPolicy: false,
			terminatingReplicas:                  ptr.To[int32](2),
			terminatingReplicasUpdate:            ptr.To[int32](5),
			expectedTerminatingReplicas:          ptr.To[int32](5),
		},
		{
			name:                                 "should allow updates when feature gate is enabled",
			enableDeploymentPodReplacementPolicy: true,
			terminatingReplicas:                  nil,
			terminatingReplicasUpdate:            ptr.To[int32](2),
			expectedTerminatingReplicas:          ptr.To[int32](2),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeploymentPodReplacementPolicy, tc.enableDeploymentPodReplacementPolicy)

			ctx := genericapirequest.NewDefaultContext()
			validSelector := map[string]string{"a": "b"}
			oldRS := newReplicaSetWithSelectorLabels(validSelector)
			oldRS.Spec.Replicas = 3
			oldRS.Status.Replicas = 3
			oldRS.Status.TerminatingReplicas = tc.terminatingReplicas

			newRS := newReplicaSetWithSelectorLabels(validSelector)
			newRS.Spec.Replicas = 3
			newRS.Status.Replicas = 2
			newRS.Status.TerminatingReplicas = tc.terminatingReplicasUpdate

			StatusStrategy.PrepareForUpdate(ctx, newRS, oldRS)
			if newRS.Status.Replicas != 2 {
				t.Errorf("ReplicaSet status updates should allow change of replicas: %v", newRS.Status.Replicas)
			}
			if !ptr.Equal(newRS.Status.TerminatingReplicas, tc.expectedTerminatingReplicas) {
				t.Errorf("ReplicaSet status updates failed, expected terminating pods: %v, got: %v", ptr.Deref(tc.expectedTerminatingReplicas, -1), ptr.Deref(newRS.Status.TerminatingReplicas, -1))
			}

			errs := StatusStrategy.ValidateUpdate(ctx, newRS, oldRS)

			if len(errs) != 0 {
				t.Errorf("Unexpected error %v", errs)
			}
		})
	}
}

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
				APIVersion: "v1beta2",
				Resource:   "replicasets",
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
				Resource:   "replicasets",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "d"},
			nil,
		},
	}

	for _, test := range tests {
		oldReplicaSet := newReplicaSetWithSelectorLabels(test.oldSelectorLabels)
		newReplicaSet := newReplicaSetWithSelectorLabels(test.newSelectorLabels)
		context := genericapirequest.NewContext()
		context = genericapirequest.WithRequestInfo(context, &test.requestInfo)
		errorList := rsStrategy{}.ValidateUpdate(context, newReplicaSet, oldReplicaSet)
		if !reflect.DeepEqual(test.expectedErrorList, errorList) {
			t.Errorf("Unexpected error list, expected: %v, actual: %v", test.expectedErrorList, errorList)
		}
	}
}

func newReplicaSetWithSelectorLabels(selectorLabels map[string]string) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:            replicasetName,
			Namespace:       namespace,
			ResourceVersion: "1",
		},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels:      selectorLabels,
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: selectorLabels,
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
}
