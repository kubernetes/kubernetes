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

package polymorphichelpers

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
)

var historytests = map[schema.GroupKind]reflect.Type{
	{Group: "apps", Kind: "DaemonSet"}:   reflect.TypeOf(&DaemonSetHistoryViewer{}),
	{Group: "apps", Kind: "StatefulSet"}: reflect.TypeOf(&StatefulSetHistoryViewer{}),
	{Group: "apps", Kind: "Deployment"}:  reflect.TypeOf(&DeploymentHistoryViewer{}),
}

func TestHistoryViewerFor(t *testing.T) {
	fakeClientset := &fake.Clientset{}

	for kind, expectedType := range historytests {
		result, err := HistoryViewerFor(kind, fakeClientset)
		if err != nil {
			t.Fatalf("error getting HistoryViewer for a %v: %v", kind.String(), err)
		}

		if reflect.TypeOf(result) != expectedType {
			t.Fatalf("unexpected output type (%v was expected but got %v)", expectedType, reflect.TypeOf(result))
		}
	}
}

func TestViewHistory(t *testing.T) {

	var (
		trueVar  = true
		replicas = int32(1)

		podStub = corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar"}},
			Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "test", Image: "nginx"}}},
		}

		ssStub = &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "moons",
				Namespace: "default",
				UID:       "1993",
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: appsv1.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: podStub.ObjectMeta.Labels}, Replicas: &replicas, Template: podStub},
		}

		ssStub1 = &appsv1.ControllerRevision{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "moons",
				Namespace:       "default",
				Labels:          map[string]string{"foo": "bar"},
				OwnerReferences: []metav1.OwnerReference{{"apps/v1", "StatefulSet", "moons", "1993", &trueVar, nil}},
			},
			TypeMeta: metav1.TypeMeta{Kind: "StatefulSet", APIVersion: "apps/v1"},
			Revision: 1,
		}
	)

	fakeClientSet := fake.NewSimpleClientset(ssStub)
	_, err := fakeClientSet.AppsV1().ControllerRevisions("default").Create(ssStub1)
	if err != nil {
		t.Fatalf("create controllerRevisions error %v occurred ", err)
	}

	var sts = &StatefulSetHistoryViewer{
		fakeClientSet,
	}

	result, err := sts.ViewHistory("default", "moons", 1)
	if err != nil {
		t.Fatalf("error getting ViewHistory for a StatefulSets moons: %v", err)
	}

	expected := `REVISION
1
`

	if result != expected {
		t.Fatalf("unexpected output  (%v was expected but got %v)", expected, result)
	}

}
