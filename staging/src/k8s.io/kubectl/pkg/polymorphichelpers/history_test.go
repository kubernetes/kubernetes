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
	"context"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	_, err := fakeClientSet.AppsV1().ControllerRevisions("default").Create(context.TODO(), ssStub1, metav1.CreateOptions{})
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

func TestApplyDaemonSetHistory(t *testing.T) {
	tests := []struct {
		name     string
		source   *appsv1.DaemonSet
		expected *appsv1.DaemonSet
	}{
		{
			"test_less",
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v3"},
						},
						Spec: corev1.PodSpec{
							InitContainers:   []corev1.Container{{Name: "i0"}},
							Containers:       []corev1.Container{{Name: "c0"}},
							Volumes:          []corev1.Volume{{Name: "v0"}},
							NodeSelector:     map[string]string{"1dsa": "n0"},
							ImagePullSecrets: []corev1.LocalObjectReference{{Name: "ips0"}},
							Tolerations:      []corev1.Toleration{{Key: "t0"}},
							HostAliases:      []corev1.HostAlias{{IP: "h0"}},
							ReadinessGates:   []corev1.PodReadinessGate{{ConditionType: corev1.PodScheduled}},
						},
					},
				},
			},
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "c1"}},
							// keep diversity field, eg: nil or empty slice
							InitContainers: []corev1.Container{},
						},
					},
				},
			},
		},
		{
			"test_more",
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{{Name: "i0"}},
						},
					},
				},
			},
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v3"},
						},
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{{Name: "i1"}},
							Containers:     []corev1.Container{{Name: "c1"}},
							Volumes:        []corev1.Volume{{Name: "v1"}},
						},
					},
				},
			},
		},
		{
			"test_equal",
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v0"},
						},
						Spec: corev1.PodSpec{
							Containers:     []corev1.Container{{Name: "c1"}},
							InitContainers: []corev1.Container{{Name: "i0"}},
							Volumes:        []corev1.Volume{{Name: "v0"}},
						},
					},
				},
			},
			&appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v1"},
						},
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{{Name: "i1"}},
							Containers:     []corev1.Container{{Name: "c1"}},
							Volumes:        []corev1.Volume{{Name: "v1"}},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch, err := getDaemonSetPatch(tt.expected)
			if err != nil {
				t.Errorf("getDaemonSetPatch failed : %v", err)
			}
			cr := &appsv1.ControllerRevision{
				Data: runtime.RawExtension{
					Raw: patch,
				},
			}
			tt.source, err = applyDaemonSetHistory(tt.source, cr)
			if err != nil {
				t.Errorf("applyDaemonSetHistory failed : %v", err)
			}
			if !apiequality.Semantic.DeepEqual(tt.source, tt.expected) {
				t.Errorf("expected out [%v]  but get [%v]", tt.expected, tt.source)
			}
		})
	}
}
