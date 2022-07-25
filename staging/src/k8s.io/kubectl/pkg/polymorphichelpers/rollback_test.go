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
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
)

var rollbackTests = map[schema.GroupKind]reflect.Type{
	{Group: "apps", Kind: "DaemonSet"}:   reflect.TypeOf(&DaemonSetRollbacker{}),
	{Group: "apps", Kind: "StatefulSet"}: reflect.TypeOf(&StatefulSetRollbacker{}),
	{Group: "apps", Kind: "Deployment"}:  reflect.TypeOf(&DeploymentRollbacker{}),
}

func TestRollbackerFor(t *testing.T) {
	fakeClientset := &fake.Clientset{}

	for kind, expectedType := range rollbackTests {
		result, err := RollbackerFor(kind, fakeClientset)
		if err != nil {
			t.Fatalf("error getting Rollbacker for a %v: %v", kind.String(), err)
		}

		if reflect.TypeOf(result) != expectedType {
			t.Fatalf("unexpected output type (%v was expected but got %v)", expectedType, reflect.TypeOf(result))
		}
	}
}

func TestGetDeploymentPatch(t *testing.T) {
	patchType, patchBytes, err := getDeploymentPatch(&corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Image: "foo"}}}}, map[string]string{"a": "true"})
	if err != nil {
		t.Error(err)
	}
	if patchType != types.JSONPatchType {
		t.Errorf("expected strategic merge patch, got %v", patchType)
	}
	expectedPatch := `[` +
		`{"op":"replace","path":"/spec/template","value":{"metadata":{"creationTimestamp":null},"spec":{"containers":[{"name":"","image":"foo","resources":{}}]}}},` +
		`{"op":"replace","path":"/metadata/annotations","value":{"a":"true"}}` +
		`]`
	if string(patchBytes) != expectedPatch {
		t.Errorf("expected:\n%s\ngot\n%s", expectedPatch, string(patchBytes))
	}
}

func TestStatefulSetApplyRevision(t *testing.T) {
	tests := []struct {
		name     string
		source   *appsv1.StatefulSet
		expected *appsv1.StatefulSet
	}{
		{
			"test_less",
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v3"},
						},
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{{Name: "i0"}},
							Containers:     []corev1.Container{{Name: "c0"}},
							Volumes:        []corev1.Volume{{Name: "v0"}},
							NodeSelector:   map[string]string{"1dsa": "n0"},
						},
					},
				},
			},
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
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
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{{Name: "i0"}},
						},
					},
				},
			},
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
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
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v3"},
						},
						Spec: corev1.PodSpec{
							Containers:     []corev1.Container{{Name: "c1"}},
							InitContainers: []corev1.Container{{Name: "i0"}},
							Volumes:        []corev1.Volume{{Name: "v0"}},
						},
					},
				},
			},
			&appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"version": "v2"},
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
			patch, err := getStatefulSetPatch(tt.expected)
			if err != nil {
				t.Errorf("getStatefulSetPatch failed : %v", err)
			}
			cr := &appsv1.ControllerRevision{
				Data: runtime.RawExtension{
					Raw: patch,
				},
			}
			tt.source, err = applyRevision(tt.source, cr)
			if err != nil {
				t.Errorf("apply revision failed : %v", err)
			}
			// applyRevision adds TypeMeta field to new the statefulset, so use spec to compare only.
			if !apiequality.Semantic.DeepEqual(tt.source.Spec, tt.expected.Spec) {
				t.Errorf("expected out [%v]  but get [%v]", tt.expected.Spec, tt.source.Spec)
			}
		})
	}
}
