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

package kubectl

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
)

var rollbacktests = map[schema.GroupKind]reflect.Type{
	{Group: "apps", Kind: "DaemonSet"}:   reflect.TypeOf(&DaemonSetRollbacker{}),
	{Group: "apps", Kind: "StatefulSet"}: reflect.TypeOf(&StatefulSetRollbacker{}),
	{Group: "apps", Kind: "Deployment"}:  reflect.TypeOf(&DeploymentRollbacker{}),
}

func TestRollbackerFor(t *testing.T) {
	fakeClientset := &fake.Clientset{}

	for kind, expectedType := range rollbacktests {
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
