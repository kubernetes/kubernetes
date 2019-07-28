/*
Copyright 2019 The Kubernetes Authors.

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

package service

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func addAnnotations(svc *v1.Service) {
	svc.Annotations["foo"] = "bar"
}

func TestPatch(t *testing.T) {
	svcOrigin := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-patch",
			Annotations: map[string]string{},
		},
		Spec: v1.ServiceSpec{
			ClusterIP: "10.0.0.1",
		},
	}
	fakeCs := fake.NewSimpleClientset(svcOrigin)

	// Issue a separate update and verify patch doesn't fail after this.
	svcToUpdate := svcOrigin.DeepCopy()
	addAnnotations(svcToUpdate)
	if _, err := fakeCs.CoreV1().Services(svcOrigin.Namespace).Update(svcToUpdate); err != nil {
		t.Fatalf("Failed to update service: %v", err)
	}

	// Attempt to patch based the original service.
	svcToPatch := svcOrigin.DeepCopy()
	svcToPatch.Finalizers = []string{"foo"}
	svcToPatch.Spec.ClusterIP = "10.0.0.2"
	svcToPatch.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "8.8.8.8"},
			},
		},
	}
	svcPatched, err := patch(fakeCs.CoreV1(), svcOrigin, svcToPatch)
	if err != nil {
		t.Fatalf("Failed to patch service: %v", err)
	}

	// Service returned by patch will contain latest content (e.g from
	// the separate update).
	addAnnotations(svcToPatch)
	if !reflect.DeepEqual(svcPatched, svcToPatch) {
		t.Errorf("PatchStatus() = %+v, want %+v", svcPatched, svcToPatch)
	}
	// Explicitly validate if spec is unchanged from origin.
	if !reflect.DeepEqual(svcPatched.Spec, svcOrigin.Spec) {
		t.Errorf("Got spec = %+v, want %+v", svcPatched.Spec, svcOrigin.Spec)
	}
}

func TestGetPatchBytes(t *testing.T) {
	origin := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "test-patch-bytes",
			Finalizers: []string{"foo"},
		},
	}
	updated := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "test-patch-bytes",
			Finalizers: []string{"foo", "bar"},
		},
	}

	b, err := getPatchBytes(origin, updated)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"metadata":{"$setElementOrder/finalizers":["foo","bar"],"finalizers":["bar"]}}`
	if string(b) != expected {
		t.Errorf("getPatchBytes(%+v, %+v) = %s ; want %s", origin, updated, string(b), expected)
	}
}
