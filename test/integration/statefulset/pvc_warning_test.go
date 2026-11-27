/*
Copyright The Kubernetes Authors.

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

package statefulset

import (
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestStatefulSetVolumeClaimTemplateWarningHeaders(t *testing.T) {
	server := apiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	config := rest.CopyConfig(server.ClientConfig)
	config.WarningHandler = rest.NoWarnings{}
	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		storage  string
		warnings []string
	}{
		{
			name: "fractional", storage: "200m",
			warnings: []string{
				`spec.volumeClaimTemplates[0].spec.resources.requests[storage]: fractional byte value "200m" is invalid, must be an integer`,
			},
		},
		{name: "integer", storage: "1Gi"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			obj := &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: tc.name, Namespace: metav1.NamespaceDefault},
				Spec: appsv1.StatefulSetSpec{
					Replicas:    ptr.To[int32](1),
					ServiceName: "headless",
					Selector:    &metav1.LabelSelector{MatchLabels: map[string]string{"app": tc.name}},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": tc.name}},
						Spec: v1.PodSpec{
							Containers: []v1.Container{{Name: "app", Image: "registry.k8s.io/pause:3.10"}},
						},
					},
					VolumeClaimTemplates: []v1.PersistentVolumeClaim{{
						ObjectMeta: metav1.ObjectMeta{Name: "data"},
						Spec: v1.PersistentVolumeClaimSpec{
							AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
							Resources: v1.VolumeResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse(tc.storage)},
							},
						},
					}},
				},
			}
			for _, operation := range []string{"create", "scale update", "metadata update"} {
				t.Run(operation, func(t *testing.T) {
					request := client.AppsV1().RESTClient().Post().Namespace(obj.Namespace).Resource("statefulsets")
					wantStatus := http.StatusCreated
					oldGeneration := obj.Generation
					if operation != "create" {
						request = client.AppsV1().RESTClient().Put().Namespace(obj.Namespace).Resource("statefulsets").Name(obj.Name)
						wantStatus = http.StatusOK
						if operation == "scale update" {
							obj.Spec.Replicas = ptr.To[int32](2)
						} else {
							obj.Annotations = map[string]string{"description": "updated"}
						}
					}
					var status int
					result := request.Body(obj).Do(t.Context()).StatusCode(&status)
					persisted := &appsv1.StatefulSet{}
					if err := result.Into(persisted); err != nil {
						t.Fatalf("API request: %v", err)
					}
					if status != wantStatus {
						t.Errorf("HTTP status = %d, want %d", status, wantStatus)
					}
					var warnings []string
					for _, warning := range result.Warnings() {
						if warning.Code != 299 {
							t.Errorf("Warning code = %d, want 299", warning.Code)
						}
						warnings = append(warnings, warning.Text)
					}
					if diff := cmp.Diff(tc.warnings, warnings, cmpopts.EquateEmpty()); diff != "" {
						t.Errorf("Warning headers (-want +got):\n%s", diff)
					}
					if operation == "metadata update" && persisted.Generation != oldGeneration {
						t.Errorf("metadata update changed generation from %d to %d", oldGeneration, persisted.Generation)
					}
					obj = persisted
					t.Logf("HTTP %d, warnings=%q", status, warnings)
				})
			}
			stored, err := client.AppsV1().StatefulSets(obj.Namespace).Get(t.Context(), obj.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			quantity := stored.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage]
			if quantity.String() != tc.storage {
				t.Errorf("stored storage = %q, want %q", quantity.String(), tc.storage)
			}
		})
	}
}
