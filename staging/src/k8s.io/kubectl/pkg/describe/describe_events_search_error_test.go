/*
Copyright 2026 The Kubernetes Authors.

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

package describe

import (
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
)

// TestDescribeEventsSearchError ensures that a failed events search surfaces
// the failure in the Events section instead of silently reporting
// "Events: <none>", which misleads users debugging a failing workload.
func TestDescribeEventsSearchError(t *testing.T) {
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "d1", Namespace: "ns1", UID: types.UID("uid-1")},
		Spec: appsv1.DeploymentSpec{
			Replicas: &[]int32{1}[0],
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "d1"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "d1"}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "c", Image: "nginx"}}},
			},
		},
	}

	clientset := fakeclientset.NewSimpleClientset(deployment)
	clientset.PrependReactor("list", "events", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewForbidden(action.GetResource().GroupResource(), "", nil)
	})

	d := &DeploymentDescriber{clientset}
	out, err := d.Describe("ns1", "d1", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Fatalf("Describe returned error: %v", err)
	}

	if !strings.Contains(out, "EventsSearchFailed") {
		t.Errorf("expected the Events section to surface the search failure, got:\n%s", out)
	}
	if strings.Contains(out, "Events:\t<none>") {
		t.Errorf("describe still reports Events: <none> on a failed lookup:\n%s", out)
	}

	// ConfigMapDescriber previously returned a hard error on the same failed
	// search, breaking describe entirely; it must surface the warning too.
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "cm1", Namespace: "ns1"},
		Data:       map[string]string{"k": "v"},
	}
	cmClientset := fakeclientset.NewSimpleClientset(configMap)
	cmClientset.PrependReactor("list", "events", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewForbidden(action.GetResource().GroupResource(), "", nil)
	})

	cmOut, cmErr := (&ConfigMapDescriber{cmClientset}).Describe("ns1", "cm1", DescriberSettings{ShowEvents: true})
	if cmErr != nil {
		t.Fatalf("ConfigMap describe returned error: %v", cmErr)
	}
	if !strings.Contains(cmOut, "EventsSearchFailed") {
		t.Errorf("expected the ConfigMap Events section to surface the search failure, got:\n%s", cmOut)
	}
}
