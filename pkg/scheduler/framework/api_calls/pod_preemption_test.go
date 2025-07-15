/*
Copyright 2025 The Kubernetes Authors.

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

package apicalls

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
)

func TestPodPreemptionCall_Execute(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	victim := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "victim-uid",
			Name:      "victim-pod",
			Namespace: "ns",
		},
	}
	preemptor := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "preemptor-uid",
			Name:      "preemptor-pod",
			Namespace: "ns",
		},
	}
	condition := &v1.PodCondition{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  v1.PodReasonPreemptionByScheduler,
		Message: "preempted",
	}

	t.Run("Successful patch and deletion", func(t *testing.T) {
		client := fake.NewClientset()
		patched := false
		deleted := false

		client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			patched = true
			patchAction := action.(clienttesting.PatchAction)
			if patchAction.GetName() != victim.Name {
				t.Errorf("Expected patch to be called for %v, but was: %v", victim.Name, patchAction.GetName())
			}
			return true, nil, nil
		})
		client.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			deleted = true
			deleteAction := action.(clienttesting.DeleteAction)
			if deleteAction.GetName() != victim.Name {
				t.Errorf("Expected delete to be called for %v, but was: %v", victim.Name, deleteAction.GetName())
			}
			return true, nil, nil
		})

		call := NewPodPreemptionCall(victim, preemptor, condition)
		if err := call.Execute(ctx, client); err != nil {
			t.Fatalf("Unexpected error returned by Execute(): %v", err)
		}
		if !patched {
			t.Error("Expected patch API to be called")
		}
		if !deleted {
			t.Error("Expected delete API to be called")
		}
	})

	t.Run("Successful deletion if patch not needed", func(t *testing.T) {
		client := fake.NewClientset()
		patched := false
		deleted := false
		client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			patched = true
			return true, nil, nil
		})
		client.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			deleted = true
			return true, nil, nil
		})

		// Create a victim that already has the target condition.
		victimWithCondition := victim.DeepCopy()
		victimWithCondition.Status.Conditions = []v1.PodCondition{*condition}
		noOpCall := NewPodPreemptionCall(victimWithCondition, preemptor, condition)
		if err := noOpCall.Execute(ctx, client); err != nil {
			t.Fatalf("Unexpected error returned by Execute(): %v", err)
		}
		if patched {
			t.Error("Expected patch API not to be called if the call is no-op")
		}
		if !deleted {
			t.Error("Expected delete API to be called")
		}
	})

	t.Run("Successful execution if pod not found", func(t *testing.T) {
		notFoundErr := apierrors.NewNotFound(v1.Resource("pods"), victim.Name)
		client := fake.NewClientset()
		patched := false
		deleted := false
		client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			patched = true
			return true, nil, nil
		})
		client.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			deleted = true
			return true, nil, notFoundErr
		})

		call := NewPodPreemptionCall(victim, preemptor, condition)
		if err := call.Execute(ctx, client); err != nil {
			t.Fatalf("Expected Execute() to handle IsNotFound error and return nil, but got: %v", err)
		}
		if !patched {
			t.Error("Expected patch API to be called")
		}
		if !deleted {
			t.Error("Expected delete API to be called")
		}
	})
}
