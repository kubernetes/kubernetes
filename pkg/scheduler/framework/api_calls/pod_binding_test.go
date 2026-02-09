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

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
)

func TestPodBindingCall_Execute(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	binding := &v1.Binding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod",
			Namespace: "ns",
		},
		Target: v1.ObjectReference{
			Name: "node",
		},
	}

	client := fake.NewClientset()
	bound := false
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		createAction := action.(clienttesting.CreateActionImpl)
		if createAction.Subresource != "binding" {
			return false, nil, nil
		}
		bound = true

		gotBinding := createAction.GetObject().(*v1.Binding)
		if diff := cmp.Diff(binding, gotBinding); diff != "" {
			t.Errorf("Execute() sent incorrect binding object (-want,+got):\n%s", diff)
		}
		return true, nil, nil
	})

	call := NewPodBindingCall(binding)
	if err := call.Execute(ctx, client); err != nil {
		t.Fatalf("Execute() returned unexpected error: %v", err)
	}
	if !bound {
		t.Error("Expected binding API to be called")
	}
}
