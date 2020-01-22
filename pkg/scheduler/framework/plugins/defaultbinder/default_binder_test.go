/*
Copyright 2020 The Kubernetes Authors.

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

package defaultbinder

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

func TestDefaultBinder(t *testing.T) {
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
	}
	testNode := "foohost.kubernetes.mydomain.com"
	tests := []struct {
		name        string
		injectErr   error
		wantBinding *v1.Binding
	}{
		{
			name: "successful",
			wantBinding: &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "foo"},
				Target:     v1.ObjectReference{Kind: "Node", Name: testNode},
			},
		}, {
			name:      "binding error",
			injectErr: errors.New("binding error"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBinding *v1.Binding
			client := fake.NewSimpleClientset(testPod)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "binding" {
					return false, nil, nil
				}
				if tt.injectErr != nil {
					return true, nil, tt.injectErr
				}
				gotBinding = action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
				return true, gotBinding, nil
			})

			fh, err := framework.NewFramework(nil, nil, nil, framework.WithClientSet(client))
			if err != nil {
				t.Fatal(err)
			}
			binder := &DefaultBinder{handle: fh}
			status := binder.Bind(context.Background(), nil, testPod, "foohost.kubernetes.mydomain.com")
			if got := status.AsError(); (tt.injectErr != nil) != (got != nil) {
				t.Errorf("got error %q, want %q", got, tt.injectErr)
			}
			if diff := cmp.Diff(tt.wantBinding, gotBinding); diff != "" {
				t.Errorf("got different binding (-want, +got): %s", diff)
			}
		})
	}
}
