/*
Copyright 2018 The Kubernetes Authors.

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

package runtimeclass_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	nodev1 "k8s.io/api/node/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"

	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/utils/ptr"
)

func TestLookupRuntimeHandler(t *testing.T) {
	tests := []struct {
		rcn         *string
		expected    string
		expectError bool
	}{
		{rcn: ptr.To(""), expected: ""},
		{rcn: ptr.To(rctest.EmptyRuntimeClass), expected: ""},
		{rcn: ptr.To(rctest.SandboxRuntimeClass), expected: "kata-containers"},
		{rcn: ptr.To("phantom"), expectError: true},
	}

	manager := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(manager)()

	for _, test := range tests {
		tname := "nil"
		if test.rcn != nil {
			tname = *test.rcn
		}
		t.Run(fmt.Sprintf("%q->%q(err:%v)", tname, test.expected, test.expectError), func(t *testing.T) {
			handler, err := manager.LookupRuntimeHandler(test.rcn)
			if test.expectError {
				assert.Error(t, err, "handler=%q", handler)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, handler)
			}
		})
	}
}

func TestLookupRuntimeHandlerForRestore(t *testing.T) {
	for _, tc := range []struct {
		name         string
		className    *string
		options      map[string]string
		policy       *nodev1.RuntimeClassPodCheckpoint
		missingClass bool
		wantErr      string
	}{
		{name: "no options uses default handler"},
		{name: "options without class", options: map[string]string{"tcp": "close"}, wantErr: "spec.runtimeClassName"},
		{name: "options with empty class", className: ptr.To(""), options: map[string]string{"tcp": "close"}, wantErr: "spec.runtimeClassName"},
		{name: "missing class", className: ptr.To("runtime"), options: map[string]string{"tcp": "close"}, missingClass: true, wantErr: "failed to read RuntimeClass"},
		{name: "missing policy", className: ptr.To("runtime"), options: map[string]string{"tcp": "close"}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "checkpoint list cannot authorize restore", className: ptr.To("runtime"), options: map[string]string{"tcp": "close"}, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "restore list authorizes key", className: ptr.To("runtime"), options: map[string]string{"tcp": "close"}, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}},
		{name: "other restore key denied", className: ptr.To("runtime"), options: map[string]string{"device-map": "sensitive-value"}, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}, wantErr: `runtime option "device-map" is not allowed`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var objects []runtime.Object
			if tc.className != nil && *tc.className != "" && !tc.missingClass {
				objects = append(objects, &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: *tc.className}, Handler: "handler", PodCheckpoint: tc.policy})
			}
			client := fake.NewClientset(objects...)
			manager := runtimeclass.NewManager(client)
			handler, err := manager.LookupRuntimeHandlerForRestore(context.Background(), tc.className, tc.options)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				require.NotContains(t, err.Error(), "sensitive-value")
				require.Empty(t, handler)
				return
			}
			require.NoError(t, err)
			if tc.className == nil {
				require.Empty(t, handler)
				require.Empty(t, client.Actions())
			} else {
				require.Equal(t, "handler", handler)
			}
		})
	}
}

func TestRestoreRuntimeHandlerAndPolicyUseSameClass(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(fmt.Sprintf("live-policy-allows=%t", allow), func(t *testing.T) {
			client := rctest.NewPopulatedClient().(*fake.Clientset)
			manager := runtimeclass.NewManager(client)
			defer rctest.StartManagerSync(manager)()
			handler, err := manager.LookupRuntimeHandler(ptr.To(rctest.SandboxRuntimeClass))
			require.NoError(t, err)
			require.Equal(t, rctest.SandboxRuntimeHandler, handler)
			// A live GET can observe a recreated class before its informer
			// observes the new handler and policy.
			live := &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: rctest.SandboxRuntimeClass}, Handler: "new-handler"}
			if allow {
				live.PodCheckpoint = &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}
			}
			client.PrependReactor("get", "runtimeclasses", func(clienttesting.Action) (bool, runtime.Object, error) {
				return true, live.DeepCopy(), nil
			})
			handler, err = manager.LookupRuntimeHandlerForRestore(context.Background(), ptr.To(rctest.SandboxRuntimeClass), map[string]string{"tcp": "close"})
			if allow {
				require.NoError(t, err)
				require.Equal(t, live.Handler, handler)
			} else {
				require.ErrorContains(t, err, `runtime option "tcp" is not allowed`)
				require.Empty(t, handler)
			}
		})
	}
}
