/*
Copyright 2023 The Kubernetes Authors.

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

package bind

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutil "k8s.io/kubernetes/test/integration/util"
)

// TestDefaultBinder tests the binding process in the scheduler.
func TestDefaultBinder(t *testing.T) {
	testCtx := testutil.InitTestSchedulerWithOptions(t, testutil.InitTestAPIServer(t, "", nil), 0)
	testutil.SyncSchedulerInformerFactory(testCtx)

	// Add a node.
	node, err := testutil.CreateNode(testCtx.ClientSet, st.MakeNode().Name("testnode").Obj())
	if err != nil {
		t.Fatal(err)
	}

	tests := map[string]struct {
		anotherUID     bool
		wantStatusCode framework.Code
	}{
		"same UID": {
			wantStatusCode: framework.Success,
		},
		"different UID": {
			anotherUID:     true,
			wantStatusCode: framework.Error,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			pod, err := testutil.CreatePausePodWithResource(testCtx.ClientSet, "fixed-name", testCtx.NS.Name, nil)
			if err != nil {
				t.Fatalf("Failed to create pod: %v", err)
			}
			defer testutil.CleanupPods(testCtx.Ctx, testCtx.ClientSet, t, []*corev1.Pod{pod})

			podCopy := pod.DeepCopy()
			if tc.anotherUID {
				podCopy.UID = "another"
			}

			status := testCtx.Scheduler.Profiles["default-scheduler"].RunBindPlugins(testCtx.Ctx, nil, podCopy, node.Name)
			if code := status.Code(); code != tc.wantStatusCode {
				t.Errorf("Bind returned code %s, want %s", code, tc.wantStatusCode)
			}
		})
	}
}
