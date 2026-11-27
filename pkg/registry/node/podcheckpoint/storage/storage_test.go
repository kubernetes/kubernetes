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

package storage

import (
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/fake"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
)

func TestAssignSourceNode(t *testing.T) {
	running := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "source", UID: "source-uid"},
		Spec:       v1.PodSpec{NodeName: "node-1"},
		Status:     v1.PodStatus{Phase: v1.PodRunning},
	}
	pending := running.DeepCopy()
	pending.Status.Phase = v1.PodPending
	for _, tc := range []struct {
		name      string
		pod       *v1.Pod
		uid       *types.UID
		forbidden bool
	}{
		{name: "running source is assigned", pod: running},
		{name: "pinned UID matches", pod: running, uid: new(types.UID("source-uid"))},
		{name: "missing source", forbidden: true},
		{name: "source not running", pod: pending, forbidden: true},
		{name: "pinned UID differs", pod: running, uid: new(types.UID("other-uid")), forbidden: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewClientset()
			if tc.pod != nil {
				client = fake.NewClientset(tc.pod)
			}
			ctx := genericapirequest.WithNamespace(t.Context(), "ns")
			pc := &nodeapi.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint"},
				Spec:       nodeapi.PodCheckpointSpec{SourcePod: nodeapi.PodReference{Name: "source", UID: tc.uid}},
				// A client cannot route a checkpoint to a node of its choice.
				Status: &nodeapi.PodCheckpointStatus{NodeName: new("other-node")},
			}
			err := assignSourceNode(ctx, client.CoreV1(), pc)
			if tc.forbidden {
				require.True(t, apierrors.IsForbidden(err), "got %v", err)
				require.Nil(t, pc.Status)
				return
			}
			require.NoError(t, err)
			require.Equal(t, &nodeapi.PodCheckpointStatus{NodeName: new("node-1"), SourcePodUID: new(types.UID("source-uid"))}, pc.Status)
		})
	}
}
