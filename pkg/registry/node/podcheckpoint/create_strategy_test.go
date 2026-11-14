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

package podcheckpoint

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	node "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/utils/ptr"
)

func TestValidateSourcePod(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mutate  func(*v1.Pod, *node.PodCheckpoint)
		missing bool
		wantErr string
	}{
		{name: "running"},
		{name: "matching-uid", mutate: func(p *v1.Pod, c *node.PodCheckpoint) { c.Spec.SourcePod.UID = &p.UID }},
		{name: "mirror", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) {
			p.Annotations = map[string]string{v1.MirrorPodAnnotationKey: "hash"}
		}, wantErr: "checkpointing static Pods is not supported"},
		{name: "mirror-matching-uid", mutate: func(p *v1.Pod, c *node.PodCheckpoint) {
			p.Annotations = map[string]string{v1.MirrorPodAnnotationKey: "hash"}
			c.Spec.SourcePod.UID = &p.UID
		}, wantErr: "checkpointing static Pods is not supported"},
		{name: "mirror-empty-annotation", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) {
			p.Annotations = map[string]string{v1.MirrorPodAnnotationKey: ""}
		}, wantErr: "checkpointing static Pods is not supported"},
		{name: "wrong-uid", mutate: func(_ *v1.Pod, c *node.PodCheckpoint) { c.Spec.SourcePod.UID = ptr.To(types.UID("other")) }, wantErr: "UID"},
		{name: "unassigned", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Spec.NodeName = "" }, wantErr: "assigned to a node and Running"},
		{name: "pending", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodPending }, wantErr: "Running"},
		{name: "succeeded", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodSucceeded }, wantErr: "Running"},
		{name: "failed", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodFailed }, wantErr: "Running"},
		{name: "deleting", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.DeletionTimestamp = new(metav1.Now()) }, wantErr: "being deleted"},
		{name: "missing", missing: true, wantErr: "cannot read source"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{NodeName: "node"}, Status: v1.PodStatus{Phase: v1.PodRunning}}
			checkpoint := &node.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}, Spec: node.PodCheckpointSpec{SourcePod: node.PodReference{Name: "source"}}}
			if tc.mutate != nil {
				tc.mutate(pod, checkpoint)
			}
			var objects []runtime.Object
			if !tc.missing {
				objects = append(objects, pod)
			}
			client := fake.NewClientset(objects...)
			errs := NewCreateStrategy(client.CoreV1()).Validate(context.Background(), checkpoint)
			if tc.wantErr == "" {
				require.Empty(t, errs)
			} else {
				require.ErrorContains(t, errs.ToAggregate(), tc.wantErr)
			}
			require.Len(t, client.Actions(), 1)
			require.Equal(t, "ns", client.Actions()[0].GetNamespace())
		})
	}
}

func TestCheckpointCanceledValidation(t *testing.T) {
	client := fake.NewClientset()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pc := &node.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}, Spec: node.PodCheckpointSpec{SourcePod: node.PodReference{Name: "source"}}}
	errs := NewCreateStrategy(client.CoreV1()).Validate(ctx, pc)
	require.ErrorContains(t, errs.ToAggregate(), context.Canceled.Error())
	require.Empty(t, client.Actions())
}
