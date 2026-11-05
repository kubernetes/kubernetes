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
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestRuntimeOptionPolicyAdmission(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "runtime-options", t)
	pods := client.CoreV1().Pods(ns.Name)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	for _, tc := range []struct {
		name              string
		class             bool
		policy            *nodev1.RuntimeClassPodCheckpointOptions
		checkpointAllowed bool
	}{
		{name: "no-class"},
		{name: "no-policy", class: true},
		{name: "checkpoint-only", class: true, policy: &nodev1.RuntimeClassPodCheckpointOptions{AllowedCheckpointOptions: []string{"tcp"}}, checkpointAllowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}
			if tc.class {
				_, err := client.NodeV1().RuntimeClasses().Create(ctx, &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: tc.name}, Handler: "handler", PodCheckpointOptions: tc.policy}, metav1.CreateOptions{})
				require.NoError(t, err)
				spec.RuntimeClassName = new(tc.name)
			}
			source, err := pods.Create(ctx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: tc.name}, Spec: spec}, metav1.CreateOptions{})
			require.NoError(t, err)
			source = markCheckpointSourceRunning(ctx, t, client, source)
			request := newPodCheckpoint(ns.Name, tc.name+"-options", source.Name)
			request.Spec.CheckpointOptions = map[string]string{"tcp": "close"}
			_, err = checkpoints.Create(ctx, request, metav1.CreateOptions{})
			if tc.checkpointAllowed {
				require.NoError(t, err)
			} else {
				require.True(t, apierrors.IsForbidden(err), "expected forbidden checkpoint, got %v", err)
			}
			_, err = checkpoints.Create(ctx, newPodCheckpoint(ns.Name, tc.name+"-defaults", source.Name), metav1.CreateOptions{})
			require.NoError(t, err, "empty checkpoint options must work")
			if tc.class && tc.checkpointAllowed {
				rc, err := client.NodeV1().RuntimeClasses().Get(ctx, tc.name, metav1.GetOptions{})
				require.NoError(t, err)
				rc.PodCheckpointOptions = nil
				_, err = client.NodeV1().RuntimeClasses().Update(ctx, rc, metav1.UpdateOptions{})
				require.NoError(t, err)
				request.Name = tc.name + "-revoked"
				_, err = checkpoints.Create(ctx, request, metav1.CreateOptions{})
				require.True(t, apierrors.IsForbidden(err), "expected revoked checkpoint policy, got %v", err)
			}
		})
	}
}
