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
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCheckpointTerminalStorageValidation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) {
		opts.Authorization.Modes = []string{"AlwaysAllow"}
		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins, "NodeRestriction")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-terminal-storage", t)
	source := createCheckpointSource(ctx, t, client, ns.Name, "source")
	for _, writer := range []string{"system:node:node-1", "checkpoint-status-writer"} {
		cfg := rest.CopyConfig(config)
		cfg.Impersonate = rest.ImpersonationConfig{UserName: writer}
		if writer == "system:node:node-1" {
			cfg.Impersonate.Groups = []string{"system:nodes"}
		}
		checkpoints := clientset.NewForConfigOrDie(cfg).NodeV1alpha1().PodCheckpoints(ns.Name)
		for i, reason := range []string{nodev1alpha1.PodCheckpointReasonCompleted, nodev1alpha1.PodCheckpointReasonFailed, nodev1alpha1.PodCheckpointReasonSourcePodReplaced} {
			t.Run(writer+"/"+reason, func(t *testing.T) {
				pc, err := checkpoints.Create(ctx, newPodCheckpoint(ns.Name, fmt.Sprintf("checkpoint-%d-%d", len(writer), i), source.Name), metav1.CreateOptions{})
				require.NoError(t, err)
				status := metav1.ConditionFalse
				if i == 0 {
					status = metav1.ConditionTrue
				}
				pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: status, Reason: reason, LastTransitionTime: metav1.Now()}}
				pc.Status.CheckpointedContainers = []nodev1alpha1.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}}
				pc, err = checkpoints.UpdateStatus(ctx, pc, metav1.UpdateOptions{})
				require.NoError(t, err)
				_, err = checkpoints.UpdateStatus(ctx, pc.DeepCopy(), metav1.UpdateOptions{})
				require.NoError(t, err)
				reordered := pc.DeepCopy()
				reordered.Status.CheckpointedContainers[0], reordered.Status.CheckpointedContainers[1] = reordered.Status.CheckpointedContainers[1], reordered.Status.CheckpointedContainers[0]
				pc, err = checkpoints.UpdateStatus(ctx, reordered, metav1.UpdateOptions{})
				require.NoError(t, err)
				updated := pc.DeepCopy()
				updated.Status.Conditions = nil
				_, err = checkpoints.UpdateStatus(ctx, updated, metav1.UpdateOptions{})
				require.True(t, apierrors.IsInvalid(err), "update: %v", err)
				_, err = checkpoints.Patch(ctx, pc.Name, types.MergePatchType, []byte(`{"status":{"conditions":null}}`), metav1.PatchOptions{}, "status")
				require.True(t, apierrors.IsInvalid(err), "merge patch: %v", err)
				updated.APIVersion = nodev1alpha1.SchemeGroupVersion.String()
				updated.Kind = "PodCheckpoint"
				updated.ResourceVersion = ""
				updated.ManagedFields = nil
				updated.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionUnknown, Reason: "Retry", LastTransitionTime: metav1.Now()}}
				data, err := json.Marshal(updated)
				require.NoError(t, err)
				_, err = checkpoints.Patch(ctx, pc.Name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "terminal-validation", Force: new(true)}, "status")
				require.True(t, apierrors.IsInvalid(err), "forced apply: %v", err)
				stored, err := checkpoints.Get(ctx, pc.Name, metav1.GetOptions{})
				require.NoError(t, err)
				require.Equal(t, pc.Status, stored.Status)
			})
		}
	}
}
