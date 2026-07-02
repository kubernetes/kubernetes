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

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestCheckpointSourceStorageValidation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-source-storage", t)
	running := createCheckpointSource(ctx, t, client, ns.Name, "running")
	pending, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pending"}, Spec: v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "pause"}}}}, metav1.CreateOptions{})
	require.NoError(t, err)
	mirror, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "mirror", Annotations: map[string]string{v1.MirrorPodAnnotationKey: ""}},
		Spec:       v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	mirror = markCheckpointSourceRunning(ctx, t, client, mirror)
	deleting := createCheckpointSource(ctx, t, client, ns.Name, "deleting")
	deleting.Finalizers = []string{"example.com/retain"}
	_, err = client.CoreV1().Pods(ns.Name).Update(ctx, deleting, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.NoError(t, client.CoreV1().Pods(ns.Name).Delete(ctx, deleting.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	for _, method := range []string{"post", "apply"} {
		for _, dryRun := range []bool{false, true} {
			for _, tc := range []struct {
				name    string
				source  string
				uid     *types.UID
				allowed bool
			}{
				{name: "running", source: running.Name, allowed: true},
				{name: "matching-uid", source: running.Name, uid: &running.UID, allowed: true},
				{name: "missing", source: "absent"},
				{name: "pending", source: pending.Name},
				{name: "mirror", source: mirror.Name},
				{name: "deleting", source: deleting.Name},
				{name: "wrong-uid", source: running.Name, uid: ptr.To(types.UID("wrong"))},
			} {
				t.Run(fmt.Sprintf("%s/dryrun=%t/%s", method, dryRun, tc.name), func(t *testing.T) {
					name := fmt.Sprintf("%s-%t-%s", method, dryRun, tc.name)
					pc := newPodCheckpoint(ns.Name, name, tc.source)
					pc.Spec.SourcePod.UID = tc.uid
					var dry []string
					if dryRun {
						dry = []string{metav1.DryRunAll}
					}
					if method == "post" {
						_, err = checkpoints.Create(ctx, pc, metav1.CreateOptions{DryRun: dry})
					} else {
						pc.APIVersion = nodev1alpha1.SchemeGroupVersion.String()
						pc.Kind = "PodCheckpoint"
						data, marshalErr := json.Marshal(pc)
						require.NoError(t, marshalErr)
						_, err = checkpoints.Patch(ctx, name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "storage-validation", DryRun: dry})
					}
					if tc.allowed {
						require.NoError(t, err)
					} else {
						require.True(t, apierrors.IsInvalid(err), "expected validation rejection, got %v", err)
					}
					_, getErr := checkpoints.Get(ctx, name, metav1.GetOptions{})
					if !tc.allowed || dryRun {
						require.True(t, apierrors.IsNotFound(getErr), "request must not be stored: %v", getErr)
					} else {
						require.NoError(t, getErr)
					}
				})
			}
		}
	}
}

func TestCheckpointUpdatesAfterSourceDeletion(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-source-deletion", t)
	source := createCheckpointSource(ctx, t, client, ns.Name, "source")
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	pc, err := checkpoints.Create(ctx, newPodCheckpoint(ns.Name, "checkpoint", source.Name), metav1.CreateOptions{})
	require.NoError(t, err)
	require.NoError(t, client.CoreV1().Pods(ns.Name).Delete(ctx, source.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
	pc.Labels = map[string]string{"updated": "true"}
	pc, err = checkpoints.Update(ctx, pc, metav1.UpdateOptions{})
	require.NoError(t, err)
	pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed, LastTransitionTime: metav1.Now()}}
	_, err = checkpoints.UpdateStatus(ctx, pc, metav1.UpdateOptions{})
	require.NoError(t, err)
}
