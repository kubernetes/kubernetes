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
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestRestoreRequiresCapturedImageDigests(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-image", t)
	pods := client.CoreV1().Pods(ns.Name)
	source, err := pods.Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "app", Image: "registry.example/app:latest",
				Args: []string{"original-private-argument"},
				Env:  []v1.EnvVar{{Name: "TOKEN", Value: "original-private-environment"}},
			}},
			InitContainers: []v1.Container{{Name: "sidecar", Image: "registry.example/app:latest", RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways)}},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	source = markCheckpointSourceRunning(tCtx, t, client, source)
	const pinnedImage = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	template := checkpointutil.SanitizePodTemplate(source)
	template.Spec.Containers[0].Image = pinnedImage
	template.Spec.InitContainers[0].Image = pinnedImage
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", source.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
	checkpoint.Status.NodeName = new("node")
	checkpoint.Status.CheckpointedPodTemplate = template
	if _, err := checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name       string
		mutate     func(*v1.PodSpec)
		wantFields []string
	}{
		{name: "captured-digests"},
		{name: "original-tag", mutate: func(spec *v1.PodSpec) { spec.Containers[0].Image = source.Spec.Containers[0].Image }, wantFields: []string{"spec.containers[0].image"}},
		{name: "different-digest", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Image = "registry.example/app@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
		}, wantFields: []string{"spec.containers[0].image"}},
		{name: "sidecar-tag", mutate: func(spec *v1.PodSpec) { spec.InitContainers[0].Image = source.Spec.InitContainers[0].Image }, wantFields: []string{"spec.initContainers[0].image"}},
		{name: "environment-and-argument", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Args[0] = "changed-private-argument"
			spec.Containers[0].Env[0].Value = "changed-private-environment"
		}, wantFields: []string{"spec.containers[0].args[0]", "spec.containers[0].env[0].value"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := template.Spec.DeepCopy()
			spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			if tc.mutate != nil {
				tc.mutate(spec)
			}
			_, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: tc.name}, Spec: *spec}, metav1.CreateOptions{})
			if len(tc.wantFields) != 0 {
				require.True(t, apierrors.IsInvalid(err), "expected spec mismatch to be invalid, got %v", err)
				var status apierrors.APIStatus
				require.ErrorAs(t, err, &status)
				require.NotNil(t, status.Status().Details)
				var fields []string
				for _, cause := range status.Status().Details.Causes {
					fields = append(fields, cause.Field)
				}
				require.Equal(t, tc.wantFields, fields)
				for _, value := range []string{"original-private-argument", "original-private-environment", "changed-private-argument", "changed-private-environment"} {
					require.NotContains(t, err.Error(), value)
				}
			} else if err != nil {
				t.Fatalf("restore with captured image digests: %v", err)
			}
		})
	}
}
