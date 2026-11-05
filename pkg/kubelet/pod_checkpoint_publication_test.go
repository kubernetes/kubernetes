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

package kubelet

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	goruntime "runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/ktesting"
	clocktesting "k8s.io/utils/clock/testing"
)

func TestRecordedCheckpointOutcomePrecedesSourceReplacement(t *testing.T) {
	for _, tc := range []struct {
		name                             string
		missing, committed, duringLookup bool
	}{
		{name: "replacement"},
		{name: "missing", missing: true},
		{name: "lost response", committed: true},
		{name: "completion during live lookup", missing: true, duringLookup: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var ctx context.Context = ktesting.Init(t)
			kl, runtime, source, pc := checkpointOperationFixture(t)
			kl.checkpointOperationContext.Store(&ctx)
			pc.Spec.SourcePod = &nodev1alpha1.PodReference{Name: source.Name}
			pc.Status.NodeName = new(string(kl.nodeName))
			pc.Status.SourcePodUID = new(source.UID)
			pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress}}
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			seedCheckpointCache(t, kl)
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("captured state"), 0600))
			operation := podCheckpointOperation{CheckpointUID: pc.UID, PodUID: source.UID, PodName: source.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}
			if tc.duringLookup {
				client.PrependReactor("list", "pods", func(clienttesting.Action) (bool, apiruntime.Object, error) {
					require.NoError(t, kl.writePodCheckpointOperation(operation))
					return false, nil, nil
				})
			} else {
				require.NoError(t, kl.writePodCheckpointOperation(operation))
			}
			if tc.committed {
				completed := pc.DeepCopy()
				completed.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted}}
				completed.Status.CheckpointLocation = &nodev1alpha1.CheckpointSource{Type: nodev1alpha1.CheckpointSourceTypeNodeLocal, NodeLocal: &nodev1alpha1.NodeLocalCheckpointSource{Path: filepath.Base(output)}}
				require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), completed, pc.Namespace))
			}
			replacement := source.DeepCopy()
			replacement.UID = "replacement-uid"
			if tc.missing {
				kl.podManager.SetPods(nil)
			} else {
				kl.podManager.SetPods([]*v1.Pod{replacement})
			}
			err = kl.syncPodCheckpoint(ctx, pc.Namespace+"/"+pc.Name)
			require.True(t, err == nil || errors.Is(err, errPodCheckpointInFlight) || errors.Is(err, errPodCheckpointCompletionPending) || errors.Is(err, errPodCheckpointNotPending), "%v", err)
			require.Eventually(t, func() bool {
				stored, err := client.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
				return err == nil && podCheckpointTerminal(stored)
			}, 5*time.Second, time.Millisecond)
			stored, err := client.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.True(t, apimeta.IsStatusConditionTrue(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady), "%v", stored.Status.Conditions)
			require.FileExists(t, filepath.Join(output, "saved"))
			require.Empty(t, runtime.CanceledPodOperations)
		})
	}
}

func TestAbandonedCheckpointCleanupRetainsRecoveryRecord(t *testing.T) {
	if goruntime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("requires unprivileged Unix directory permissions")
	}
	for _, recovering := range []bool{false, true} {
		t.Run(fmt.Sprintf("recovering=%t", recovering), func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, source, pc := checkpointOperationFixture(t)
			kl.kubeClient = fake.NewClientset()
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			locked := filepath.Join(output, "locked")
			require.NoError(t, os.Mkdir(locked, 0700))
			require.NoError(t, os.WriteFile(filepath.Join(locked, "saved"), []byte("captured state"), 0600))
			require.NoError(t, os.Chmod(locked, 0500))
			t.Cleanup(func() { _ = os.Chmod(locked, 0700) })
			require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: pc.UID, PodUID: source.UID, PodName: source.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}))
			if recovering {
				require.Error(t, kl.finalizeRecoveredPodCheckpoint(ctx, pc), "failed output removal must remain recoverable")
			} else {
				var captureCtx context.Context
				captureCtx, cancel := context.WithCancel(ctx)
				done := make(chan struct{})
				kl.checkpointPublications.Store(pc.UID, struct{}{})
				go func() {
					defer close(done)
					kl.doPublishPodCheckpointOutcome(captureCtx, &podCheckpointRun{pod: source, namespace: pc.Namespace, name: pc.Name, uid: pc.UID, outputPath: output}, true, filepath.Base(output), "")
				}()
				require.Eventually(t, func() bool {
					select {
					case <-done:
						return true
					default:
					}
					_, blocked := kl.checkpointCleanupBlocked.Load(pc.UID)
					return blocked
				}, 5*time.Second, time.Millisecond)
				cancel()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("publisher ignored cancellation")
				}
			}
			records, err := kl.readPodCheckpointOperations(source.UID)
			require.NoError(t, err)
			require.Len(t, records, 1, "a failed removal must not erase the durable recovery record")
			require.True(t, records[0].Complete, "disk cleanup does not make runtime completion uncertain")
			require.FileExists(t, filepath.Join(locked, "saved"))
			require.NoError(t, os.Chmod(locked, 0700))
			require.NoError(t, kl.finalizeRecoveredPodCheckpoint(ctx, pc))
			require.NoDirExists(t, output)
			records, err = kl.readPodCheckpointOperations(source.UID)
			require.NoError(t, err)
			require.Empty(t, records)
		})
	}
}

func TestCheckpointPublicationBackoffRetainsOutcome(t *testing.T) {
	for _, denied := range []error{
		apierrors.NewForbidden(schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, "checkpoint", errors.New("status permission denied")),
		apierrors.NewInvalid(schema.GroupKind{Group: "node.k8s.io", Kind: "PodCheckpoint"}, "checkpoint", field.ErrorList{field.Invalid(field.NewPath("status"), "invalid", "injected validation error")}),
	} {
		t.Run(string(apierrors.ReasonForError(denied)), func(t *testing.T) {
			ctx, cancel := context.WithCancel(ktesting.Init(t))
			kl, _, source, pc := checkpointOperationFixture(t)
			clock := clocktesting.NewFakeClock(time.Now())
			kl.clock = clock
			client := kl.kubeClient.(*fake.Clientset)
			attempts := make(chan struct{}, 16)
			var allowed atomic.Bool
			client.PrependReactor("update", "podcheckpoints", func(clienttesting.Action) (bool, apiruntime.Object, error) {
				attempts <- struct{}{}
				if !allowed.Load() {
					return true, nil, denied
				}
				return false, nil, nil
			})
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("captured state"), 0600))
			require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: pc.UID, PodUID: source.UID, PodName: source.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}))
			kl.checkpointsInFlight.Store(source.UID, pc.UID)
			kl.checkpointPublications.Store(pc.UID, struct{}{})
			done := make(chan struct{})
			go func() {
				defer close(done)
				kl.doPublishPodCheckpointOutcome(ctx, &podCheckpointRun{pod: source, namespace: pc.Namespace, name: pc.Name, uid: pc.UID, outputPath: output}, true, filepath.Base(output), "")
			}()
			t.Cleanup(func() {
				cancel()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Error("publisher ignored cancellation")
				}
			})
			receive := func() {
				t.Helper()
				select {
				case <-attempts:
				case <-time.After(5 * time.Second):
					t.Fatal("publication attempt did not run")
				}
			}
			receive()
			for _, delay := range []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, 32 * time.Second, time.Minute, time.Minute} {
				require.Eventually(t, clock.HasWaiters, 5*time.Second, time.Millisecond)
				clock.Step(delay - time.Nanosecond)
				require.Empty(t, attempts, "retry must honor exponential backoff")
				clock.Step(delay/10 + 2*time.Nanosecond)
				receive()
				require.FileExists(t, filepath.Join(output, "saved"))
				require.True(t, kl.podCheckpointInFlight(source.UID))
			}
			allowed.Store(true)
			require.Eventually(t, clock.HasWaiters, 5*time.Second, time.Millisecond)
			clock.Step(66 * time.Second)
			receive()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("publisher did not finish after permission recovered")
			}
			require.FileExists(t, filepath.Join(output, "saved"))
			require.False(t, kl.podCheckpointInFlight(source.UID))
			records, err := kl.readPodCheckpointOperations(source.UID)
			require.NoError(t, err)
			require.Empty(t, records)
		})
	}
}

func TestOlderCompletedRecordDoesNotBlockAnotherCheckpoint(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, source, pc := checkpointOperationFixture(t)
	kl.checkpointOperationContext.Store(&ctx)
	pc.Spec.SourcePod = &nodev1alpha1.PodReference{Name: source.Name}
	pc.Status.NodeName, pc.Status.SourcePodUID = new(string(kl.nodeName)), new(source.UID)
	older := pc.DeepCopy()
	older.Name, older.UID = "older", "older-checkpoint-uid"
	kl.kubeClient = fake.NewClientset(pc, older)
	seedCheckpointCache(t, kl)
	_, output, err := kl.createPodCheckpointOutput(older.UID)
	require.NoError(t, err)
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: older.UID, PodUID: source.UID, PodName: source.Name, Namespace: older.Namespace, Name: older.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}))
	require.NoError(t, kl.syncPodCheckpoint(ctx, pc.Namespace+"/"+pc.Name))
	require.Eventually(t, func() bool {
		stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
		return err == nil && apimeta.IsStatusConditionTrue(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	}, 5*time.Second, time.Millisecond)
	require.Eventually(t, func() bool {
		records, err := kl.readPodCheckpointOperations(source.UID)
		return err == nil && len(records) == 0
	}, 5*time.Second, time.Millisecond)
	runtime.Lock()
	defer runtime.Unlock()
	calls := 0
	for _, function := range runtime.CalledFunctions {
		if function == "CheckpointPod" {
			calls++
		}
	}
	require.Equal(t, 1, calls, "only the new checkpoint may invoke capture")
}

func TestStaleCheckpointRecordCannotRestartPublication(t *testing.T) {
	for _, recreated := range []bool{false, true} {
		t.Run(fmt.Sprintf("recreated=%t", recreated), func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, source, pc := checkpointOperationFixture(t)
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("captured state"), 0600))
			stale := podCheckpointOperation{CheckpointUID: pc.UID, PodUID: source.UID, PodName: source.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}
			require.NoError(t, kl.writePodCheckpointOperation(stale))
			require.NoError(t, kl.finalizeRecoveredPodCheckpoint(ctx, pc))
			require.NoError(t, kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Delete(ctx, pc.Name, metav1.DeleteOptions{}))
			if recreated {
				replacement := pc.DeepCopy()
				replacement.UID = "replacement-checkpoint-uid"
				_, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Create(ctx, replacement, metav1.CreateOptions{})
				require.NoError(t, err)
			}
			require.NoError(t, kl.publishRecordedPodCheckpointOutcome(ctx, stale))
			require.Eventually(t, func() bool { _, publishing := kl.checkpointPublications.Load(pc.UID); return !publishing }, 5*time.Second, time.Millisecond)
			require.FileExists(t, filepath.Join(output, "saved"), "an already published archive is not owned by stale recovery")
			if recreated {
				replacement, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
				require.NoError(t, err)
				require.Empty(t, replacement.Status.Conditions)
			}
		})
	}
}
