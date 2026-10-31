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

package job

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"testing"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/utils/ptr"
)

var errInjected = errors.New("injected error")

// flakyPodControl fails the first deletion of the named pods. It records
// deletion attempts, including the failed ones, separately from successful
// deletions, which the test reflects in the pod informer. Deletions don't go
// through the embedded FakePodControl, whose DeletePodName also records
// failed attempts.
type flakyPodControl struct {
	*controller.FakePodControl
	mu             sync.Mutex
	failDelete     map[string]bool
	deleteAttempts int
	deleted        []string
	deletedTotal   int
	patched        []string
}

func (f *flakyPodControl) PatchPod(ctx context.Context, namespace, name string, data []byte) error {
	if err := f.FakePodControl.PatchPod(ctx, namespace, name, data); err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.patched = append(f.patched, name)
	return nil
}

func (f *flakyPodControl) takePatched() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := f.patched
	f.patched = nil
	return p
}

func (f *flakyPodControl) pendingDeletions() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.deleted)
}

func (f *flakyPodControl) DeletePod(ctx context.Context, namespace, podID string, object runtime.Object) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleteAttempts++
	if f.failDelete[podID] {
		delete(f.failDelete, podID)
		return errInjected
	}
	f.deleted = append(f.deleted, podID)
	f.deletedTotal++
	return nil
}

// deletionCounts returns the number of deletion attempts and of successful
// deletions so far.
func (f *flakyPodControl) deletionCounts() (attempts, successful int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.deleteAttempts, f.deletedTotal
}

func (f *flakyPodControl) takeDeleted() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	d := f.deleted
	f.deleted = nil
	return d
}

func hasCondition(job *batch.Job, cType batch.JobConditionType, reason string) bool {
	for _, c := range job.Status.Conditions {
		if c.Type == cType && c.Status == v1.ConditionTrue && (reason == "" || c.Reason == reason) {
			return true
		}
	}
	return false
}

// TestInterimConditionPersistedBeforePodDeletion verifies that when the
// controller decides that a Job is finishing (FailureTarget or
// SuccessCriteriaMet) and deletes the remaining active pods, the decision is
// not lost if the sync is interrupted: the pods deleted by the controller must
// not be re-counted as failures that change the outcome on the next sync.
func TestInterimConditionPersistedBeforePodDeletion(t *testing.T) {
	type interruption int
	const (
		noInterruption interruption = iota
		failFirstStatusUpdate
		failOnePodDeletion
		unsatisfiedExpectations
		staleJobCache
	)
	deadlineJob := func(parallelism int32) *batch.Job {
		job := newJob(parallelism, parallelism, 0, batch.NonIndexedCompletion)
		job.Spec.ActiveDeadlineSeconds = ptr.To[int64](10)
		start := metav1.Unix(metav1.Now().Time.Unix()-15, 0)
		job.Status.StartTime = &start
		return job
	}
	successPolicyJob := func() *batch.Job {
		job := newJob(3, 3, 0, batch.IndexedCompletion)
		job.Spec.SuccessPolicy = &batch.SuccessPolicy{
			Rules: []batch.SuccessPolicyRule{{SucceededCount: ptr.To[int32](1)}},
		}
		now := metav1.Now()
		job.Status.StartTime = &now
		return job
	}
	cases := map[string]struct {
		job          *batch.Job
		indexed      bool
		runningPods  int
		interruption interruption
		wantType     batch.JobConditionType
		wantReason   string
	}{
		"deadline, no interruption": {
			job: deadlineJob(1), runningPods: 1, interruption: noInterruption,
			wantType: batch.JobFailureTarget, wantReason: batch.JobReasonDeadlineExceeded,
		},
		"success policy, no interruption": {
			job: successPolicyJob(), indexed: true, interruption: noInterruption,
			wantType: batch.JobSuccessCriteriaMet, wantReason: batch.JobReasonSuccessPolicy,
		},
		"deadline, status update after deletion fails": {
			job: deadlineJob(1), runningPods: 1, interruption: failFirstStatusUpdate,
			wantType: batch.JobFailureTarget, wantReason: batch.JobReasonDeadlineExceeded,
		},
		"deadline, one of the pod deletions fails": {
			job: deadlineJob(2), runningPods: 2, interruption: failOnePodDeletion,
			wantType: batch.JobFailureTarget, wantReason: batch.JobReasonDeadlineExceeded,
		},
		"deadline, expectations not satisfied": {
			job: deadlineJob(1), runningPods: 1, interruption: unsatisfiedExpectations,
			wantType: batch.JobFailureTarget, wantReason: batch.JobReasonDeadlineExceeded,
		},
		"success policy, status update after deletion fails": {
			job: successPolicyJob(), indexed: true, interruption: failFirstStatusUpdate,
			wantType: batch.JobSuccessCriteriaMet, wantReason: batch.JobReasonSuccessPolicy,
		},
		"success policy, one of the pod deletions fails": {
			job: successPolicyJob(), indexed: true, interruption: failOnePodDeletion,
			wantType: batch.JobSuccessCriteriaMet, wantReason: batch.JobReasonSuccessPolicy,
		},
		"success policy, expectations not satisfied": {
			job: successPolicyJob(), indexed: true, interruption: unsatisfiedExpectations,
			wantType: batch.JobSuccessCriteriaMet, wantReason: batch.JobReasonSuccessPolicy,
		},
		"deadline, Job informer stale after the status update": {
			job: deadlineJob(1), runningPods: 1, interruption: staleJobCache,
			wantType: batch.JobFailureTarget, wantReason: batch.JobReasonDeadlineExceeded,
		},
		"success policy, Job informer stale after the status update": {
			job: successPolicyJob(), indexed: true, interruption: staleJobCache,
			wantType: batch.JobSuccessCriteriaMet, wantReason: batch.JobReasonSuccessPolicy,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(ctx, t, clientSet, controller.NoResyncPeriodFunc)
			podControl := &flakyPodControl{FakePodControl: &controller.FakePodControl{}, failDelete: map[string]bool{}}
			manager.podControl = podControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady

			job := tc.job
			jobIndexer := sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer()
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			if err := jobIndexer.Add(job); err != nil {
				t.Fatal(err)
			}
			var podNames []string
			if tc.indexed {
				// Index 0 succeeded; indexes 1 and 2 are still running.
				for i, phase := range []v1.PodPhase{v1.PodSucceeded, v1.PodRunning, v1.PodRunning} {
					p := buildPod().name(fmt.Sprintf("pod-%d", i)).job(job).index(strconv.Itoa(i)).phase(phase).trackingFinalizer().Pod
					p.UID = types.UID("uid-" + p.Name)
					if err := podIndexer.Add(p); err != nil {
						t.Fatal(err)
					}
					if phase == v1.PodRunning {
						podNames = append(podNames, p.Name)
					}
				}
			} else {
				for i := range tc.runningPods {
					p := buildPod().name(fmt.Sprintf("pod-%d", i)).job(job).phase(v1.PodRunning).trackingFinalizer().Pod
					p.UID = types.UID("uid-" + p.Name)
					if err := podIndexer.Add(p); err != nil {
						t.Fatal(err)
					}
					podNames = append(podNames, p.Name)
				}
			}

			key := testutil.GetKey(job, t)
			switch tc.interruption {
			case failOnePodDeletion:
				podControl.failDelete[podNames[len(podNames)-1]] = true
			case unsatisfiedExpectations:
				if err := manager.expectations.ExpectCreations(logger, key, 1); err != nil {
					t.Fatal(err)
				}
			}

			// stored is the Job as persisted by the API server, which rejects
			// updates based on a stale resourceVersion.
			stored := job.DeepCopy()
			statusUpdates, conflicts := 0, 0
			var writes []batch.JobStatus
			manager.updateStatusHandler = func(ctx context.Context, j *batch.Job) (*batch.Job, error) {
				statusUpdates++
				if tc.interruption == failFirstStatusUpdate && statusUpdates == 1 {
					return nil, errInjected
				}
				if j.ResourceVersion != stored.ResourceVersion {
					conflicts++
					return nil, apierrors.NewConflict(batch.Resource("jobs"), j.Name, errors.New("stale resourceVersion"))
				}
				stored = j.DeepCopy()
				rv, _ := strconv.Atoi(stored.ResourceVersion)
				stored.ResourceVersion = strconv.Itoa(rv + 1)
				writes = append(writes, *stored.Status.DeepCopy())
				return stored.DeepCopy(), nil
			}

			// For staleJobCache, the Job informer does not observe the Job
			// for a few syncs after the first successful status update.
			staleSyncsLeft := 0
			if tc.interruption == staleJobCache {
				staleSyncsLeft = 3
			}
			successfulDeletes := map[string]int{}
			runSyncs := func(n int) {
				for range n {
					_ = manager.syncJob(ctx, key)
					if len(writes) > 0 && staleSyncsLeft > 0 {
						staleSyncsLeft--
					} else if err := jobIndexer.Update(stored.DeepCopy()); err != nil {
						t.Fatal(err)
					}
					for _, name := range podControl.takeDeleted() {
						successfulDeletes[name]++
						obj, exists, err := podIndexer.GetByKey(cache.MetaObjectToName(&metav1.ObjectMeta{Namespace: job.Namespace, Name: name}).String())
						if err != nil || !exists {
							t.Fatalf("pod %s not found: %v", name, err)
						}
						p := obj.(*v1.Pod).DeepCopy()
						p.DeletionTimestamp = new(metav1.Now())
						if err := podIndexer.Update(p); err != nil {
							t.Fatal(err)
						}
					}
					if tc.interruption == unsatisfiedExpectations {
						manager.expectations.CreationObserved(logger, key)
					}
				}
			}

			// Phase 1: the deleted pods are still terminating.
			runSyncs(6)
			for i, w := range writes {
				for _, c := range w.Conditions {
					if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
						t.Errorf("Status write %d added terminal condition %s while pods were still terminating: %+v", i, c.Type, w.Conditions)
					}
				}
			}
			if !hasCondition(stored, tc.wantType, tc.wantReason) {
				t.Errorf("Job is missing condition %s with reason %s; conditions: %+v", tc.wantType, tc.wantReason, stored.Status.Conditions)
			}
			if tc.interruption == staleJobCache && conflicts == 0 {
				t.Error("Expected status updates based on the stale Job to be rejected, but there were none")
			}

			// Phase 2: the terminating pods are gone.
			for _, obj := range podIndexer.List() {
				if p := obj.(*v1.Pod); p.DeletionTimestamp != nil {
					if err := podIndexer.Delete(p); err != nil {
						t.Fatal(err)
					}
				}
			}
			runSyncs(3)
			wantTerminal := batch.JobComplete
			if tc.wantType == batch.JobFailureTarget {
				wantTerminal = batch.JobFailed
			}
			if !hasCondition(stored, wantTerminal, tc.wantReason) {
				t.Errorf("Job is missing terminal condition %s with reason %s; conditions: %+v", wantTerminal, tc.wantReason, stored.Status.Conditions)
			}

			// No write, at any point, recorded a conflicting decision.
			for i, w := range writes {
				for _, c := range w.Conditions {
					if c.Status != v1.ConditionTrue {
						continue
					}
					conflicting := false
					switch c.Type {
					case batch.JobFailureTarget, batch.JobFailed:
						conflicting = tc.wantType != batch.JobFailureTarget || c.Reason != tc.wantReason
					case batch.JobSuccessCriteriaMet, batch.JobComplete:
						conflicting = tc.wantType != batch.JobSuccessCriteriaMet
					}
					if conflicting {
						t.Errorf("Status write %d recorded a conflicting %s=%s condition: %+v", i, c.Type, c.Reason, w.Conditions)
					}
				}
			}
			for name, n := range successfulDeletes {
				if n > 1 {
					t.Errorf("Pod %s was deleted %d times", name, n)
				}
			}
		})
	}
}

// TestPreExistingInterimConditionIsNotPersistedAgain verifies that when the
// Job already has the interim condition, the controller deletes the active
// pods without an additional status update and never duplicates the
// condition.
func TestPreExistingInterimConditionIsNotPersistedAgain(t *testing.T) {
	cases := map[string]struct {
		condition batch.JobCondition
		indexed   bool
	}{
		"FailureTarget": {
			condition: batch.JobCondition{Type: batch.JobFailureTarget, Status: v1.ConditionTrue, Reason: batch.JobReasonDeadlineExceeded, Message: "Job was active longer than specified deadline"},
		},
		"SuccessCriteriaMet": {
			condition: batch.JobCondition{Type: batch.JobSuccessCriteriaMet, Status: v1.ConditionTrue, Reason: batch.JobReasonSuccessPolicy, Message: "Matched rules at index 0"},
			indexed:   true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(ctx, t, clientSet, controller.NoResyncPeriodFunc)
			podControl := &flakyPodControl{FakePodControl: &controller.FakePodControl{}, failDelete: map[string]bool{}}
			manager.podControl = podControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady

			var job *batch.Job
			var phases []v1.PodPhase
			if tc.indexed {
				job = newJob(3, 3, 0, batch.IndexedCompletion)
				job.Spec.SuccessPolicy = &batch.SuccessPolicy{Rules: []batch.SuccessPolicyRule{{SucceededCount: ptr.To[int32](1)}}}
				job.Status.CompletedIndexes = "0"
				job.Status.Succeeded = 1
				phases = []v1.PodPhase{v1.PodSucceeded, v1.PodRunning, v1.PodRunning}
			} else {
				job = newJob(2, 2, 0, batch.NonIndexedCompletion)
				job.Spec.ActiveDeadlineSeconds = ptr.To[int64](10)
				phases = []v1.PodPhase{v1.PodRunning, v1.PodRunning}
			}
			job.Status.StartTime = new(metav1.Unix(metav1.Now().Unix()-15, 0))
			job.Status.Conditions = []batch.JobCondition{tc.condition}
			if err := sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job); err != nil {
				t.Fatal(err)
			}
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			for i, phase := range phases {
				pb := buildPod().name(fmt.Sprintf("pod-%d", i)).job(job).phase(phase).trackingFinalizer()
				if tc.indexed {
					pb = pb.index(strconv.Itoa(i))
				}
				if err := podIndexer.Add(pb.Pod); err != nil {
					t.Fatal(err)
				}
			}

			var writes []batch.JobStatus
			manager.updateStatusHandler = func(ctx context.Context, j *batch.Job) (*batch.Job, error) {
				if attempts, _ := podControl.deletionCounts(); attempts == 0 {
					t.Errorf("Status update before deleting the active pods, although the Job already has the %s condition", tc.condition.Type)
				}
				writes = append(writes, *j.Status.DeepCopy())
				return j, nil
			}
			if err := manager.syncJob(ctx, testutil.GetKey(job, t)); err != nil {
				t.Fatalf("syncJob: %v", err)
			}
			if attempts, successful := podControl.deletionCounts(); attempts != 2 || successful != 2 {
				t.Errorf("Got %d deletion attempts and %d successful deletions, want 2 and 2", attempts, successful)
			}
			if len(writes) == 0 {
				t.Fatal("No status update")
			}
			for i, w := range writes {
				if n := len(getConditionsByType(w.Conditions, tc.condition.Type)); n != 1 {
					t.Errorf("Status update %d has %d %s conditions, want 1: %+v", i, n, tc.condition.Type, w.Conditions)
				}
			}
		})
	}
}

// TestTerminalConditionWaitsForTerminatingPods verifies the status updates of
// each sync after the interim condition is persisted: the terminal Failed or
// Complete condition is not added while the deleted pods are terminating,
// and is added once they are terminated.
func TestTerminalConditionWaitsForTerminatingPods(t *testing.T) {
	cases := map[string]struct {
		job          *batch.Job
		phases       []v1.PodPhase
		indexed      bool
		wantInterim  batch.JobConditionType
		wantTerminal batch.JobConditionType
		wantReason   string
	}{
		"deadline": {
			job: func() *batch.Job {
				j := newJob(2, 2, 0, batch.NonIndexedCompletion)
				j.Spec.ActiveDeadlineSeconds = ptr.To[int64](10)
				j.Status.StartTime = new(metav1.Unix(metav1.Now().Unix()-15, 0))
				return j
			}(),
			phases:       []v1.PodPhase{v1.PodRunning, v1.PodRunning},
			wantInterim:  batch.JobFailureTarget,
			wantTerminal: batch.JobFailed,
			wantReason:   batch.JobReasonDeadlineExceeded,
		},
		"success policy": {
			job: func() *batch.Job {
				j := newJob(3, 3, 0, batch.IndexedCompletion)
				j.Spec.SuccessPolicy = &batch.SuccessPolicy{Rules: []batch.SuccessPolicyRule{{SucceededCount: ptr.To[int32](1)}}}
				j.Status.StartTime = new(metav1.Now())
				return j
			}(),
			phases:       []v1.PodPhase{v1.PodSucceeded, v1.PodRunning, v1.PodRunning},
			indexed:      true,
			wantInterim:  batch.JobSuccessCriteriaMet,
			wantTerminal: batch.JobComplete,
			wantReason:   batch.JobReasonSuccessPolicy,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(ctx, t, clientSet, controller.NoResyncPeriodFunc)
			podControl := &flakyPodControl{FakePodControl: &controller.FakePodControl{}, failDelete: map[string]bool{}}
			manager.podControl = podControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			jobIndexer := sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer()
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()

			job := tc.job
			if err := jobIndexer.Add(job); err != nil {
				t.Fatal(err)
			}
			for i, phase := range tc.phases {
				pb := buildPod().name(fmt.Sprintf("pod-%d", i)).job(job).phase(phase).trackingFinalizer()
				if tc.indexed {
					pb = pb.index(strconv.Itoa(i))
				}
				if err := podIndexer.Add(pb.Pod); err != nil {
					t.Fatal(err)
				}
			}
			stored := job.DeepCopy()
			var writes []batch.JobStatus
			manager.updateStatusHandler = func(ctx context.Context, j *batch.Job) (*batch.Job, error) {
				stored = j.DeepCopy()
				writes = append(writes, *stored.Status.DeepCopy())
				return stored.DeepCopy(), nil
			}
			key := testutil.GetKey(job, t)
			syncOnce := func() []batch.JobStatus {
				t.Helper()
				writes = nil
				if err := manager.syncJob(ctx, key); err != nil {
					t.Fatalf("syncJob: %v", err)
				}
				if err := jobIndexer.Update(stored.DeepCopy()); err != nil {
					t.Fatal(err)
				}
				return writes
			}
			setDeletedPods := func(mutate func(*v1.Pod)) {
				t.Helper()
				for _, name := range podControl.takeDeleted() {
					obj, _, err := podIndexer.GetByKey(job.Namespace + "/" + name)
					if err != nil || obj == nil {
						t.Fatalf("pod %s not found: %v", name, err)
					}
					p := obj.(*v1.Pod).DeepCopy()
					mutate(p)
					if err := podIndexer.Update(p); err != nil {
						t.Fatal(err)
					}
				}
			}
			terminal := func(w batch.JobStatus) *batch.JobCondition {
				for i, c := range w.Conditions {
					if (c.Type == batch.JobFailed || c.Type == batch.JobComplete) && c.Status == v1.ConditionTrue {
						return &w.Conditions[i]
					}
				}
				return nil
			}

			// Sync 1: the interim condition is persisted alone, then the pods
			// are deleted and remain terminating.
			sync1 := syncOnce()
			if len(sync1) < 2 {
				t.Fatalf("Got %d status updates in the first sync, want at least 2", len(sync1))
			}
			if c := sync1[0].Conditions; len(c) != 1 || c[0].Type != tc.wantInterim || c[0].Reason != tc.wantReason {
				t.Errorf("First status update has conditions %+v, want only %s with reason %s", c, tc.wantInterim, tc.wantReason)
			}
			for i, w := range sync1 {
				if c := terminal(w); c != nil {
					t.Errorf("Sync 1, status update %d added %s while the deleted pods are terminating", i, c.Type)
				}
			}
			if got := ptr.Deref(sync1[len(sync1)-1].Terminating, 0); got != 2 {
				t.Errorf("Got terminating=%d after the first sync, want 2", got)
			}

			// Sync 2: the informer observes the deletions; the pods are still
			// terminating.
			setDeletedPods(func(p *v1.Pod) {
				p.DeletionTimestamp = new(metav1.Now())
				podControl.mu.Lock()
				podControl.deleted = append(podControl.deleted, p.Name)
				podControl.mu.Unlock()
			})
			for i, w := range syncOnce() {
				if c := terminal(w); c != nil {
					t.Errorf("Sync 2, status update %d added %s while the deleted pods are terminating", i, c.Type)
				}
			}

			// Sync 3: the pods have terminated.
			setDeletedPods(func(p *v1.Pod) {
				p.Status.Phase = v1.PodFailed
			})
			syncOnce()
			if c := terminal(stored.Status); c == nil || c.Type != tc.wantTerminal || c.Reason != tc.wantReason {
				t.Errorf("Got terminal condition %+v once the pods terminated, want %s with reason %s", c, tc.wantTerminal, tc.wantReason)
			}
		})
	}
}
