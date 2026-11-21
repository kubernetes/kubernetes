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
	"math"
	"slices"
	"strconv"
	"testing"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	consistencyutil "k8s.io/client-go/util/consistency"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/utils/ptr"
)

type scenarioInterruption string

const (
	scenarioNone                    scenarioInterruption = "none"
	scenarioLostStatusUpdate        scenarioInterruption = "lost status update"
	scenarioLostStatusResponse      scenarioInterruption = "status update applied but response lost"
	scenarioFailedPodDeletion       scenarioInterruption = "failed pod deletion"
	scenarioUnsatisfiedExpectations scenarioInterruption = "unsatisfied expectations"
	scenarioStaleJobInformer        scenarioInterruption = "stale Job informer"
	scenarioControllerRestart       scenarioInterruption = "controller restart after deletion"
)

type scenarioPod struct {
	index    *int
	phase    v1.PodPhase
	exitCode int32
}

type scenarioOutcome struct {
	condition batch.JobConditionType
	reason    string
}

// TestInterimConditionScenarios runs Jobs that finish with active pods
// through interruptions of the sync that deletes those pods, and verifies
// that every interruption leads to the same terminal condition and reason as
// the uninterrupted run. The stale Job informer case runs with the
// StaleControllerConsistencyJob read-your-writes guarantee, as in the
// controller's default configuration.
func TestInterimConditionScenarios(t *testing.T) {
	deadline := func(j *batch.Job) *batch.Job {
		j.Spec.ActiveDeadlineSeconds = ptr.To[int64](10)
		j.Status.StartTime = new(metav1.Unix(metav1.Now().Unix()-15, 0))
		return j
	}
	started := func(j *batch.Job) *batch.Job {
		j.Status.StartTime = new(metav1.Now())
		return j
	}
	withPFP := func(j *batch.Job, rules ...batch.PodFailurePolicyRule) *batch.Job {
		j.Spec.PodFailurePolicy = &batch.PodFailurePolicy{Rules: rules}
		j.Spec.PodReplacementPolicy = ptr.To(batch.Failed)
		return j
	}
	onExitCode := func(action batch.PodFailurePolicyAction, code int32) batch.PodFailurePolicyRule {
		return batch.PodFailurePolicyRule{
			Action:      action,
			OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{Operator: batch.PodFailurePolicyOnExitCodesOpIn, Values: []int32{code}},
		}
	}
	perIndex := func(j *batch.Job, backoffLimitPerIndex, maxFailedIndexes int32) *batch.Job {
		j.Spec.BackoffLimitPerIndex = new(backoffLimitPerIndex)
		j.Spec.MaxFailedIndexes = new(maxFailedIndexes)
		return j
	}
	successPolicy := func(j *batch.Job) *batch.Job {
		j.Spec.SuccessPolicy = &batch.SuccessPolicy{Rules: []batch.SuccessPolicyRule{{SucceededCount: ptr.To[int32](1)}}}
		return j
	}
	running := func(n int) []scenarioPod {
		var pods []scenarioPod
		for range n {
			pods = append(pods, scenarioPod{phase: v1.PodRunning})
		}
		return pods
	}
	indexed := func(phases ...v1.PodPhase) []scenarioPod {
		var pods []scenarioPod
		for i, p := range phases {
			pods = append(pods, scenarioPod{index: new(i), phase: p})
		}
		return pods
	}
	repeat := func(p v1.PodPhase, n int) []v1.PodPhase {
		var r []v1.PodPhase
		for range n {
			r = append(r, p)
		}
		return r
	}
	nonIndexed, indexedMode := batch.NonIndexedCompletion, batch.IndexedCompletion
	withExitCode := func(pods []scenarioPod, i int, code int32) []scenarioPod {
		pods[i].phase = v1.PodFailed
		pods[i].exitCode = code
		return pods
	}

	cases := map[string]struct {
		job *batch.Job
		// pods is the initial state of the Job's pods.
		pods []scenarioPod
		// killWithExitCode makes the pods deleted by the controller fail with
		// the given exit code, as if the kubelet terminated them.
		killWithExitCode *int32
		want             scenarioOutcome
	}{
		"deadline": {
			job:  deadline(newJob(1, 1, 0, nonIndexed)),
			pods: running(1),
			want: scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonDeadlineExceeded},
		},
		"deadline, multiple active pods": {
			job:  deadline(newJob(3, 3, 0, nonIndexed)),
			pods: running(3),
			want: scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonDeadlineExceeded},
		},
		"deadline, podReplacementPolicy=Failed, deleted pods fail": {
			job: func() *batch.Job {
				j := deadline(newJob(2, 2, 0, nonIndexed))
				j.Spec.PodReplacementPolicy = ptr.To(batch.Failed)
				return j
			}(),
			pods:             running(2),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonDeadlineExceeded},
		},
		"deadline, podFailurePolicy FailJob matches the exit code of deleted pods": {
			job:              withPFP(deadline(newJob(2, 2, 6, nonIndexed)), onExitCode(batch.PodFailurePolicyActionFailJob, 143)),
			pods:             running(2),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonDeadlineExceeded},
		},
		"podFailurePolicy FailJob": {
			job:              withPFP(started(newJob(2, 2, 0, nonIndexed)), onExitCode(batch.PodFailurePolicyActionFailJob, 42)),
			pods:             withExitCode(running(2), 0, 42),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonPodFailurePolicy},
		},
		"backoffLimit, podFailurePolicy FailJob matches the exit code of deleted pods": {
			job:              withPFP(started(newJob(2, 2, 0, nonIndexed)), onExitCode(batch.PodFailurePolicyActionFailJob, 143)),
			pods:             withExitCode(running(2), 0, 1),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonBackoffLimitExceeded},
		},
		"maxFailedIndexes": {
			job:  perIndex(started(newJob(4, 4, math.MaxInt32, indexedMode)), 0, 0),
			pods: indexed(v1.PodFailed, v1.PodRunning, v1.PodRunning, v1.PodRunning),
			want: scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonMaxFailedIndexesExceeded},
		},
		"maxFailedIndexes, with backoffLimit": {
			job:  perIndex(started(newJob(4, 4, 1, indexedMode)), 0, 0),
			pods: indexed(v1.PodFailed, v1.PodRunning, v1.PodRunning, v1.PodRunning),
			want: scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonMaxFailedIndexesExceeded},
		},
		"podFailurePolicy FailIndex": {
			job:              withPFP(perIndex(started(newJob(3, 3, math.MaxInt32, indexedMode)), 1, 0), onExitCode(batch.PodFailurePolicyActionFailIndex, 42)),
			pods:             withExitCode(indexed(v1.PodRunning, v1.PodRunning, v1.PodRunning), 0, 42),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobFailed, reason: batch.JobReasonMaxFailedIndexesExceeded},
		},
		"success policy": {
			job:  successPolicy(started(newJob(3, 3, 0, indexedMode))),
			pods: indexed(v1.PodSucceeded, v1.PodRunning, v1.PodRunning),
			want: scenarioOutcome{condition: batch.JobComplete, reason: batch.JobReasonSuccessPolicy},
		},
		"success policy, default backoffLimit": {
			job:  successPolicy(started(newJob(10, 10, 6, indexedMode))),
			pods: indexed(append([]v1.PodPhase{v1.PodSucceeded}, repeat(v1.PodRunning, 9)...)...),
			want: scenarioOutcome{condition: batch.JobComplete, reason: batch.JobReasonSuccessPolicy},
		},
		"success policy, backoffLimitPerIndex": {
			job:  successPolicy(perIndex(started(newJob(3, 3, math.MaxInt32, indexedMode)), 0, 0)),
			pods: indexed(v1.PodSucceeded, v1.PodRunning, v1.PodRunning),
			want: scenarioOutcome{condition: batch.JobComplete, reason: batch.JobReasonSuccessPolicy},
		},
		"success policy, podReplacementPolicy=Failed, deleted pods fail": {
			job: func() *batch.Job {
				j := successPolicy(started(newJob(3, 3, 0, indexedMode)))
				j.Spec.PodReplacementPolicy = ptr.To(batch.Failed)
				return j
			}(),
			pods:             indexed(v1.PodSucceeded, v1.PodRunning, v1.PodRunning),
			killWithExitCode: ptr.To[int32](143),
			want:             scenarioOutcome{condition: batch.JobComplete, reason: batch.JobReasonSuccessPolicy},
		},
	}
	interruptions := []scenarioInterruption{
		scenarioNone, scenarioLostStatusUpdate, scenarioLostStatusResponse, scenarioFailedPodDeletion,
		scenarioUnsatisfiedExpectations, scenarioStaleJobInformer, scenarioControllerRestart,
	}
	for name, tc := range cases {
		for _, interruption := range interruptions {
			t.Run(fmt.Sprintf("%s, %s", name, interruption), func(t *testing.T) {
				got := runInterimConditionScenario(t, tc.job.DeepCopy(), tc.pods, tc.killWithExitCode, interruption)
				if got != tc.want {
					t.Errorf("Unexpected outcome\n got: %+v\nwant: %+v", got, tc.want)
				}
			})
		}
	}
}

func runInterimConditionScenario(t *testing.T, job *batch.Job, initialPods []scenarioPod, killWithExitCode *int32, interruption scenarioInterruption) scenarioOutcome {
	t.Helper()
	logger, ctx := ktesting.NewTestContext(t)
	key := testutil.GetKey(job, t)

	// The cluster state, as seen by the API server.
	stored := job.DeepCopy()
	pods := map[string]*v1.Pod{}
	var activeNames []string
	for i, sp := range initialPods {
		pb := buildPod().name(fmt.Sprintf("pod-%d", i)).job(job).phase(sp.phase).trackingFinalizer()
		if sp.index != nil {
			pb = pb.index(strconv.Itoa(*sp.index))
		}
		p := pb.Pod
		p.UID = types.UID("uid-" + p.Name)
		if sp.phase == v1.PodFailed || sp.phase == v1.PodSucceeded {
			p.Status.ContainerStatuses = []v1.ContainerStatus{{Name: "main", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: sp.exitCode, FinishedAt: metav1.Now()}}}}
		} else {
			activeNames = append(activeNames, p.Name)
		}
		pods[p.Name] = p
	}

	podControl := &flakyPodControl{FakePodControl: &controller.FakePodControl{}, failDelete: map[string]bool{}}
	if interruption == scenarioFailedPodDeletion {
		podControl.failDelete[activeNames[len(activeNames)-1]] = true
	}

	var manager *Controller
	var jobIndexer, podIndexer cache.Indexer
	statusUpdates := 0
	// Evidence that the interruption was exercised.
	var (
		lostStatusUpdate     bool
		lostStatusResponse   bool
		crashed              bool
		crashAfterDeletions  int
		restarts             int
		unsatisfiedFirstSync bool
		staleSyncs           int
		staleSyncsBlocked    int
	)
	newManager := func() {
		clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
		m, informerFactory := newControllerFromClient(ctx, t, clientSet, controller.NoResyncPeriodFunc)
		m.podControl = podControl
		m.podStoreSynced = alwaysReady
		m.jobStoreSynced = alwaysReady
		m.updateStatusHandler = func(ctx context.Context, j *batch.Job) (*batch.Job, error) {
			statusUpdates++
			if interruption == scenarioLostStatusUpdate && statusUpdates == 1 {
				lostStatusUpdate = true
				return nil, errInjected
			}
			if interruption == scenarioControllerRestart && !crashed && podControl.pendingDeletions() > 0 {
				// The controller crashes after deleting the pods, before
				// this status update.
				crashed = true
				crashAfterDeletions = podControl.pendingDeletions()
				return nil, errInjected
			}
			if j.ResourceVersion != stored.ResourceVersion {
				return nil, apierrors.NewConflict(batch.Resource("jobs"), j.Name, errors.New("stale resourceVersion"))
			}
			seen := map[batch.JobConditionType]bool{}
			for _, c := range j.Status.Conditions {
				if seen[c.Type] {
					t.Errorf("Status update has a duplicate %s condition: %+v", c.Type, j.Status.Conditions)
				}
				seen[c.Type] = true
			}
			for _, c := range j.Status.Conditions {
				if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
					for _, p := range pods {
						if p.DeletionTimestamp != nil && p.Status.Phase != v1.PodFailed && p.Status.Phase != v1.PodSucceeded {
							t.Errorf("Terminal condition %s added while pod %s is still terminating", c.Type, p.Name)
						}
					}
				}
			}
			stored = j.DeepCopy()
			rv, _ := strconv.Atoi(stored.ResourceVersion)
			stored.ResourceVersion = strconv.Itoa(rv + 1)
			if interruption == scenarioLostStatusResponse && !lostStatusResponse {
				// The API server applied the update, but the response is lost.
				lostStatusResponse = true
				return nil, errInjected
			}
			if interruption == scenarioStaleJobInformer {
				// What updateJobStatus records when StaleControllerConsistencyJob is enabled.
				m.consistencyStore.WroteAt(types.NamespacedName{Namespace: stored.Namespace, Name: stored.Name}, stored.UID, jobGroupResource, stored.ResourceVersion)
			}
			return stored.DeepCopy(), nil
		}
		manager = m
		jobIndexer = informerFactory.Batch().V1().Jobs().Informer().GetIndexer()
		podIndexer = informerFactory.Core().V1().Pods().Informer().GetIndexer()
		if err := jobIndexer.Add(stored.DeepCopy()); err != nil {
			t.Fatal(err)
		}
	}
	newManager()
	if interruption == scenarioUnsatisfiedExpectations {
		if err := manager.expectations.ExpectCreations(logger, key, 1); err != nil {
			t.Fatal(err)
		}
	}

	staleSyncsLeft := 0
	if interruption == scenarioStaleJobInformer {
		staleSyncsLeft = 3
	}
	successfulWrites := func() bool { return stored.ResourceVersion != job.ResourceVersion }
	verifyInterruption := func() {
		t.Helper()
		switch interruption {
		case scenarioLostStatusUpdate:
			if !lostStatusUpdate {
				t.Error("The status update was never lost")
			}
		case scenarioLostStatusResponse:
			if !lostStatusResponse {
				t.Error("No status update response was lost")
			}
		case scenarioFailedPodDeletion:
			if len(podControl.failDelete) != 0 {
				t.Errorf("The injected pod deletion failure was never hit: %v", podControl.failDelete)
			}
		case scenarioUnsatisfiedExpectations:
			if !unsatisfiedFirstSync {
				t.Error("The first sync didn't run with unsatisfied expectations")
			}
		case scenarioStaleJobInformer:
			if staleSyncs != 3 {
				t.Errorf("Got %d syncs with a stale Job informer, want 3", staleSyncs)
			}
			if staleSyncsBlocked == 0 {
				t.Error("No sync was blocked by the read-your-writes check")
			}
		case scenarioControllerRestart:
			if !crashed || crashAfterDeletions == 0 {
				t.Errorf("The controller didn't crash after deleting pods (crashed=%v, deletions before the crash=%d)", crashed, crashAfterDeletions)
			}
			if restarts != 1 {
				t.Errorf("Got %d controller restarts, want 1", restarts)
			}
		}
	}
	for sync := range 20 {
		var podObjs []interface{}
		for _, p := range pods {
			podObjs = append(podObjs, p.DeepCopy())
		}
		if err := podIndexer.Replace(podObjs, ""); err != nil {
			t.Fatal(err)
		}
		stale := successfulWrites() && staleSyncsLeft > 0
		if stale {
			staleSyncsLeft--
			staleSyncs++
		} else if err := jobIndexer.Replace([]interface{}{stored.DeepCopy()}, stored.ResourceVersion); err != nil {
			t.Fatal(err)
		}
		if sync == 0 && interruption == scenarioUnsatisfiedExpectations {
			unsatisfiedFirstSync = !manager.expectations.SatisfiedExpectations(logger, key)
		}

		err := manager.syncJob(ctx, key)
		var consistencyErr *consistencyutil.ConsistencyError
		if stale && errors.As(err, &consistencyErr) {
			staleSyncsBlocked++
		}

		// Apply the effects of the pod deletions and finalizer removals.
		for _, name := range podControl.takeDeleted() {
			p := pods[name]
			if p == nil || p.DeletionTimestamp != nil {
				continue
			}
			p.DeletionTimestamp = new(metav1.Now())
			if killWithExitCode != nil && p.Status.Phase != v1.PodSucceeded && p.Status.Phase != v1.PodFailed {
				p.Status.Phase = v1.PodFailed
				p.Status.ContainerStatuses = []v1.ContainerStatus{{Name: "main", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: *killWithExitCode, FinishedAt: metav1.Now()}}}}
			}
		}
		for _, name := range podControl.takePatched() {
			if p := pods[name]; p != nil {
				p.Finalizers = slices.DeleteFunc(p.Finalizers, func(f string) bool { return f == batch.JobTrackingFinalizer })
			}
		}
		for name, p := range pods {
			if p.DeletionTimestamp != nil && len(p.Finalizers) == 0 {
				delete(pods, name)
			}
		}
		if interruption == scenarioUnsatisfiedExpectations {
			manager.expectations.CreationObserved(logger, key)
		}
		if interruption == scenarioControllerRestart && crashed && restarts == 0 {
			newManager()
			restarts++
		}
		for _, c := range stored.Status.Conditions {
			if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
				verifyInterruption()
				return scenarioOutcome{condition: c.Type, reason: c.Reason}
			}
		}
	}
	verifyInterruption()
	t.Fatalf("Job did not finish; conditions: %+v", stored.Status.Conditions)
	return scenarioOutcome{}
}
