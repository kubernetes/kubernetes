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

package leaderelection

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
	fakeclient "k8s.io/client-go/testing"
	rl "k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	recoveryTestLeaseDuration = 15 * time.Second
	recoveryTestRenewDeadline = 1 * time.Second
	recoveryTestRetryPeriod   = 200 * time.Millisecond
	recoveryTestIdentity      = "baz"
)

// recoveryHarness runs a LeaderElector against a fake lease store whose API
// calls can be made to fail on demand, simulating API server unavailability.
type recoveryHarness struct {
	t *testing.T

	mu         sync.Mutex
	lockObj    runtime.Object
	failing    bool
	creates    int
	failedGets int
	// failRenewUpdates makes renewal updates fail with a conflict while
	// letting release updates through, so recovery runs out of time.
	failRenewUpdates bool
	// releaseSawGateOpen records whether a write through the gate passed at
	// the moment the lease was released.
	releaseSawGateOpen bool

	// hangNextUpdate makes the next Update block until its context is
	// cancelled; updateHanging is closed when it starts blocking.
	hangNextUpdate atomic.Bool
	updateHanging  chan struct{}

	recorder *record.FakeRecorder

	started    chan struct{}
	startedCtx context.Context
	stopped    chan struct{}
	newLeader  chan string

	gated  http.RoundTripper
	done   chan struct{}
	cancel context.CancelFunc
}

// harnessLock wraps the lease lock so that requests fail once their context
// is cancelled, as they would against a real API server (the fake clientset
// ignores contexts), and so that a test can make an Update hang until its
// context is cancelled, as a request to an unresponsive API would.
type harnessLock struct {
	rl.Interface
	h *recoveryHarness
}

func (l *harnessLock) Get(ctx context.Context) (*rl.LeaderElectionRecord, []byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return l.Interface.Get(ctx)
}

func (l *harnessLock) Update(ctx context.Context, ler rl.LeaderElectionRecord) error {
	if l.h.hangNextUpdate.CompareAndSwap(true, false) {
		close(l.h.updateHanging)
		<-ctx.Done()
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return l.Interface.Update(ctx, ler)
}

// withReleaseOnCancel configures the harness elector to release the lease
// when its run ends.
func withReleaseOnCancel(lec *LeaderElectionConfig) {
	lec.ReleaseOnCancel = true
}

func newRecoveryHarness(t *testing.T, recovery *RecoveryConfig, opts ...func(*LeaderElectionConfig)) *recoveryHarness {
	h := &recoveryHarness{
		t:         t,
		recorder:  record.NewFakeRecorder(100),
		started:   make(chan struct{}),
		stopped:   make(chan struct{}, 100),
		newLeader: make(chan string, 100),
		done:      make(chan struct{}),
	}

	unavailable := func() error {
		return apierrors.NewServiceUnavailable("apiserver unavailable")
	}
	c := &fake.Clientset{}
	c.AddReactor("get", "leases", func(action fakeclient.Action) (bool, runtime.Object, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.failing {
			h.failedGets++
			return true, nil, unavailable()
		}
		if h.lockObj == nil {
			return true, nil, apierrors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
		}
		return true, h.lockObj.DeepCopyObject(), nil
	})
	c.AddReactor("create", "leases", func(action fakeclient.Action) (bool, runtime.Object, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.failing {
			return true, nil, unavailable()
		}
		h.creates++
		h.lockObj = action.(fakeclient.CreateAction).GetObject().DeepCopyObject()
		return true, h.lockObj.DeepCopyObject(), nil
	})
	c.AddReactor("update", "leases", func(action fakeclient.Action) (bool, runtime.Object, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.failing {
			return true, nil, unavailable()
		}
		obj := action.(fakeclient.UpdateAction).GetObject()
		if h.lockObj == nil {
			return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), obj.(*coordinationv1.Lease).Name)
		}
		if seconds := obj.(*coordinationv1.Lease).Spec.LeaseDurationSeconds; seconds != nil && *seconds == 1 {
			// A release: record whether gated writes were still possible.
			h.releaseSawGateOpen = h.write() == nil
		}
		if h.failRenewUpdates {
			if seconds := obj.(*coordinationv1.Lease).Spec.LeaseDurationSeconds; seconds == nil || *seconds != 1 {
				// A release sets LeaseDurationSeconds to 1; everything else is a renewal.
				return true, nil, apierrors.NewConflict(coordinationv1.Resource("leases"), obj.(*coordinationv1.Lease).Name, fmt.Errorf("simulated conflict"))
			}
		}
		h.lockObj = obj.DeepCopyObject()
		return true, h.lockObj.DeepCopyObject(), nil
	})
	c.AddReactor("*", "*", func(action fakeclient.Action) (bool, runtime.Object, error) {
		t.Errorf("unreachable action: %+v", action)
		return true, nil, fmt.Errorf("unreachable action")
	})

	lock, err := rl.New("leases", "foo", "bar", c.CoreV1(), c.CoordinationV1(), rl.ResourceLockConfig{
		Identity:      recoveryTestIdentity,
		EventRecorder: h.recorder,
	})
	if err != nil {
		t.Fatal("resourcelock.New() = ", err)
	}

	h.updateHanging = make(chan struct{})
	lec := LeaderElectionConfig{
		Lock:          &harnessLock{Interface: lock, h: h},
		LeaseDuration: recoveryTestLeaseDuration,
		RenewDeadline: recoveryTestRenewDeadline,
		RetryPeriod:   recoveryTestRetryPeriod,
		Recovery:      recovery,
		Callbacks: LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				h.startedCtx = ctx
				close(h.started)
			},
			OnStoppedLeading: func() {
				h.stopped <- struct{}{}
			},
			OnNewLeader: func(identity string) {
				h.newLeader <- identity
			},
		},
	}
	for _, opt := range opts {
		opt(&lec)
	}
	elector, err := NewLeaderElector(lec)
	if err != nil {
		t.Fatal("Failed to create leader elector: ", err)
	}
	h.gated = elector.WriteGate()(roundTripperFunc(okResponse))

	_, ctx := ktesting.NewTestContext(t)
	ctx, h.cancel = context.WithCancel(ctx)
	go func() {
		defer close(h.done)
		elector.Run(ctx)
	}()
	t.Cleanup(func() {
		h.cancel()
		h.waitForRunToReturn()
	})

	select {
	case <-h.started:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("failed to become the leader")
	}
	h.expectWriteAllowed()
	h.expectNewLeader(recoveryTestIdentity)
	return h
}

func (h *recoveryHarness) setFailing(failing bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.failing = failing
}

// write sends a POST through the gated transport.
func (h *recoveryHarness) write() error {
	req, err := http.NewRequest(http.MethodPost, "https://localhost/", strings.NewReader("body"))
	if err != nil {
		h.t.Fatal(err)
	}
	resp, err := h.gated.RoundTrip(req)
	if err == nil {
		_ = resp.Body.Close()
	}
	return err
}

func (h *recoveryHarness) expectWriteAllowed() {
	h.t.Helper()
	if err := h.write(); err != nil {
		h.t.Fatalf("expected write to pass the gate, got %v", err)
	}
}

func (h *recoveryHarness) expectWriteRejected() {
	h.t.Helper()
	if err := h.write(); !errors.Is(err, ErrWriteGateClosed) {
		h.t.Fatalf("expected write to be rejected with ErrWriteGateClosed, got %v", err)
	}
}

func (h *recoveryHarness) expectStoppedLeading() {
	h.t.Helper()
	select {
	case <-h.stopped:
	case <-time.After(wait.ForeverTestTimeout):
		h.t.Fatal("OnStoppedLeading was not called")
	}
}

func (h *recoveryHarness) expectNoStoppedLeading() {
	h.t.Helper()
	select {
	case <-h.stopped:
		h.t.Fatal("OnStoppedLeading was called unexpectedly")
	default:
	}
}

func (h *recoveryHarness) expectNewLeader(identity string) {
	h.t.Helper()
	select {
	case got := <-h.newLeader:
		if got != identity {
			h.t.Fatalf("expected OnNewLeader(%q), got %q", identity, got)
		}
	case <-time.After(wait.ForeverTestTimeout):
		h.t.Fatalf("OnNewLeader(%q) was not called", identity)
	}
}

// waitForFailedRecoveryAttempt blocks until the recovery loop has made at
// least one more attempt that failed against the unavailable API. Call it
// only after OnStoppedLeading, when the renew loop's own attempts are over.
func (h *recoveryHarness) waitForFailedRecoveryAttempt() {
	h.t.Helper()
	h.mu.Lock()
	before := h.failedGets
	h.mu.Unlock()
	err := wait.PollUntilContextTimeout(context.Background(), 20*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		return h.failedGets > before, nil
	})
	if err != nil {
		h.t.Fatal("the recovery loop made no attempt against the unavailable API")
	}
}

func (h *recoveryHarness) waitForGateToOpen() {
	h.t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), 20*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return h.write() == nil, nil
	})
	if err != nil {
		h.t.Fatal("the gate did not reopen")
	}
}

func (h *recoveryHarness) waitForRunToReturn() {
	h.t.Helper()
	select {
	case <-h.done:
	case <-time.After(wait.ForeverTestTimeout):
		h.t.Fatal("Run did not return")
	}
}

func (h *recoveryHarness) expectRunning() {
	h.t.Helper()
	select {
	case <-h.done:
		h.t.Fatal("Run returned unexpectedly")
	default:
	}
	if err := h.startedCtx.Err(); err != nil {
		h.t.Fatalf("OnStartedLeading context was cancelled while Run is still running: %v", err)
	}
}

// giveLeaseTo hands the lease to another identity, as a candidate taking over
// would, and makes the API available again. The fake clientset does not check
// resource versions, so renewal updates are made to conflict from here on, as
// they would against a lease whose resource version has moved on.
func (h *recoveryHarness) giveLeaseTo(identity string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	lease := h.lockObj.(*coordinationv1.Lease).DeepCopy()
	now := metav1.NewMicroTime(time.Now())
	lease.Spec.HolderIdentity = &identity
	lease.Spec.AcquireTime = &now
	lease.Spec.RenewTime = &now
	lease.ResourceVersion = "999"
	h.lockObj = lease
	h.failing = false
	h.failRenewUpdates = true
}

// failRenewals makes the API available again but rejects every renewal with
// a conflict, so that recovery keeps failing until its deadline.
func (h *recoveryHarness) failRenewals() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.failing = false
	h.failRenewUpdates = true
}

// deleteLease deletes the lease and makes the API available again.
func (h *recoveryHarness) deleteLease() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.lockObj = nil
	h.failing = false
}

func (h *recoveryHarness) holder() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	lease := h.lockObj.(*coordinationv1.Lease)
	if lease.Spec.HolderIdentity == nil {
		return ""
	}
	return *lease.Spec.HolderIdentity
}

func TestRecoveryReopensGateAfterTransientFailure(t *testing.T) {
	h := newRecoveryHarness(t, &RecoveryConfig{})

	failingSince := time.Now()
	h.setFailing(true)
	h.expectStoppedLeading()
	// Allow for the renew iteration that first observed the failure having
	// started a moment before the failure was injected.
	if elapsed := time.Since(failingSince); elapsed < recoveryTestRenewDeadline-50*time.Millisecond {
		t.Errorf("OnStoppedLeading was called after %v, before RenewDeadline %v", elapsed, recoveryTestRenewDeadline)
	}
	h.expectWriteRejected()
	h.expectRunning()
	h.waitForFailedRecoveryAttempt()

	h.setFailing(false)
	h.waitForGateToOpen()
	h.expectRunning()
	h.expectNoStoppedLeading()
	if h.holder() != recoveryTestIdentity {
		t.Errorf("expected lease holder %q after recovery, got %q", recoveryTestIdentity, h.holder())
	}

	// A second interruption closes the gate and notifies again.
	h.setFailing(true)
	h.expectStoppedLeading()
	h.expectWriteRejected()
	h.waitForFailedRecoveryAttempt()
	h.setFailing(false)
	h.waitForGateToOpen()

	h.cancel()
	h.waitForRunToReturn()
	h.expectStoppedLeading()
	h.expectNoStoppedLeading()
	if h.startedCtx.Err() == nil {
		t.Error("OnStartedLeading context was not cancelled when Run returned")
	}
	h.expectWriteRejected()

	assertEqualEvents(t, []string{
		"Normal LeaderElection baz became leader",
		"Normal LeaderElection baz stopped leading",
		"Normal LeaderElection baz resumed leading",
		"Normal LeaderElection baz stopped leading",
		"Normal LeaderElection baz resumed leading",
		"Normal LeaderElection baz stopped leading",
	}, h.recorder.Events)
}

// TestRecoveryEnds covers every way the election loop ends once the gate has
// closed: the lease changes hands or is deleted, the recovery deadline passes,
// the context is cancelled, or recovery was never configured.
func TestRecoveryEnds(t *testing.T) {
	const other = "other"
	deadline := time.Second
	tests := []struct {
		name     string
		recovery *RecoveryConfig
		// during ends the outage one way or another. Nil lets the deadline pass.
		during func(h *recoveryHarness)
		// newLeader is the identity OnNewLeader must report, if any.
		newLeader string
		// whileLeading applies during with the elector leading normally,
		// instead of during an outage with the gate already closed.
		whileLeading bool
		// minDuration and maxDuration bound how long Run keeps going after
		// the gate closed, or after during for whileLeading rows.
		minDuration, maxDuration time.Duration
		opts                     []func(*LeaderElectionConfig)
		check                    func(t *testing.T, h *recoveryHarness)
	}{
		{
			// The next renewal sees the new holder. The gate must close on that
			// observation, within a retry period, not at the end of the renew
			// deadline, and the lease must not be re-acquired.
			name:         "another holder takes the lease while leading",
			recovery:     &RecoveryConfig{},
			whileLeading: true,
			during:       func(h *recoveryHarness) { h.giveLeaseTo(other) },
			newLeader:    other,
			maxDuration:  recoveryTestRenewDeadline,
			check: func(t *testing.T, h *recoveryHarness) {
				if got := h.holder(); got != other {
					t.Errorf("the lease must stay with the new holder, got %q", got)
				}
			},
		},
		{
			name:         "lease is deleted while leading",
			recovery:     &RecoveryConfig{},
			whileLeading: true,
			during:       (*recoveryHarness).deleteLease,
			maxDuration:  recoveryTestRenewDeadline,
			check: func(t *testing.T, h *recoveryHarness) {
				h.mu.Lock()
				defer h.mu.Unlock()
				if h.creates != 1 {
					t.Errorf("the lease must not be re-created, but it was created %d times", h.creates)
				}
			},
		},
		{
			name:      "another holder takes over",
			recovery:  &RecoveryConfig{},
			during:    func(h *recoveryHarness) { h.giveLeaseTo(other) },
			newLeader: other,
			check: func(t *testing.T, h *recoveryHarness) {
				if got := h.holder(); got != other {
					t.Errorf("recovery must never re-acquire the lease, but the holder is %q", got)
				}
			},
		},
		{
			name:     "lease is deleted",
			recovery: &RecoveryConfig{},
			during:   (*recoveryHarness).deleteLease,
			check: func(t *testing.T, h *recoveryHarness) {
				h.mu.Lock()
				defer h.mu.Unlock()
				if h.creates != 1 {
					t.Errorf("recovery must never re-create the lease, but it was created %d times", h.creates)
				}
			},
		},
		{
			name:        "recovery deadline passes",
			recovery:    &RecoveryConfig{RecoveryDeadline: &deadline},
			minDuration: deadline,
		},
		{
			// The API is reachable but renewals keep failing. When recovery
			// gives up, the lease is still ours and must be released so that
			// a standby can take over right away.
			name:        "recovery deadline passes with release on cancel",
			recovery:    &RecoveryConfig{RecoveryDeadline: &deadline},
			opts:        []func(*LeaderElectionConfig){withReleaseOnCancel},
			during:      (*recoveryHarness).failRenewals,
			minDuration: deadline,
			check: func(t *testing.T, h *recoveryHarness) {
				if got := h.holder(); got != "" {
					t.Errorf("expected the lease to be released after recovery gave up, but the holder is %q", got)
				}
			},
		},
		{
			name:     "context is cancelled",
			recovery: &RecoveryConfig{},
			during:   func(h *recoveryHarness) { h.cancel() },
		},
		{
			name: "recovery not configured",
			check: func(t *testing.T, h *recoveryHarness) {
				assertEqualEvents(t, []string{
					"Normal LeaderElection baz became leader",
					"Normal LeaderElection baz stopped leading",
				}, h.recorder.Events)
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newRecoveryHarness(t, tc.recovery, tc.opts...)
			if !tc.whileLeading {
				h.setFailing(true)
				if tc.recovery != nil {
					// In recovery mode the gate closes first and Run keeps going.
					h.expectStoppedLeading()
					h.expectRunning()
				}
			}
			startedAt := time.Now()
			if tc.during != nil {
				tc.during(h)
			}
			if tc.newLeader != "" {
				h.expectNewLeader(tc.newLeader)
			}
			h.waitForRunToReturn()
			if tc.recovery == nil || tc.whileLeading {
				// Without recovery OnStoppedLeading fires when Run returns; a
				// loss while leading fires it when the gate closes.
				h.expectStoppedLeading()
			}
			if elapsed := time.Since(startedAt); elapsed < tc.minDuration {
				t.Errorf("Run returned %v after the gate closed, before %v", elapsed, tc.minDuration)
			}
			if tc.maxDuration > 0 {
				if elapsed := time.Since(startedAt); elapsed >= tc.maxDuration {
					t.Errorf("Run took %v to end, expected less than %v", elapsed, tc.maxDuration)
				}
			}
			// OnStoppedLeading fires once per close, not again at exit.
			h.expectNoStoppedLeading()
			if h.startedCtx.Err() == nil {
				t.Error("OnStartedLeading context was not cancelled when Run returned")
			}
			h.expectWriteRejected()
			if tc.check != nil {
				tc.check(t, h)
			}
		})
	}
}

// TestRunReportsStoppedLeadingOnEveryRun verifies that a LeaderElector that is
// run more than once calls OnStoppedLeading at the end of each run, with and
// without recovery configured.
func TestRunReportsStoppedLeadingOnEveryRun(t *testing.T) {
	for _, recovery := range []*RecoveryConfig{nil, {}} {
		t.Run(fmt.Sprintf("recovery=%v", recovery != nil), func(t *testing.T) {
			stopped := 0
			le, err := NewLeaderElector(LeaderElectionConfig{
				Lock:          &rl.LeaseLock{LockConfig: rl.ResourceLockConfig{Identity: "baz"}},
				LeaseDuration: 15 * time.Second,
				RenewDeadline: 10 * time.Second,
				RetryPeriod:   2 * time.Second,
				Recovery:      recovery,
				Callbacks: LeaderCallbacks{
					OnStartedLeading: func(context.Context) {},
					OnStoppedLeading: func() { stopped++ },
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			// A cancelled context makes Run return before acquiring, which
			// still reports OnStoppedLeading, as documented.
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			le.Run(ctx)
			le.Run(ctx)
			if stopped != 2 {
				t.Errorf("expected OnStoppedLeading once per Run, got %d calls over 2 runs", stopped)
			}
		})
	}
}

// TestReleaseOnCancel verifies that cancelling the run releases the lease,
// whether the elector is idle or in the middle of a renewal, and that with
// Recovery configured gated writes are already rejected by the time the
// release happens. Without Recovery the gate only closes when Run ends, as
// documented on WriteGate, and renew stays as it was.
func TestReleaseOnCancel(t *testing.T) {
	tests := []struct {
		name     string
		recovery *RecoveryConfig
		// inFlight cancels while a renewal request is blocked, so that the
		// renewal fails with the context error rather than being mistaken
		// for a lost lease.
		inFlight bool
	}{
		{name: "idle, without recovery"},
		{name: "idle, with recovery", recovery: &RecoveryConfig{}},
		{name: "during a renewal, with recovery", recovery: &RecoveryConfig{}, inFlight: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newRecoveryHarness(t, tc.recovery, withReleaseOnCancel)
			if tc.inFlight {
				h.hangNextUpdate.Store(true)
				select {
				case <-h.updateHanging:
				case <-time.After(wait.ForeverTestTimeout):
					t.Fatal("no renewal request was made")
				}
			} else {
				// Let a renewal or two go by, then cancel between renewals.
				time.Sleep(2 * recoveryTestRetryPeriod)
			}
			h.cancel()
			h.waitForRunToReturn()
			h.expectStoppedLeading()
			if got := h.holder(); got != "" {
				t.Fatalf("expected the lease to be released on cancellation, but the holder is %q", got)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if tc.recovery != nil && h.releaseSawGateOpen {
				t.Error("a gated write passed while the lease was being released")
			}
		})
	}
}

func TestRecoveryConfigValidation(t *testing.T) {
	base := func() LeaderElectionConfig {
		return LeaderElectionConfig{
			Lock:          &rl.LeaseLock{LockConfig: rl.ResourceLockConfig{Identity: "baz"}},
			LeaseDuration: 15 * time.Second,
			RenewDeadline: 10 * time.Second,
			RetryPeriod:   2 * time.Second,
			Callbacks: LeaderCallbacks{
				OnStartedLeading: func(context.Context) {},
				OnStoppedLeading: func() {},
			},
		}
	}
	tooShort := 999 * time.Millisecond
	minimum := time.Second
	tests := []struct {
		name        string
		recovery    *RecoveryConfig
		coordinated bool
		wantErr     bool
	}{
		{name: "no recovery", recovery: nil},
		{name: "no recovery with coordinated", recovery: nil, coordinated: true},
		{name: "recovery without deadline", recovery: &RecoveryConfig{}},
		{name: "recovery with minimum deadline", recovery: &RecoveryConfig{RecoveryDeadline: &minimum}},
		{name: "recovery with too short deadline", recovery: &RecoveryConfig{RecoveryDeadline: &tooShort}, wantErr: true},
		{name: "recovery with coordinated", recovery: &RecoveryConfig{}, coordinated: true, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			lec := base()
			lec.Recovery = tc.recovery
			lec.Coordinated = tc.coordinated
			_, err := NewLeaderElector(lec)
			if (err != nil) != tc.wantErr {
				t.Fatalf("NewLeaderElector() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestCheckInRecoveryMode(t *testing.T) {
	const tolerance = 5 * time.Second
	newElector := func(recovery *RecoveryConfig) (*LeaderElector, *testingclock.FakeClock) {
		fakeClock := testingclock.NewFakeClock(time.Now())
		le := &LeaderElector{
			config: LeaderElectionConfig{
				Lock:          &rl.LeaseLock{LockConfig: rl.ResourceLockConfig{Identity: "baz"}},
				LeaseDuration: 15 * time.Second,
				Name:          "test",
				Recovery:      recovery,
			},
			clock:     fakeClock,
			metrics:   noMetrics{},
			writeGate: &writeGate{},
		}
		le.recordAttempt()
		le.setObservedRecord(&rl.LeaderElectionRecord{HolderIdentity: "baz", LeaseDurationSeconds: 15})
		return le, fakeClock
	}

	t.Run("without recovery an unobserved lease is unhealthy", func(t *testing.T) {
		le, fakeClock := newElector(nil)
		fakeClock.Step(le.config.LeaseDuration + tolerance + time.Second)
		le.recordAttempt()
		if err := le.Check(tolerance); err == nil {
			t.Error("expected Check to fail for an unobserved lease without recovery")
		}
	})

	t.Run("with recovery an inactive holder is healthy while attempts continue", func(t *testing.T) {
		le, fakeClock := newElector(&RecoveryConfig{})
		fakeClock.Step(le.config.LeaseDuration + tolerance + time.Second)
		le.recordAttempt()
		if err := le.Check(tolerance); err != nil {
			t.Errorf("expected Check to pass while the election loop keeps attempting, got %v", err)
		}
	})

	t.Run("with recovery an inactive holder is unhealthy once attempts stop", func(t *testing.T) {
		le, fakeClock := newElector(&RecoveryConfig{})
		fakeClock.Step(le.config.LeaseDuration + tolerance + time.Second)
		if err := le.Check(tolerance); err == nil {
			t.Error("expected Check to fail once the election loop stopped attempting")
		}
	})

	t.Run("not the leader is healthy", func(t *testing.T) {
		le, fakeClock := newElector(&RecoveryConfig{})
		le.setObservedRecord(&rl.LeaderElectionRecord{HolderIdentity: "other", LeaseDurationSeconds: 15})
		fakeClock.Step(le.config.LeaseDuration + tolerance + time.Second)
		if err := le.Check(tolerance); err != nil {
			t.Errorf("expected Check to pass for a non-leader, got %v", err)
		}
	})
}
