/*
Copyright 2015 The Kubernetes Authors.

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

// Package leaderelection implements leader election of a set of endpoints.
// It uses an annotation in the endpoints object to store the record of the
// election state. This implementation does not guarantee that only one
// client is acting as a leader (a.k.a. fencing).
//
// A client only acts on timestamps captured locally to infer the state of the
// leader election. The client does not consider timestamps in the leader
// election record to be accurate because these timestamps may not have been
// produced by a local clock. The implemention does not depend on their
// accuracy and only uses their change to indicate that another client has
// renewed the leader lease. Thus the implementation is tolerant to arbitrary
// clock skew, but is not tolerant to arbitrary clock skew rate.
//
// However the level of tolerance to skew rate can be configured by setting
// RenewDeadline and LeaseDuration appropriately. The tolerance expressed as a
// maximum tolerated ratio of time passed on the fastest node to time passed on
// the slowest node can be approximately achieved with a configuration that sets
// the same ratio of LeaseDuration to RenewDeadline. For example if a user wanted
// to tolerate some nodes progressing forward in time twice as fast as other nodes,
// the user could set LeaseDuration to 60 seconds and RenewDeadline to 30 seconds.
//
// While not required, some method of clock synchronization between nodes in the
// cluster is highly recommended. It's important to keep in mind when configuring
// this client that the tolerance to skew rate varies inversely to master
// availability.
//
// Larger clusters often have a more lenient SLA for API latency. This should be
// taken into account when configuring the client. The rate of leader transitions
// should be monitored and RetryPeriod and LeaseDuration should be increased
// until the rate is stable and acceptably low. It's important to keep in mind
// when configuring this client that the tolerance to API latency varies inversely
// to master availability.
//
// DISCLAIMER: this is an alpha API. This library will likely change significantly
// or even be removed entirely in subsequent releases. Depend on this API at
// your own risk.
package leaderelection

import (
	"bytes"
	"context"
	stderrors "errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	rl "k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	JitterFactor = 1.2
)

// errLeaseLost ends a renewal poll as soon as the lease is observed to have
// changed hands or been deleted, so that the write gate closes right away.
var errLeaseLost = stderrors.New("lease changed hands or was deleted")

// NewLeaderElector creates a LeaderElector from a LeaderElectionConfig
func NewLeaderElector(lec LeaderElectionConfig) (*LeaderElector, error) {
	if lec.LeaseDuration <= lec.RenewDeadline {
		return nil, fmt.Errorf("leaseDuration must be greater than renewDeadline")
	}
	if lec.RenewDeadline <= time.Duration(JitterFactor*float64(lec.RetryPeriod)) {
		return nil, fmt.Errorf("renewDeadline must be greater than retryPeriod*JitterFactor")
	}
	if lec.LeaseDuration < 1 {
		return nil, fmt.Errorf("leaseDuration must be greater than zero")
	}
	if lec.RenewDeadline < 1 {
		return nil, fmt.Errorf("renewDeadline must be greater than zero")
	}
	if lec.RetryPeriod < 1 {
		return nil, fmt.Errorf("retryPeriod must be greater than zero")
	}
	if lec.Recovery != nil && lec.Recovery.RecoveryDeadline != nil && *lec.Recovery.RecoveryDeadline < time.Second {
		return nil, fmt.Errorf("recoveryDeadline must be at least one second")
	}
	if lec.Recovery != nil && lec.Coordinated {
		// With coordinated leader election the apiserver's election controller
		// decides who holds the lease and may reassign it during an outage, so
		// the renew-only recovery below does not apply. KEP-6361 leaves
		// extending recovery to coordinated election to post-alpha.
		return nil, fmt.Errorf("recovery is not supported with coordinated leader election")
	}
	if lec.Callbacks.OnStartedLeading == nil {
		return nil, fmt.Errorf("OnStartedLeading callback must not be nil")
	}
	if lec.Callbacks.OnStoppedLeading == nil {
		return nil, fmt.Errorf("OnStoppedLeading callback must not be nil")
	}

	if lec.Lock == nil {
		return nil, fmt.Errorf("Lock must not be nil.")
	}
	id := lec.Lock.Identity()
	if id == "" {
		return nil, fmt.Errorf("Lock identity is empty")
	}

	le := LeaderElector{
		config:    lec,
		clock:     clock.RealClock{},
		metrics:   globalMetricsFactory.newLeaderMetrics(),
		writeGate: &writeGate{},
	}
	le.metrics.leaderOff(le.config.Name)
	return &le, nil
}

type LeaderElectionConfig struct {
	// Lock is the resource that will be used for locking
	Lock rl.Interface

	// LeaseDuration is the duration that non-leader candidates will
	// wait to force acquire leadership. This is measured against time of
	// last observed ack.
	//
	// A client needs to wait a full LeaseDuration without observing a change to
	// the record before it can attempt to take over. When all clients are
	// shutdown and a new set of clients are started with different names against
	// the same leader record, they must wait the full LeaseDuration before
	// attempting to acquire the lease. Thus LeaseDuration should be as short as
	// possible (within your tolerance for clock skew rate) to avoid a possible
	// long waits in the scenario.
	//
	// Core clients default this value to 15 seconds.
	LeaseDuration time.Duration
	// RenewDeadline is the duration that the acting master will retry
	// refreshing leadership before giving up.
	//
	// Core clients default this value to 10 seconds.
	RenewDeadline time.Duration
	// RetryPeriod is the duration the LeaderElector clients should wait
	// between tries of actions.
	//
	// Core clients default this value to 2 seconds.
	RetryPeriod time.Duration

	// Callbacks are callbacks that are triggered during certain lifecycle
	// events of the LeaderElector
	Callbacks LeaderCallbacks

	// WatchDog is the associated health checker
	// WatchDog may be null if it's not needed/configured.
	WatchDog *HealthzAdaptor

	// ReleaseOnCancel should be set true if the lock should be released
	// when the run context is cancelled. If you set this to true, you must
	// ensure all code guarded by this lease has successfully completed
	// prior to cancelling the context, or you may have two processes
	// simultaneously acting on the critical path.
	ReleaseOnCancel bool

	// Name is the name of the resource lock for debugging
	Name string

	// Coordinated will use the Coordinated Leader Election feature
	// WARNING: Coordinated leader election is ALPHA.
	Coordinated bool

	// Recovery configures how a leader that has failed to renew its lease,
	// recovers once it can renew again. This often happens during temporary
	// connection loss to the API server.
	//
	// Without Recovery, Run returns once renewal has failed for RenewDeadline
	// and the caller is expected to exit. With Recovery, OnStoppedLeading is
	// still called at that point, but Run keeps going. The elector keeps
	// trying to renew the lease it held, at RetryPeriod, and resumes leading
	// if it succeeds. While it is not leading, write requests made through
	// clients wrapped with WriteGate are rejected, which is what keeps a
	// recovering leader from acting on a lease it may have lost. See
	// RecoveryConfig for when Run returns instead.
	//
	// Recovery is not supported together with Coordinated.
	// WARNING: Leader election recovery is ALPHA.
	Recovery *RecoveryConfig
}

// RecoveryConfig configures how a leader that has failed to renew its lease
// recovers, see LeaderElectionConfig.Recovery.
type RecoveryConfig struct {
	// RecoveryDeadline limits how long to keep trying to renew the lease
	// after failing to renew it within RenewDeadline. When the limit is
	// reached, Run returns and callers exit.
	//
	// Regardless of the deadline, if another candidate claims the lease, or the
	// lease is deleted, Run returns and callers exit. Recovery only ever renews
	// the lease this elector still holds, it never re-acquires it.
	//
	// No limit if unset. The value must be at least one second if set.
	RecoveryDeadline *time.Duration
}

// LeaderCallbacks are callbacks that are triggered during certain
// lifecycle events of the LeaderElector. These are invoked asynchronously.
//
// possible future callbacks:
//   - OnChallenge()
type LeaderCallbacks struct {
	// OnStartedLeading is called when a LeaderElector client starts leading.
	// It is called at most once per Run. Its context is cancelled when Run
	// returns, not when a recovering elector temporarily stops leading, so
	// work started from this callback keeps running throughout a recovery.
	OnStartedLeading func(context.Context)
	// OnStoppedLeading is called when a LeaderElector client stops leading.
	// Users should not assume that OnStoppedLeading is only called after OnStartedLeading.
	// see: https://github.com/kubernetes/kubernetes/pull/127675#discussion_r1780059887
	//
	// Without LeaderElectionConfig.Recovery it is called exactly once, when
	// Run returns, even if the elector never led. With Recovery it is called
	// each time the elector stops leading, at most once per such event, and
	// is not called again when Run returns if it already fired for the loss
	// that ended the run. Run returning is the signal that the elector has
	// given up, in both modes.
	OnStoppedLeading func()
	// OnNewLeader is called when the client observes a leader that is
	// not the previously observed leader. This includes the first observed
	// leader when the client starts.
	OnNewLeader func(identity string)
}

// LeaderElector is a leader election client.
type LeaderElector struct {
	config LeaderElectionConfig
	// internal bookkeeping
	observedRecord    rl.LeaderElectionRecord
	observedRawRecord []byte
	observedTime      time.Time
	// used to implement OnNewLeader(), may lag slightly from the
	// value observedRecord.HolderIdentity if the transition has
	// not yet been reported.
	reportedLeader string

	// clock is wrapper around time to allow for less flaky testing
	clock clock.Clock

	// used to lock the observedRecord and the observedTime
	observedRecordLock sync.RWMutex

	// lastAttemptTime is when the elector last started an attempt to acquire
	// or renew the lease. Protected by observedRecordLock.
	lastAttemptTime time.Time

	// writeGate is open while this elector is leading and closed otherwise,
	// including before it first acquires the lease. While it is closed,
	// transports wrapped with WriteGate reject non-safe requests and cancel
	// writes that are in flight, see writegate.go.
	writeGate *writeGate

	// stoppedLeadingNotified records that OnStoppedLeading has been called
	// for the current loss of leadership, so that Run does not call it again
	// on exit. Only accessed from the goroutine running Run.
	stoppedLeadingNotified bool

	metrics leaderMetricsAdapter
}

// Run starts the leader election loop. Run will not return
// before leader election loop is stopped by ctx or it has
// stopped holding the leader lease
func (le *LeaderElector) Run(ctx context.Context) {
	defer runtime.HandleCrashWithContext(ctx)
	// A LeaderElector may be run more than once; each run reports its own loss.
	le.stoppedLeadingNotified = false
	defer le.notifyStoppedLeading()

	if !le.acquire(ctx) {
		return // ctx signalled done
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	// Whatever ends the run, writes through gated clients stop with it.
	defer le.writeGate.close()
	go le.config.Callbacks.OnStartedLeading(ctx)
	if le.config.Recovery == nil {
		le.renew(ctx)
		return
	}
	le.renewWithRecovery(ctx)
}

// RunOrDie starts a client with the provided config or panics if the config
// fails to validate. RunOrDie blocks until leader election loop is
// stopped by ctx or it has stopped holding the leader lease
func RunOrDie(ctx context.Context, lec LeaderElectionConfig) {
	le, err := NewLeaderElector(lec)
	if err != nil {
		panic(err)
	}
	if lec.WatchDog != nil {
		lec.WatchDog.SetLeaderElection(le)
	}
	le.Run(ctx)
}

// GetLeader returns the identity of the last observed leader or returns the empty string if
// no leader has yet been observed.
// This function is for informational purposes. (e.g. monitoring, logs, etc.)
func (le *LeaderElector) GetLeader() string {
	return le.getObservedRecord().HolderIdentity
}

// IsLeader returns true if the last observed leader was this client else returns false.
func (le *LeaderElector) IsLeader() bool {
	return le.getObservedRecord().HolderIdentity == le.config.Lock.Identity()
}

// WriteGate returns a transport wrapper that blocks write requests whenever
// this elector is not leading.
//
// Only the safe methods of RFC 9110 (GET, HEAD, OPTIONS, TRACE) pass while the
// elector is not leading. Every other method is rejected with
// ErrWriteGateClosed, and write requests that are in flight when the elector
// stops leading are cancelled. The gate opens when the elector starts leading
// and closes when it stops leading, alongside OnStartedLeading and
// OnStoppedLeading.
//
// The gate is meant for electors with Recovery configured, which keep running
// after losing the lease: there it closes as soon as the loss is observed and
// before any release. Without Recovery, Run returns on loss as it always has
// and the gate only closes then, which is when callers are expected to exit.
//
// The client used for the Lease itself must not be gated, or the elector
// could not renew it while recovering.
func (le *LeaderElector) WriteGate() transport.WrapperFunc {
	return le.writeGate.wrapper()
}

// acquire loops calling tryAcquireOrRenew and returns true immediately when tryAcquireOrRenew succeeds.
// Returns false if ctx signals done.
func (le *LeaderElector) acquire(ctx context.Context) bool {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	succeeded := false
	desc := le.config.Lock.Describe()
	logger := klog.FromContext(ctx)
	logger.Info("Attempting to acquire leader lease...", "lock", desc)
	wait.JitterUntilWithContext(ctx, func(ctx context.Context) {
		if !le.config.Coordinated {
			succeeded = le.tryAcquireOrRenew(ctx)
		} else {
			succeeded = le.tryCoordinatedRenew(ctx)
		}
		le.maybeReportTransition()
		if !succeeded {
			logger.V(4).Info("Failed to acquire lease", "lock", desc)
			return
		}
		le.config.Lock.RecordEvent("became leader")
		le.metrics.leaderOn(le.config.Name)
		le.writeGate.open()
		logger.Info("Successfully acquired lease", "lock", desc)
		cancel()
	}, le.config.RetryPeriod, JitterFactor, true)
	return succeeded
}

// renew loops calling tryAcquireOrRenew and returns immediately when tryAcquireOrRenew fails or ctx signals done.
func (le *LeaderElector) renew(ctx context.Context) {
	defer le.config.Lock.RecordEvent("stopped leading")
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(ctx)
	wait.UntilWithContext(ctx, func(ctx context.Context) {
		err := wait.PollUntilContextTimeout(ctx, le.config.RetryPeriod, le.config.RenewDeadline, true, func(ctx context.Context) (done bool, err error) {
			// PollUntilContextTimeout invokes condition even when the context is canceled when immediate=true.
			// Short-circuit this to prevent unnecessary processing and error log messages.
			if err := ctx.Err(); err != nil {
				return false, err
			}
			if !le.config.Coordinated {
				return le.tryAcquireOrRenew(ctx), nil
			} else {
				return le.tryCoordinatedRenew(ctx), nil
			}
		})
		le.maybeReportTransition()
		desc := le.config.Lock.Describe()
		if err == nil {
			logger.V(5).Info("Successfully renewed lease", "lock", desc)
			return
		}
		le.metrics.leaderOff(le.config.Name)
		logger.Info("Failed to renew lease", "lock", desc, "err", err)
		cancel()
	}, le.config.RetryPeriod)

	// if we hold the lease, give it up
	if le.config.ReleaseOnCancel {
		le.release(logger)
	}
}

// renewWithRecovery is renew for an elector with Recovery configured. Failing
// to renew within RenewDeadline closes the write gate and calls
// OnStoppedLeading as renew would end the run, but the loop only ends if the
// lease cannot be recovered, see LeaderElectionConfig.Recovery.
func (le *LeaderElector) renewWithRecovery(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(ctx)
	desc := le.config.Lock.Describe()
	leading := true
	wait.UntilWithContext(ctx, func(ctx context.Context) {
		err := wait.PollUntilContextTimeout(ctx, le.config.RetryPeriod, le.config.RenewDeadline, true, func(ctx context.Context) (done bool, err error) {
			// PollUntilContextTimeout invokes condition even when the context is canceled when immediate=true.
			// Short-circuit this to prevent unnecessary processing and error log messages.
			if err := ctx.Err(); err != nil {
				return false, err
			}
			// Renew only; this loop never acquires, and it stops as soon as
			// the lease is observed to be someone else's or gone, so that the
			// gate closes on the first observation rather than at the end of
			// the renew deadline.
			renewed, lost := le.tryRecover(ctx)
			if lost {
				return false, errLeaseLost
			}
			return renewed, nil
		})
		le.maybeReportTransition()
		if err == nil {
			logger.V(5).Info("Successfully renewed lease", "lock", desc)
			return
		}
		logger.Info("Failed to renew lease", "lock", desc, "err", err)
		leading = false
		le.writeGate.close()
		le.metrics.leaderOff(le.config.Name)
		le.config.Lock.RecordEvent("stopped leading")
		if ctx.Err() != nil {
			cancel()
			return
		}
		// Writes are blocked by the gate now, which is what OnStoppedLeading
		// signals. Keep trying to renew the lease we held.
		le.notifyStoppedLeading()
		if !le.recoverLease(ctx) {
			cancel()
			return
		}
		leading = true
		le.stoppedLeadingNotified = false
		le.writeGate.open()
		le.metrics.leaderOn(le.config.Name)
		le.config.Lock.RecordEvent("resumed leading")
		logger.Info("Successfully recovered lease", "lock", desc)
	}, le.config.RetryPeriod)
	if leading {
		// The loop ended on context cancellation between two renewals.
		le.config.Lock.RecordEvent("stopped leading")
	}

	// Give the lease up whether the loop ended on cancellation or because
	// recovery failed: in the latter case the lease is usually ours and merely
	// expired, and releasing it lets a standby take over right away. If it
	// has changed hands in the meantime, tryRelease leaves it alone.
	if le.config.ReleaseOnCancel {
		// Stop gated writes before the lease is handed back, the deferred
		// close in Run is only a backstop.
		le.writeGate.close()
		le.release(logger)
	}
}

// notifyStoppedLeading calls OnStoppedLeading unless it has already been called
// since the elector last started leading.
func (le *LeaderElector) notifyStoppedLeading() {
	if le.stoppedLeadingNotified {
		return
	}
	le.stoppedLeadingNotified = true
	le.config.Callbacks.OnStoppedLeading()
}

// recoverLease keeps attempting to renew the lease this elector held, at
// RetryPeriod, until one of: the renewal succeeds (returns true), the lease is
// observed to have changed hands or been deleted, the recovery deadline
// passes, or ctx is cancelled (all return false).
func (le *LeaderElector) recoverLease(ctx context.Context) bool {
	logger := klog.FromContext(ctx)
	desc := le.config.Lock.Describe()
	var cancel context.CancelFunc
	if deadline := le.config.Recovery.RecoveryDeadline; deadline != nil {
		ctx, cancel = context.WithTimeout(ctx, *deadline)
		logger.Info("Attempting to recover leader lease...", "lock", desc, "recoveryDeadline", *deadline)
	} else {
		ctx, cancel = context.WithCancel(ctx)
		logger.Info("Attempting to recover leader lease...", "lock", desc)
	}
	defer cancel()
	recovered := false
	wait.UntilWithContext(ctx, func(ctx context.Context) {
		// Bound each attempt the way the renew loop bounds its attempts, so a
		// lock client without its own timeout cannot stall recovery.
		attemptCtx, cancelAttempt := context.WithTimeout(ctx, le.config.RenewDeadline)
		renewed, lost := le.tryRecover(attemptCtx)
		cancelAttempt()
		le.maybeReportTransition()
		if renewed {
			recovered = true
			cancel()
			return
		}
		if lost {
			logger.Info("Giving up on recovering leader lease", "lock", desc)
			cancel()
			return
		}
		logger.V(4).Info("Failed attempt to recover lease", "lock", desc)
	}, le.config.RetryPeriod)
	if !recovered && ctx.Err() != nil {
		logger.Info("Stopped attempting to recover leader lease", "lock", desc, "err", ctx.Err())
	}
	return recovered
}

// tryRecover attempts to renew the lease if, and only if, this elector is still
// its holder. It never creates or re-acquires the lease. It returns
// renewed=true if the lease was renewed, and lost=true if the lease was
// observed to have changed hands or been deleted, in which case the elector
// must stop. It is the renewal step for electors with Recovery configured,
// both while leading and while recovering.
func (le *LeaderElector) tryRecover(ctx context.Context) (renewed, lost bool) {
	logger := klog.FromContext(ctx)
	le.recordAttempt()
	now := metav1.NewTime(le.clock.Now())

	// 1. fast path: optimistically update the record observed last time while
	// it is still valid locally. The update is conditional on that record's
	// resource version, so a lease that changed hands fails here and is
	// examined by the slow path.
	if le.IsLeader() && le.isLeaseValid(now.Time) {
		oldObservedRecord := le.getObservedRecord()
		leaderElectionRecord := rl.LeaderElectionRecord{
			HolderIdentity:       le.config.Lock.Identity(),
			LeaseDurationSeconds: int(le.config.LeaseDuration / time.Second),
			RenewTime:            now,
			AcquireTime:          oldObservedRecord.AcquireTime,
			LeaderTransitions:    oldObservedRecord.LeaderTransitions,
		}
		if err := le.config.Lock.Update(ctx, leaderElectionRecord); err == nil {
			le.setObservedRecord(&leaderElectionRecord)
			return true, false
		} else {
			logger.V(2).Info("Failed to update lease optimistically, falling back to slow path", "lock", le.config.Lock.Describe(), "err", err)
		}
	}

	// 2. obtain the ElectionRecord. Unlike tryAcquireOrRenew, never create it.
	oldLeaderElectionRecord, oldLeaderElectionRawRecord, err := le.config.Lock.Get(ctx)
	if err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "Error retrieving lease lock", "lock", le.config.Lock.Describe())
			return false, false
		}
		logger.Info("Lease lock was deleted", "lock", le.config.Lock.Describe())
		return false, true
	}

	// 3. Record obtained, check the Identity
	if !bytes.Equal(le.observedRawRecord, oldLeaderElectionRawRecord) {
		le.setObservedRecord(oldLeaderElectionRecord)

		le.observedRawRecord = oldLeaderElectionRawRecord
	}
	// Only ever renew a lease this elector still holds. If it changed hands
	// while we could not observe it, give up even if the other holder has
	// since released it or expired.
	if !le.IsLeader() {
		logger.Info("Lease lock changed holder", "lock", le.config.Lock.Describe(), "holder", oldLeaderElectionRecord.HolderIdentity)
		return false, true
	}

	// 4. We are still the holder, renew. The update is conditional on the
	// resource version observed above.
	leaderElectionRecord := rl.LeaderElectionRecord{
		HolderIdentity:       le.config.Lock.Identity(),
		LeaseDurationSeconds: int(le.config.LeaseDuration / time.Second),
		RenewTime:            now,
		AcquireTime:          oldLeaderElectionRecord.AcquireTime,
		LeaderTransitions:    oldLeaderElectionRecord.LeaderTransitions,
	}
	if err = le.config.Lock.Update(ctx, leaderElectionRecord); err != nil {
		logger.Error(err, "Failed to update lease", "lock", le.config.Lock.Describe())
		return false, false
	}

	le.setObservedRecord(&leaderElectionRecord)
	return true, false
}

// release attempts to release the leader lease if we have acquired it.
// It retries on conflict, which may occur if the context cancellation
// races with an inflight renew() operation. The client will see a ctx
// cancellation error while the apiserver completes the update and bumps
// the resource version.
func (le *LeaderElector) release(logger klog.Logger) bool {
	ctx := klog.NewContext(context.Background(), logger)
	timeoutCtx, timeoutCancel := context.WithTimeout(ctx, le.config.RenewDeadline)
	defer timeoutCancel()
	return le.tryRelease(timeoutCtx)
}

func (le *LeaderElector) tryRelease(ctx context.Context) bool {
	logger := klog.FromContext(ctx)
	oldLeaderElectionRecord, _, err := le.config.Lock.Get(ctx)
	if err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "error retrieving resource lock", "lock", le.config.Lock.Describe())
			return false
		}
		logger.Info("lease lock not found", "lock", le.config.Lock.Describe())
		return false
	}

	// Check the record just fetched as well as the cached one: after an
	// outage the cache can still name this elector as the holder while the
	// lease has changed hands, and releasing would clear the new holder.
	if !le.IsLeader() || oldLeaderElectionRecord.HolderIdentity != le.config.Lock.Identity() {
		return true
	}
	now := metav1.NewTime(le.clock.Now())
	leaderElectionRecord := rl.LeaderElectionRecord{
		LeaderTransitions:    oldLeaderElectionRecord.LeaderTransitions,
		LeaseDurationSeconds: 1,
		RenewTime:            now,
		AcquireTime:          now,
	}
	if err := le.config.Lock.Update(ctx, leaderElectionRecord); err != nil {
		if errors.IsConflict(err) {
			logger.V(4).Info("Conflict when releasing lease, retrying", "lock", le.config.Lock.Describe())
			return le.tryRelease(ctx)
		}
		logger.Error(err, "Failed to release lease", "lock", le.config.Lock.Describe())
		return false
	}

	le.setObservedRecord(&leaderElectionRecord)
	return true
}

// tryCoordinatedRenew checks if it acquired a lease and tries to renew the
// lease if it has already been acquired. Returns true on success else returns
// false.
func (le *LeaderElector) tryCoordinatedRenew(ctx context.Context) bool {
	logger := klog.FromContext(ctx)
	now := metav1.NewTime(le.clock.Now())
	leaderElectionRecord := rl.LeaderElectionRecord{
		HolderIdentity:       le.config.Lock.Identity(),
		LeaseDurationSeconds: int(le.config.LeaseDuration / time.Second),
		RenewTime:            now,
		AcquireTime:          now,
	}

	// 1. obtain the electionRecord
	oldLeaderElectionRecord, oldLeaderElectionRawRecord, err := le.config.Lock.Get(ctx)
	if err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "Error retrieving lease lock", "lock", le.config.Lock.Describe())
			return false
		}
		logger.Info("Lease lock not found", "lock", le.config.Lock.Describe(), "err", err)
		return false
	}

	// 2. Record obtained, check the Identity & Time
	if !bytes.Equal(le.observedRawRecord, oldLeaderElectionRawRecord) {
		le.setObservedRecord(oldLeaderElectionRecord)

		le.observedRawRecord = oldLeaderElectionRawRecord
	}

	le.observedRecordLock.RLock()
	obsTime := le.observedTime
	le.observedRecordLock.RUnlock()

	hasExpired := obsTime.Add(time.Second * time.Duration(oldLeaderElectionRecord.LeaseDurationSeconds)).Before(now.Time)
	if hasExpired {
		logger.Info("Lease has expired", "lock", le.config.Lock.Describe())
		return false
	}

	if !le.IsLeader() {
		logger.V(6).Info("Lease is held and has not yet expired", "lock", le.config.Lock.Describe(), "holder", oldLeaderElectionRecord.HolderIdentity)
		return false
	}

	// 2b. If the lease has been marked as "end of term", don't renew it
	if le.IsLeader() && oldLeaderElectionRecord.PreferredHolder != "" {
		logger.V(4).Info("Lease is marked as 'end of term'", "lock", le.config.Lock.Describe())
		// TODO: Instead of letting lease expire, the holder may deleted it directly
		// This will not be compatible with all controllers, so it needs to be opt-in behavior.
		// We must ensure all code guarded by this lease has successfully completed
		// prior to releasing or there may be two processes
		// simultaneously acting on the critical path.
		// Usually once this returns false, the process is terminated..
		// xref: OnStoppedLeading
		return false
	}

	// 3. We're going to try to update. The leaderElectionRecord is set to it's default
	// here. Let's correct it before updating.
	if le.IsLeader() {
		leaderElectionRecord.AcquireTime = oldLeaderElectionRecord.AcquireTime
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions
		leaderElectionRecord.Strategy = oldLeaderElectionRecord.Strategy
		le.metrics.slowpathExercised(le.config.Name)
	} else {
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions + 1
	}

	// update the lock itself
	if err = le.config.Lock.Update(ctx, leaderElectionRecord); err != nil {
		logger.Error(err, "Failed to update lock", "lock", le.config.Lock.Describe())
		return false
	}

	le.setObservedRecord(&leaderElectionRecord)
	return true
}

// tryAcquireOrRenew tries to acquire a leader lease if it is not already acquired,
// else it tries to renew the lease if it has already been acquired. Returns true
// on success else returns false.
func (le *LeaderElector) tryAcquireOrRenew(ctx context.Context) bool {
	logger := klog.FromContext(ctx)
	le.recordAttempt()
	now := metav1.NewTime(le.clock.Now())
	leaderElectionRecord := rl.LeaderElectionRecord{
		HolderIdentity:       le.config.Lock.Identity(),
		LeaseDurationSeconds: int(le.config.LeaseDuration / time.Second),
		RenewTime:            now,
		AcquireTime:          now,
	}

	// 1. fast path for the leader to update optimistically assuming that the record observed
	// last time is the current version.
	if le.IsLeader() && le.isLeaseValid(now.Time) {
		oldObservedRecord := le.getObservedRecord()
		leaderElectionRecord.AcquireTime = oldObservedRecord.AcquireTime
		leaderElectionRecord.LeaderTransitions = oldObservedRecord.LeaderTransitions

		err := le.config.Lock.Update(ctx, leaderElectionRecord)
		if err == nil {
			le.setObservedRecord(&leaderElectionRecord)
			return true
		}
		logger.V(2).Info("Failed to update lease optimistically, falling back to slow path", "lock", le.config.Lock.Describe(), "err", err)
	}

	// 2. obtain or create the ElectionRecord
	oldLeaderElectionRecord, oldLeaderElectionRawRecord, err := le.config.Lock.Get(ctx)
	if err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "Error retrieving lease lock", "lock", le.config.Lock.Describe())
			return false
		}
		if err = le.config.Lock.Create(ctx, leaderElectionRecord); err != nil {
			logger.Error(err, "Error initially creating lease lock", "lock", le.config.Lock.Describe())
			return false
		}

		le.setObservedRecord(&leaderElectionRecord)

		return true
	}

	// 3. Record obtained, check the Identity & Time
	if !bytes.Equal(le.observedRawRecord, oldLeaderElectionRawRecord) {
		le.setObservedRecord(oldLeaderElectionRecord)

		le.observedRawRecord = oldLeaderElectionRawRecord
	}
	if len(oldLeaderElectionRecord.HolderIdentity) > 0 && le.isLeaseValid(now.Time) && !le.IsLeader() {
		logger.V(4).Info("Lease is held by and has not yet expired", "lock", le.config.Lock.Describe(), "holder", oldLeaderElectionRecord.HolderIdentity)
		return false
	}

	// 4. We're going to try to update. The leaderElectionRecord is set to it's default
	// here. Let's correct it before updating.
	if le.IsLeader() {
		leaderElectionRecord.AcquireTime = oldLeaderElectionRecord.AcquireTime
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions
		le.metrics.slowpathExercised(le.config.Name)
	} else {
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions + 1
	}

	// update the lock itself
	if err = le.config.Lock.Update(ctx, leaderElectionRecord); err != nil {
		logger.Error(err, "Failed to update lease", "lock", le.config.Lock.Describe())
		return false
	}

	le.setObservedRecord(&leaderElectionRecord)
	return true
}

func (le *LeaderElector) maybeReportTransition() {
	if le.observedRecord.HolderIdentity == le.reportedLeader {
		return
	}
	le.reportedLeader = le.observedRecord.HolderIdentity
	if le.config.Callbacks.OnNewLeader != nil {
		go le.config.Callbacks.OnNewLeader(le.reportedLeader)
	}
}

// Check will determine if the current lease is expired by more than timeout.
func (le *LeaderElector) Check(maxTolerableExpiredLease time.Duration) error {
	if !le.IsLeader() {
		// Currently not concerned with the case that we are hot standby
		return nil
	}
	// If we are more than timeout seconds after the lease duration that is past the timeout
	// on the lease renew. Time to start reporting ourselves as unhealthy. We should have
	// died but conditions like deadlock can prevent this. (See #70819)
	le.observedRecordLock.RLock()
	lastObservation := le.observedTime
	lastAttempt := le.lastAttemptTime
	leaseDuration := le.config.LeaseDuration
	le.observedRecordLock.RUnlock()

	if le.config.Recovery != nil {
		// In recovery mode the elector deliberately stays alive while it cannot
		// observe the lease, so only report failure once the election loop has
		// stopped making attempts.
		if le.clock.Since(lastAttempt) > leaseDuration+maxTolerableExpiredLease {
			return fmt.Errorf("stopped attempting to renew leadership on lease %s", le.config.Name)
		}
		return nil
	}

	if le.clock.Since(lastObservation) > leaseDuration+maxTolerableExpiredLease {
		return fmt.Errorf("failed election to renew leadership on lease %s", le.config.Name)
	}

	return nil
}

// recordAttempt notes that the elector is about to attempt to acquire or renew
// the lease. Protect critical sections with lock.
func (le *LeaderElector) recordAttempt() {
	le.observedRecordLock.Lock()
	defer le.observedRecordLock.Unlock()

	le.lastAttemptTime = le.clock.Now()
}

func (le *LeaderElector) isLeaseValid(now time.Time) bool {
	// Lock to safely read both the time and the record
	le.observedRecordLock.RLock()
	defer le.observedRecordLock.RUnlock()

	return le.observedTime.Add(time.Second * time.Duration(le.observedRecord.LeaseDurationSeconds)).After(now)
}

// setObservedRecord will set a new observedRecord and update observedTime to the current time.
// Protect critical sections with lock.
func (le *LeaderElector) setObservedRecord(observedRecord *rl.LeaderElectionRecord) {
	le.observedRecordLock.Lock()
	defer le.observedRecordLock.Unlock()

	le.observedRecord = *observedRecord
	le.observedTime = le.clock.Now()
}

// getObservedRecord returns observersRecord.
// Protect critical sections with lock.
func (le *LeaderElector) getObservedRecord() rl.LeaderElectionRecord {
	le.observedRecordLock.RLock()
	defer le.observedRecordLock.RUnlock()

	return le.observedRecord
}
