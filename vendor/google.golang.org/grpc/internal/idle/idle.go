/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package idle contains a component for managing idleness (entering and exiting)
// based on RPC activity.
package idle

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/grpclog"
)

// For overriding in unit tests.
var timeAfterFunc = func(d time.Duration, f func()) *time.Timer {
	return time.AfterFunc(d, f)
}

// Enforcer is the functionality provided by grpc.ClientConn to enter
// and exit from idle mode.
type Enforcer interface {
	ExitIdleMode() error
	EnterIdleMode() error
}

// Manager defines the functionality required to track RPC activity on a
// channel.
type Manager interface {
	OnCallBegin() error
	OnCallEnd()
	Close()
}

type noopManager struct{}

func (noopManager) OnCallBegin() error { return nil }
func (noopManager) OnCallEnd()         {}
func (noopManager) Close()             {}

// manager implements the Manager interface. It uses atomic operations to
// synchronize access to shared state and a mutex to guarantee mutual exclusion
// in a critical section.
type manager struct {
	// State accessed atomically.
	lastCallEndTime           int64 // Unix timestamp in nanos; time when the most recent RPC completed.
	activeCallsCount          int32 // Count of active RPCs; -math.MaxInt32 means channel is idle or is trying to get there.
	activeSinceLastTimerCheck int32 // Boolean; True if there was an RPC since the last timer callback.
	closed                    int32 // Boolean; True when the manager is closed.

	// Can be accessed without atomics or mutex since these are set at creation
	// time and read-only after that.
	enforcer Enforcer // Functionality provided by grpc.ClientConn.
	timeout  int64    // Idle timeout duration nanos stored as an int64.
	logger   grpclog.LoggerV2

	// idleMu is used to guarantee mutual exclusion in two scenarios:
	// - Opposing intentions:
	//   - a: Idle timeout has fired and handleIdleTimeout() is trying to put
	//     the channel in idle mode because the channel has been inactive.
	//   - b: At the same time an RPC is made on the channel, and OnCallBegin()
	//     is trying to prevent the channel from going idle.
	// - Competing intentions:
	//   - The channel is in idle mode and there are multiple RPCs starting at
	//     the same time, all trying to move the channel out of idle. Only one
	//     of them should succeed in doing so, while the other RPCs should
	//     piggyback on the first one and be successfully handled.
	idleMu       sync.RWMutex
	actuallyIdle bool
	timer        *time.Timer
}

// ManagerOptions is a collection of options used by
// NewManager.
type ManagerOptions struct {
	Enforcer Enforcer
	Timeout  time.Duration
	Logger   grpclog.LoggerV2
}

// NewManager creates a new idleness manager implementation for the
// given idle timeout.
func NewManager(opts ManagerOptions) Manager {
	if opts.Timeout == 0 {
		return noopManager{}
	}

	m := &manager{
		enforcer: opts.Enforcer,
		timeout:  int64(opts.Timeout),
		logger:   opts.Logger,
	}
	m.timer = timeAfterFunc(opts.Timeout, m.handleIdleTimeout)
	return m
}

// resetIdleTimer resets the idle timer to the given duration. This method
// should only be called from the timer callback.
func (m *manager) resetIdleTimer(d time.Duration) {
	m.idleMu.Lock()
	defer m.idleMu.Unlock()

	if m.timer == nil {
		// Only close sets timer to nil. We are done.
		return
	}

	// It is safe to ignore the return value from Reset() because this method is
	// only ever called from the timer callback, which means the timer has
	// already fired.
	m.timer.Reset(d)
}

// handleIdleTimeout is the timer callback that is invoked upon expiry of the
// configured idle timeout. The channel is considered inactive if there are no
// ongoing calls and no RPC activity since the last time the timer fired.
func (m *manager) handleIdleTimeout() {
	if m.isClosed() {
		return
	}

	if atomic.LoadInt32(&m.activeCallsCount) > 0 {
		m.resetIdleTimer(time.Duration(m.timeout))
		return
	}

	// There has been activity on the channel since we last got here. Reset the
	// timer and return.
	if atomic.LoadInt32(&m.activeSinceLastTimerCheck) == 1 {
		// Set the timer to fire after a duration of idle timeout, calculated
		// from the time the most recent RPC completed.
		atomic.StoreInt32(&m.activeSinceLastTimerCheck, 0)
		m.resetIdleTimer(time.Duration(atomic.LoadInt64(&m.lastCallEndTime) + m.timeout - time.Now().UnixNano()))
		return
	}

	// This CAS operation is extremely likely to succeed given that there has
	// been no activity since the last time we were here.  Setting the
	// activeCallsCount to -math.MaxInt32 indicates to OnCallBegin() that the
	// channel is either in idle mode or is trying to get there.
	if !atomic.CompareAndSwapInt32(&m.activeCallsCount, 0, -math.MaxInt32) {
		// This CAS operation can fail if an RPC started after we checked for
		// activity at the top of this method, or one was ongoing from before
		// the last time we were here. In both case, reset the timer and return.
		m.resetIdleTimer(time.Duration(m.timeout))
		return
	}

	// Now that we've set the active calls count to -math.MaxInt32, it's time to
	// actually move to idle mode.
	if m.tryEnterIdleMode() {
		// Successfully entered idle mode. No timer needed until we exit idle.
		return
	}

	// Failed to enter idle mode due to a concurrent RPC that kept the channel
	// active, or because of an error from the channel. Undo the attempt to
	// enter idle, and reset the timer to try again later.
	atomic.AddInt32(&m.activeCallsCount, math.MaxInt32)
	m.resetIdleTimer(time.Duration(m.timeout))
}

// tryEnterIdleMode instructs the channel to enter idle mode. But before
// that, it performs a last minute check to ensure that no new RPC has come in,
// making the channel active.
//
// Return value indicates whether or not the channel moved to idle mode.
//
// Holds idleMu which ensures mutual exclusion with exitIdleMode.
func (m *manager) tryEnterIdleMode() bool {
	m.idleMu.Lock()
	defer m.idleMu.Unlock()

	if atomic.LoadInt32(&m.activeCallsCount) != -math.MaxInt32 {
		// We raced and lost to a new RPC. Very rare, but stop entering idle.
		return false
	}
	if atomic.LoadInt32(&m.activeSinceLastTimerCheck) == 1 {
		// An very short RPC could have come in (and also finished) after we
		// checked for calls count and activity in handleIdleTimeout(), but
		// before the CAS operation. So, we need to check for activity again.
		return false
	}

	// No new RPCs have come in since we last set the active calls count value
	// -math.MaxInt32 in the timer callback. And since we have the lock, it is
	// safe to enter idle mode now.
	if err := m.enforcer.EnterIdleMode(); err != nil {
		m.logger.Errorf("Failed to enter idle mode: %v", err)
		return false
	}

	// Successfully entered idle mode.
	m.actuallyIdle = true
	return true
}

// OnCallBegin is invoked at the start of every RPC.
func (m *manager) OnCallBegin() error {
	if m.isClosed() {
		return nil
	}

	if atomic.AddInt32(&m.activeCallsCount, 1) > 0 {
		// Channel is not idle now. Set the activity bit and allow the call.
		atomic.StoreInt32(&m.activeSinceLastTimerCheck, 1)
		return nil
	}

	// Channel is either in idle mode or is in the process of moving to idle
	// mode. Attempt to exit idle mode to allow this RPC.
	if err := m.exitIdleMode(); err != nil {
		// Undo the increment to calls count, and return an error causing the
		// RPC to fail.
		atomic.AddInt32(&m.activeCallsCount, -1)
		return err
	}

	atomic.StoreInt32(&m.activeSinceLastTimerCheck, 1)
	return nil
}

// exitIdleMode instructs the channel to exit idle mode.
//
// Holds idleMu which ensures mutual exclusion with tryEnterIdleMode.
func (m *manager) exitIdleMode() error {
	m.idleMu.Lock()
	defer m.idleMu.Unlock()

	if !m.actuallyIdle {
		// This can happen in two scenarios:
		// - handleIdleTimeout() set the calls count to -math.MaxInt32 and called
		//   tryEnterIdleMode(). But before the latter could grab the lock, an RPC
		//   came in and OnCallBegin() noticed that the calls count is negative.
		// - Channel is in idle mode, and multiple new RPCs come in at the same
		//   time, all of them notice a negative calls count in OnCallBegin and get
		//   here. The first one to get the lock would got the channel to exit idle.
		//
		// Either way, nothing to do here.
		return nil
	}

	if err := m.enforcer.ExitIdleMode(); err != nil {
		return fmt.Errorf("channel failed to exit idle mode: %v", err)
	}

	// Undo the idle entry process. This also respects any new RPC attempts.
	atomic.AddInt32(&m.activeCallsCount, math.MaxInt32)
	m.actuallyIdle = false

	// Start a new timer to fire after the configured idle timeout.
	m.timer = timeAfterFunc(time.Duration(m.timeout), m.handleIdleTimeout)
	return nil
}

// OnCallEnd is invoked at the end of every RPC.
func (m *manager) OnCallEnd() {
	if m.isClosed() {
		return
	}

	// Record the time at which the most recent call finished.
	atomic.StoreInt64(&m.lastCallEndTime, time.Now().UnixNano())

	// Decrement the active calls count. This count can temporarily go negative
	// when the timer callback is in the process of moving the channel to idle
	// mode, but one or more RPCs come in and complete before the timer callback
	// can get done with the process of moving to idle mode.
	atomic.AddInt32(&m.activeCallsCount, -1)
}

func (m *manager) isClosed() bool {
	return atomic.LoadInt32(&m.closed) == 1
}

func (m *manager) Close() {
	atomic.StoreInt32(&m.closed, 1)

	m.idleMu.Lock()
	m.timer.Stop()
	m.timer = nil
	m.idleMu.Unlock()
}
