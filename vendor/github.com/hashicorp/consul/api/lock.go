package api

import (
	"fmt"
	"sync"
	"time"
)

const (
	// DefaultLockSessionName is the Session Name we assign if none is provided
	DefaultLockSessionName = "Consul API Lock"

	// DefaultLockSessionTTL is the default session TTL if no Session is provided
	// when creating a new Lock. This is used because we do not have another
	// other check to depend upon.
	DefaultLockSessionTTL = "15s"

	// DefaultLockWaitTime is how long we block for at a time to check if lock
	// acquisition is possible. This affects the minimum time it takes to cancel
	// a Lock acquisition.
	DefaultLockWaitTime = 15 * time.Second

	// DefaultLockRetryTime is how long we wait after a failed lock acquisition
	// before attempting to do the lock again. This is so that once a lock-delay
	// is in effect, we do not hot loop retrying the acquisition.
	DefaultLockRetryTime = 5 * time.Second

	// DefaultMonitorRetryTime is how long we wait after a failed monitor check
	// of a lock (500 response code). This allows the monitor to ride out brief
	// periods of unavailability, subject to the MonitorRetries setting in the
	// lock options which is by default set to 0, disabling this feature. This
	// affects locks and semaphores.
	DefaultMonitorRetryTime = 2 * time.Second

	// LockFlagValue is a magic flag we set to indicate a key
	// is being used for a lock. It is used to detect a potential
	// conflict with a semaphore.
	LockFlagValue = 0x2ddccbc058a50c18
)

var (
	// ErrLockHeld is returned if we attempt to double lock
	ErrLockHeld = fmt.Errorf("Lock already held")

	// ErrLockNotHeld is returned if we attempt to unlock a lock
	// that we do not hold.
	ErrLockNotHeld = fmt.Errorf("Lock not held")

	// ErrLockInUse is returned if we attempt to destroy a lock
	// that is in use.
	ErrLockInUse = fmt.Errorf("Lock in use")

	// ErrLockConflict is returned if the flags on a key
	// used for a lock do not match expectation
	ErrLockConflict = fmt.Errorf("Existing key does not match lock use")
)

// Lock is used to implement client-side leader election. It is follows the
// algorithm as described here: https://www.consul.io/docs/guides/leader-election.html.
type Lock struct {
	c    *Client
	opts *LockOptions

	isHeld       bool
	sessionRenew chan struct{}
	lockSession  string
	l            sync.Mutex
}

// LockOptions is used to parameterize the Lock behavior.
type LockOptions struct {
	Key              string        // Must be set and have write permissions
	Value            []byte        // Optional, value to associate with the lock
	Session          string        // Optional, created if not specified
	SessionName      string        // Optional, defaults to DefaultLockSessionName
	SessionTTL       string        // Optional, defaults to DefaultLockSessionTTL
	MonitorRetries   int           // Optional, defaults to 0 which means no retries
	MonitorRetryTime time.Duration // Optional, defaults to DefaultMonitorRetryTime
	LockWaitTime     time.Duration // Optional, defaults to DefaultLockWaitTime
	LockTryOnce      bool          // Optional, defaults to false which means try forever
}

// LockKey returns a handle to a lock struct which can be used
// to acquire and release the mutex. The key used must have
// write permissions.
func (c *Client) LockKey(key string) (*Lock, error) {
	opts := &LockOptions{
		Key: key,
	}
	return c.LockOpts(opts)
}

// LockOpts returns a handle to a lock struct which can be used
// to acquire and release the mutex. The key used must have
// write permissions.
func (c *Client) LockOpts(opts *LockOptions) (*Lock, error) {
	if opts.Key == "" {
		return nil, fmt.Errorf("missing key")
	}
	if opts.SessionName == "" {
		opts.SessionName = DefaultLockSessionName
	}
	if opts.SessionTTL == "" {
		opts.SessionTTL = DefaultLockSessionTTL
	} else {
		if _, err := time.ParseDuration(opts.SessionTTL); err != nil {
			return nil, fmt.Errorf("invalid SessionTTL: %v", err)
		}
	}
	if opts.MonitorRetryTime == 0 {
		opts.MonitorRetryTime = DefaultMonitorRetryTime
	}
	if opts.LockWaitTime == 0 {
		opts.LockWaitTime = DefaultLockWaitTime
	}
	l := &Lock{
		c:    c,
		opts: opts,
	}
	return l, nil
}

// Lock attempts to acquire the lock and blocks while doing so.
// Providing a non-nil stopCh can be used to abort the lock attempt.
// Returns a channel that is closed if our lock is lost or an error.
// This channel could be closed at any time due to session invalidation,
// communication errors, operator intervention, etc. It is NOT safe to
// assume that the lock is held until Unlock() unless the Session is specifically
// created without any associated health checks. By default Consul sessions
// prefer liveness over safety and an application must be able to handle
// the lock being lost.
func (l *Lock) Lock(stopCh <-chan struct{}) (<-chan struct{}, error) {
	// Hold the lock as we try to acquire
	l.l.Lock()
	defer l.l.Unlock()

	// Check if we already hold the lock
	if l.isHeld {
		return nil, ErrLockHeld
	}

	// Check if we need to create a session first
	l.lockSession = l.opts.Session
	if l.lockSession == "" {
		if s, err := l.createSession(); err != nil {
			return nil, fmt.Errorf("failed to create session: %v", err)
		} else {
			l.sessionRenew = make(chan struct{})
			l.lockSession = s
			session := l.c.Session()
			go session.RenewPeriodic(l.opts.SessionTTL, s, nil, l.sessionRenew)

			// If we fail to acquire the lock, cleanup the session
			defer func() {
				if !l.isHeld {
					close(l.sessionRenew)
					l.sessionRenew = nil
				}
			}()
		}
	}

	// Setup the query options
	kv := l.c.KV()
	qOpts := &QueryOptions{
		WaitTime: l.opts.LockWaitTime,
	}

	start := time.Now()
	attempts := 0
WAIT:
	// Check if we should quit
	select {
	case <-stopCh:
		return nil, nil
	default:
	}

	// Handle the one-shot mode.
	if l.opts.LockTryOnce && attempts > 0 {
		elapsed := time.Now().Sub(start)
		if elapsed > qOpts.WaitTime {
			return nil, nil
		}

		qOpts.WaitTime -= elapsed
	}
	attempts++

	// Look for an existing lock, blocking until not taken
	pair, meta, err := kv.Get(l.opts.Key, qOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to read lock: %v", err)
	}
	if pair != nil && pair.Flags != LockFlagValue {
		return nil, ErrLockConflict
	}
	locked := false
	if pair != nil && pair.Session == l.lockSession {
		goto HELD
	}
	if pair != nil && pair.Session != "" {
		qOpts.WaitIndex = meta.LastIndex
		goto WAIT
	}

	// Try to acquire the lock
	pair = l.lockEntry(l.lockSession)
	locked, _, err = kv.Acquire(pair, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire lock: %v", err)
	}

	// Handle the case of not getting the lock
	if !locked {
		// Determine why the lock failed
		qOpts.WaitIndex = 0
		pair, meta, err = kv.Get(l.opts.Key, qOpts)
		if pair != nil && pair.Session != "" {
			//If the session is not null, this means that a wait can safely happen
			//using a long poll
			qOpts.WaitIndex = meta.LastIndex
			goto WAIT
		} else {
			// If the session is empty and the lock failed to acquire, then it means
			// a lock-delay is in effect and a timed wait must be used
			select {
			case <-time.After(DefaultLockRetryTime):
				goto WAIT
			case <-stopCh:
				return nil, nil
			}
		}
	}

HELD:
	// Watch to ensure we maintain leadership
	leaderCh := make(chan struct{})
	go l.monitorLock(l.lockSession, leaderCh)

	// Set that we own the lock
	l.isHeld = true

	// Locked! All done
	return leaderCh, nil
}

// Unlock released the lock. It is an error to call this
// if the lock is not currently held.
func (l *Lock) Unlock() error {
	// Hold the lock as we try to release
	l.l.Lock()
	defer l.l.Unlock()

	// Ensure the lock is actually held
	if !l.isHeld {
		return ErrLockNotHeld
	}

	// Set that we no longer own the lock
	l.isHeld = false

	// Stop the session renew
	if l.sessionRenew != nil {
		defer func() {
			close(l.sessionRenew)
			l.sessionRenew = nil
		}()
	}

	// Get the lock entry, and clear the lock session
	lockEnt := l.lockEntry(l.lockSession)
	l.lockSession = ""

	// Release the lock explicitly
	kv := l.c.KV()
	_, _, err := kv.Release(lockEnt, nil)
	if err != nil {
		return fmt.Errorf("failed to release lock: %v", err)
	}
	return nil
}

// Destroy is used to cleanup the lock entry. It is not necessary
// to invoke. It will fail if the lock is in use.
func (l *Lock) Destroy() error {
	// Hold the lock as we try to release
	l.l.Lock()
	defer l.l.Unlock()

	// Check if we already hold the lock
	if l.isHeld {
		return ErrLockHeld
	}

	// Look for an existing lock
	kv := l.c.KV()
	pair, _, err := kv.Get(l.opts.Key, nil)
	if err != nil {
		return fmt.Errorf("failed to read lock: %v", err)
	}

	// Nothing to do if the lock does not exist
	if pair == nil {
		return nil
	}

	// Check for possible flag conflict
	if pair.Flags != LockFlagValue {
		return ErrLockConflict
	}

	// Check if it is in use
	if pair.Session != "" {
		return ErrLockInUse
	}

	// Attempt the delete
	didRemove, _, err := kv.DeleteCAS(pair, nil)
	if err != nil {
		return fmt.Errorf("failed to remove lock: %v", err)
	}
	if !didRemove {
		return ErrLockInUse
	}
	return nil
}

// createSession is used to create a new managed session
func (l *Lock) createSession() (string, error) {
	session := l.c.Session()
	se := &SessionEntry{
		Name: l.opts.SessionName,
		TTL:  l.opts.SessionTTL,
	}
	id, _, err := session.Create(se, nil)
	if err != nil {
		return "", err
	}
	return id, nil
}

// lockEntry returns a formatted KVPair for the lock
func (l *Lock) lockEntry(session string) *KVPair {
	return &KVPair{
		Key:     l.opts.Key,
		Value:   l.opts.Value,
		Session: session,
		Flags:   LockFlagValue,
	}
}

// monitorLock is a long running routine to monitor a lock ownership
// It closes the stopCh if we lose our leadership.
func (l *Lock) monitorLock(session string, stopCh chan struct{}) {
	defer close(stopCh)
	kv := l.c.KV()
	opts := &QueryOptions{RequireConsistent: true}
WAIT:
	retries := l.opts.MonitorRetries
RETRY:
	pair, meta, err := kv.Get(l.opts.Key, opts)
	if err != nil {
		// If configured we can try to ride out a brief Consul unavailability
		// by doing retries. Note that we have to attempt the retry in a non-
		// blocking fashion so that we have a clean place to reset the retry
		// counter if service is restored.
		if retries > 0 && IsServerError(err) {
			time.Sleep(l.opts.MonitorRetryTime)
			retries--
			opts.WaitIndex = 0
			goto RETRY
		}
		return
	}
	if pair != nil && pair.Session == session {
		opts.WaitIndex = meta.LastIndex
		goto WAIT
	}
}
