package state

import (
	"sync"
	"time"
)

// Delay is used to mark certain locks as unacquirable. When a lock is
// forcefully released (failing health check, destroyed session, etc.), it is
// subject to the LockDelay impossed by the session. This prevents another
// session from acquiring the lock for some period of time as a protection
// against split-brains. This is inspired by the lock-delay in Chubby. Because
// this relies on wall-time, we cannot assume all peers perceive time as flowing
// uniformly. This means KVSLock MUST ignore lockDelay, since the lockDelay may
// have expired on the leader, but not on the follower. Rejecting the lock could
// result in inconsistencies in the FSMs due to the rate time progresses. Instead,
// only the opinion of the leader is respected, and the Raft log is never
// questioned.
type Delay struct {
	// delay has the set of active delay expiration times, organized by key.
	delay map[string]time.Time

	// lock protects the delay map.
	lock sync.RWMutex
}

// NewDelay returns a new delay manager.
func NewDelay() *Delay {
	return &Delay{delay: make(map[string]time.Time)}
}

// GetExpiration returns the expiration time of a key lock delay. This must be
// checked on the leader node, and not in KVSLock due to the variability of
// clocks.
func (d *Delay) GetExpiration(key string) time.Time {
	d.lock.RLock()
	expires := d.delay[key]
	d.lock.RUnlock()
	return expires
}

// SetExpiration sets the expiration time for the lock delay to the given
// delay from the given now time.
func (d *Delay) SetExpiration(key string, now time.Time, delay time.Duration) {
	d.lock.Lock()
	defer d.lock.Unlock()

	d.delay[key] = now.Add(delay)
	time.AfterFunc(delay, func() {
		d.lock.Lock()
		delete(d.delay, key)
		d.lock.Unlock()
	})
}
