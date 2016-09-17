package consul

import (
	"fmt"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/structs"
)

// initializeSessionTimers is used when a leader is newly elected to create
// a new map to track session expiration and to reset all the timers from
// the previously known set of timers.
func (s *Server) initializeSessionTimers() error {
	// Scan all sessions and reset their timer
	state := s.fsm.State()
	_, sessions, err := state.SessionList()
	if err != nil {
		return err
	}
	for _, session := range sessions {
		if err := s.resetSessionTimer(session.ID, session); err != nil {
			return err
		}
	}
	return nil
}

// resetSessionTimer is used to renew the TTL of a session.
// This can be used for new sessions and existing ones. A session
// will be faulted in if not given.
func (s *Server) resetSessionTimer(id string, session *structs.Session) error {
	// Fault the session in if not given
	if session == nil {
		state := s.fsm.State()
		_, s, err := state.SessionGet(id)
		if err != nil {
			return err
		}
		if s == nil {
			return fmt.Errorf("Session '%s' not found", id)
		}
		session = s
	}

	// Bail if the session has no TTL, fast-path some common inputs
	switch session.TTL {
	case "", "0", "0s", "0m", "0h":
		return nil
	}

	// Parse the TTL, and skip if zero time
	ttl, err := time.ParseDuration(session.TTL)
	if err != nil {
		return fmt.Errorf("Invalid Session TTL '%s': %v", session.TTL, err)
	}
	if ttl == 0 {
		return nil
	}

	// Reset the session timer
	s.sessionTimersLock.Lock()
	defer s.sessionTimersLock.Unlock()
	s.resetSessionTimerLocked(id, ttl)
	return nil
}

// resetSessionTimerLocked is used to reset a session timer
// assuming the sessionTimerLock is already held
func (s *Server) resetSessionTimerLocked(id string, ttl time.Duration) {
	// Ensure a timer map exists
	if s.sessionTimers == nil {
		s.sessionTimers = make(map[string]*time.Timer)
	}

	// Adjust the given TTL by the TTL multiplier. This is done
	// to give a client a grace period and to compensate for network
	// and processing delays. The contract is that a session is not expired
	// before the TTL, but there is no explicit promise about the upper
	// bound so this is allowable.
	ttl = ttl * structs.SessionTTLMultiplier

	// Renew the session timer if it exists
	if timer, ok := s.sessionTimers[id]; ok {
		timer.Reset(ttl)
		return
	}

	// Create a new timer to track expiration of thi ssession
	timer := time.AfterFunc(ttl, func() {
		s.invalidateSession(id)
	})
	s.sessionTimers[id] = timer
}

// invalidateSession is invoked when a session TTL is reached and we
// need to invalidate the session.
func (s *Server) invalidateSession(id string) {
	defer metrics.MeasureSince([]string{"consul", "session_ttl", "invalidate"}, time.Now())
	// Clear the session timer
	s.sessionTimersLock.Lock()
	delete(s.sessionTimers, id)
	s.sessionTimersLock.Unlock()

	// Create a session destroy request
	args := structs.SessionRequest{
		Datacenter: s.config.Datacenter,
		Op:         structs.SessionDestroy,
		Session: structs.Session{
			ID: id,
		},
	}
	s.logger.Printf("[DEBUG] consul.state: Session %s TTL expired", id)

	// Apply the update to destroy the session
	if _, err := s.raftApply(structs.SessionRequestType, args); err != nil {
		s.logger.Printf("[ERR] consul.session: Invalidation failed: %v", err)
	}
}

// clearSessionTimer is used to clear the session time for
// a single session. This is used when a session is destroyed
// explicitly and no longer needed.
func (s *Server) clearSessionTimer(id string) error {
	s.sessionTimersLock.Lock()
	defer s.sessionTimersLock.Unlock()

	if timer, ok := s.sessionTimers[id]; ok {
		timer.Stop()
		delete(s.sessionTimers, id)
	}
	return nil
}

// clearAllSessionTimers is used when a leader is stepping
// down and we no longer need to track any session timers.
func (s *Server) clearAllSessionTimers() error {
	s.sessionTimersLock.Lock()
	defer s.sessionTimersLock.Unlock()

	for _, t := range s.sessionTimers {
		t.Stop()
	}
	s.sessionTimers = nil
	return nil
}

// sessionStats is a long running routine used to capture
// the number of active sessions being tracked
func (s *Server) sessionStats() {
	for {
		select {
		case <-time.After(5 * time.Second):
			s.sessionTimersLock.Lock()
			num := len(s.sessionTimers)
			s.sessionTimersLock.Unlock()
			metrics.SetGauge([]string{"consul", "session_ttl", "active"}, float32(num))

		case <-s.shutdownCh:
			return
		}
	}
}
