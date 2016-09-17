package memberlist

import (
	"sync"
	"time"

	"github.com/armon/go-metrics"
)

// awareness manages a simple metric for tracking the estimated health of the
// local node. Health is primary the node's ability to respond in the soft
// real-time manner required for correct health checking of other nodes in the
// cluster.
type awareness struct {
	sync.RWMutex

	// max is the upper threshold for the timeout scale (the score will be
	// constrained to be from 0 <= score < max).
	max int

	// score is the current awareness score. Lower values are healthier and
	// zero is the minimum value.
	score int
}

// newAwareness returns a new awareness object.
func newAwareness(max int) *awareness {
	return &awareness{
		max:   max,
		score: 0,
	}
}

// ApplyDelta takes the given delta and applies it to the score in a thread-safe
// manner. It also enforces a floor of zero and a max of max, so deltas may not
// change the overall score if it's railed at one of the extremes.
func (a *awareness) ApplyDelta(delta int) {
	a.Lock()
	initial := a.score
	a.score += delta
	if a.score < 0 {
		a.score = 0
	} else if a.score > (a.max - 1) {
		a.score = (a.max - 1)
	}
	final := a.score
	a.Unlock()

	if initial != final {
		metrics.SetGauge([]string{"memberlist", "health", "score"}, float32(final))
	}
}

// GetHealthScore returns the raw health score.
func (a *awareness) GetHealthScore() int {
	a.RLock()
	score := a.score
	a.RUnlock()
	return score
}

// ScaleTimeout takes the given duration and scales it based on the current
// score. Less healthyness will lead to longer timeouts.
func (a *awareness) ScaleTimeout(timeout time.Duration) time.Duration {
	a.RLock()
	score := a.score
	a.RUnlock()
	return timeout * (time.Duration(score) + 1)
}
