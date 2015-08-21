package key

import (
	"errors"
	"time"

	"github.com/jonboulle/clockwork"

	"github.com/coreos/pkg/timeutil"
)

func NewKeySetSyncer(r ReadableKeySetRepo, w WritableKeySetRepo) *KeySetSyncer {
	return &KeySetSyncer{
		readable: r,
		writable: w,
		clock:    clockwork.NewRealClock(),
	}
}

type KeySetSyncer struct {
	readable ReadableKeySetRepo
	writable WritableKeySetRepo
	clock    clockwork.Clock
}

func (s *KeySetSyncer) Run() chan struct{} {
	stop := make(chan struct{})
	go func() {
		var failing bool
		var next time.Duration
		for {
			exp, err := sync(s.readable, s.writable, s.clock)
			if err != nil || exp == 0 {
				if !failing {
					failing = true
					next = time.Second
				} else {
					next = timeutil.ExpBackoff(next, time.Minute)
				}
				if exp == 0 {
					log.Errorf("Synced to already expired key set, retrying in %v: %v", next, err)

				} else {
					log.Errorf("Failed syncing key set, retrying in %v: %v", next, err)
				}
			} else {
				failing = false
				next = exp / 2
				log.Infof("Synced key set, checking again in %v", next)
			}

			select {
			case <-s.clock.After(next):
				continue
			case <-stop:
				return
			}
		}
	}()

	return stop
}

func Sync(r ReadableKeySetRepo, w WritableKeySetRepo) (time.Duration, error) {
	return sync(r, w, clockwork.NewRealClock())
}

// sync copies the keyset from r to the KeySet at w and returns the duration in which the KeySet will expire.
// If keyset has already expired, returns a zero duration.
func sync(r ReadableKeySetRepo, w WritableKeySetRepo, clock clockwork.Clock) (exp time.Duration, err error) {
	var ks KeySet
	ks, err = r.Get()
	if err != nil {
		return
	}

	if ks == nil {
		err = errors.New("no source KeySet")
		return
	}

	if err = w.Set(ks); err != nil {
		return
	}

	now := clock.Now()
	if ks.ExpiresAt().After(now) {
		exp = ks.ExpiresAt().Sub(now)
	}
	return
}
