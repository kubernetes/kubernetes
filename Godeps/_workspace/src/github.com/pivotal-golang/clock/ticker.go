package clock

import "time"

type Ticker interface {
	C() <-chan time.Time
	Stop()
}

type realTicker struct {
	t *time.Ticker
}

func (t *realTicker) C() <-chan time.Time {
	return t.t.C
}

func (t *realTicker) Stop() {
	t.t.Stop()
}
