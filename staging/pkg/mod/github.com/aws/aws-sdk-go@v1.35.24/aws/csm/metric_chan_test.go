package csm

import (
	"testing"
)

func TestMetricChanPush(t *testing.T) {
	ch := newMetricChan(5)
	defer close(ch.ch)

	pushed := ch.Push(metric{})
	if !pushed {
		t.Errorf("expected metrics to be pushed")
	}

	if e, a := 1, len(ch.ch); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}

func TestMetricChanPauseContinue(t *testing.T) {
	ch := newMetricChan(5)
	defer close(ch.ch)
	ch.Pause()

	if !ch.IsPaused() {
		t.Errorf("expected to be paused, but did not pause properly")
	}

	ch.Continue()
	if ch.IsPaused() {
		t.Errorf("expected to be not paused, but did not continue properly")
	}

	pushed := ch.Push(metric{})
	if !pushed {
		t.Errorf("expected metrics to be pushed")
	}

	if e, a := 1, len(ch.ch); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}

func TestMetricChanPushWhenPaused(t *testing.T) {
	ch := newMetricChan(5)
	defer close(ch.ch)
	ch.Pause()

	pushed := ch.Push(metric{})
	if pushed {
		t.Errorf("expected metrics to not be pushed")
	}

	if e, a := 0, len(ch.ch); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}

func TestMetricChanNonBlocking(t *testing.T) {
	ch := newMetricChan(0)
	defer close(ch.ch)

	pushed := ch.Push(metric{})
	if pushed {
		t.Errorf("expected metrics to be not pushed")
	}

	if e, a := 0, len(ch.ch); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}
