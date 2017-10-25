package flowcontrol_test

import (
	"context"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/util/flowcontrol"
)

func TestSleepContext(t *testing.T) {
	duration := time.Minute
	start := time.Now()
	clock := clock.NewFakeClock(start)
	clock.AdvanceOnAfter = true
	ctx := context.Background()

	err := flowcontrol.SleepContext(ctx, clock, duration)

	if err != nil {
		t.Error("unexpected error")
	}

	if clock.HasWaiters() {
		t.Error("unexpected waiters")
	}

	if actual := clock.Since(start); actual != duration {
		t.Errorf("unexpected sleep duration %v != %v", actual, duration)
	}
}

func TestSleepContext_preempted(t *testing.T) {
	duration := time.Second / 10
	clock := clock.RealClock{}
	start := clock.Now()
	ctx, cancel := context.WithCancel(context.Background())

	ch := make(chan struct{})
	var err error
	go func() {
		defer close(ch)
		err = flowcontrol.SleepContext(ctx, clock, duration)
	}()

	cancel()

	<-ch

	if err != ctx.Err() {
		t.Error("expected error not returned")
	}

	if actual := clock.Since(start); actual >= duration {
		t.Errorf("unexpected sleep duration %v >= %v", actual, duration)
	}
}
