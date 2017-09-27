package flowcontrol_test

import (
	"context"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/util/flowcontrol"
)

func TestSleepContext(t *testing.T) {
	clock := clock.NewFakeClock(time.Now())
	ctx := context.Background()

	ch := make(chan struct{})
	var err error
	go func() {
		defer close(ch)
		err = flowcontrol.SleepContext(ctx, clock, time.Minute)
	}()

	time.Sleep(time.Second / 20)
	clock.Step(time.Hour * 2)

	<-ch

	if err != nil {
		t.Error("unexpected error")
	}

	if clock.HasWaiters() {
		t.Error("unexpected waiters")
	}
}

func TestSleepContext_preempted(t *testing.T) {
	clock := clock.NewFakeClock(time.Now())
	ctx, cancel := context.WithCancel(context.Background())

	ch := make(chan struct{})
	var err error
	go func() {
		defer close(ch)
		err = flowcontrol.SleepContext(ctx, clock, time.Minute)
	}()

	cancel()

	<-ch

	if err != ctx.Err() {
		t.Error("expected error not returned")
	}
}
