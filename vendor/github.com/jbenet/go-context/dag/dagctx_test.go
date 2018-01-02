package ctxext

import (
	"math/rand"
	"testing"
	"time"

	context "golang.org/x/net/context"
)

func TestWithParentsSingle(t *testing.T) {
	ctx1, cancel := context.WithCancel(context.Background())
	ctx2 := WithParents(ctx1)

	select {
	case <-ctx2.Done():
		t.Fatal("ended too early")
	case <-time.After(time.Millisecond):
	}

	cancel()

	select {
	case <-ctx2.Done():
	case <-time.After(time.Millisecond):
		t.Error("should've cancelled it")
	}

	if ctx2.Err() != ctx1.Err() {
		t.Error("errors should match")
	}
}

func TestWithParentsDeadline(t *testing.T) {
	ctx1, _ := context.WithCancel(context.Background())
	ctx2, _ := context.WithTimeout(context.Background(), time.Second)
	ctx3, _ := context.WithTimeout(context.Background(), time.Second*2)

	ctx := WithParents(ctx1)
	d, ok := ctx.Deadline()
	if ok {
		t.Error("ctx should have no deadline")
	}

	ctx = WithParents(ctx1, ctx2, ctx3)
	d, ok = ctx.Deadline()
	d2, ok2 := ctx2.Deadline()
	if !ok {
		t.Error("ctx should have deadline")
	} else if !ok2 {
		t.Error("ctx2 should have deadline")
	} else if !d.Equal(d2) {
		t.Error("ctx should have ctx2 deadline")
	}
}

func SubtestWithParentsMany(t *testing.T, n int) {

	ctxs := make([]context.Context, n)
	cancels := make([]context.CancelFunc, n)
	for i := 0; i < n; i++ {
		if i == 0 { // first must be new.
			ctxs[i], cancels[i] = context.WithCancel(context.Background())
			continue
		}

		r := rand.Intn(i) // select a previous context
		switch rand.Intn(6) {
		case 0: // same as old
			ctxs[i], cancels[i] = ctxs[r], cancels[r]
		case 1: // derive from another
			ctxs[i], cancels[i] = context.WithCancel(ctxs[r])
		case 2: // deadline
			t := (time.Second) * time.Duration(r+2) // +2 so we dont run into 0 or timing bugs
			ctxs[i], cancels[i] = context.WithTimeout(ctxs[r], t)
		default: // new context
			ctxs[i], cancels[i] = context.WithCancel(context.Background())
		}
	}

	ctx := WithParents(ctxs...)

	// test deadline is earliest.
	d1 := earliestDeadline(ctxs)
	d2, ok := ctx.Deadline()
	switch {
	case d1 == nil && ok:
		t.Error("nil, should not have deadline")
	case d1 != nil && !ok:
		t.Error("not nil, should have deadline")
	case d1 != nil && ok && !d1.Equal(d2):
		t.Error("should find same deadline")
	}
	if ok {
		t.Logf("deadline - now: %s", d2.Sub(time.Now()))
	}

	select {
	case <-ctx.Done():
		t.Fatal("ended too early")
	case <-time.After(time.Millisecond):
	}

	// cancel just one
	r := rand.Intn(len(cancels))
	cancels[r]()

	select {
	case <-ctx.Done():
	case <-time.After(time.Millisecond):
		t.Error("any should've cancelled it")
	}

	if ctx.Err() != ctxs[r].Err() {
		t.Error("errors should match")
	}
}

func TestWithParentsMany(t *testing.T) {
	n := 100
	for i := 1; i < n; i++ {
		SubtestWithParentsMany(t, i)
	}
}
