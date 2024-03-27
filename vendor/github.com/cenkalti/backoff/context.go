package backoff

import (
	"context"
	"time"
)

// BackOffContext is a backoff policy that stops retrying after the context
// is canceled.
type BackOffContext interface {
	BackOff
	Context() context.Context
}

type backOffContext struct {
	BackOff
	ctx context.Context
}

// WithContext returns a BackOffContext with context ctx
//
// ctx must not be nil
func WithContext(b BackOff, ctx context.Context) BackOffContext {
	if ctx == nil {
		panic("nil context")
	}

	if b, ok := b.(*backOffContext); ok {
		return &backOffContext{
			BackOff: b.BackOff,
			ctx:     ctx,
		}
	}

	return &backOffContext{
		BackOff: b,
		ctx:     ctx,
	}
}

func ensureContext(b BackOff) BackOffContext {
	if cb, ok := b.(BackOffContext); ok {
		return cb
	}
	return WithContext(b, context.Background())
}

func (b *backOffContext) Context() context.Context {
	return b.ctx
}

func (b *backOffContext) NextBackOff() time.Duration {
	select {
	case <-b.ctx.Done():
		return Stop
	default:
	}
	next := b.BackOff.NextBackOff()
	if deadline, ok := b.ctx.Deadline(); ok && deadline.Sub(time.Now()) < next {
		return Stop
	}
	return next
}
