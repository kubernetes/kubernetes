package ctxext

import (
	"sync"
	"time"

	context "golang.org/x/net/context"
)

// WithParents returns a Context that listens to all given
// parents. It effectively transforms the Context Tree into
// a Directed Acyclic Graph. This is useful when a context
// may be cancelled for more than one reason. For example,
// consider a database with the following Get function:
//
//   func (db *DB) Get(ctx context.Context, ...) {}
//
// DB.Get may have to stop for two different contexts:
//  * the caller's context (caller might cancel)
//  * the database's context (might be shut down mid-request)
//
// WithParents saves the day by allowing us to "merge" contexts
// and continue on our merry contextual way:
//
//   ctx = ctxext.WithParents(ctx, db.ctx)
//
// Passing related (mutually derived) contexts to WithParents is
// actually ok. The child is cancelled when any of its parents is
// cancelled, so if any of its parents are also related, the cancel
// propagation will reach the child via the shortest path.
func WithParents(ctxts ...context.Context) context.Context {
	if len(ctxts) < 1 {
		panic("no contexts provided")
	}

	ctx := &errCtx{
		done: make(chan struct{}),
		dead: earliestDeadline(ctxts),
	}

	// listen to all contexts and use the first.
	for _, c2 := range ctxts {
		go func(pctx context.Context) {
			select {
			case <-ctx.Done(): // cancelled by another parent
				return
			case <-pctx.Done(): // this parent cancelled
				// race: two parents may have cancelled at the same time.
				// break tie with mutex (inside c.cancel)
				ctx.cancel(pctx.Err())
			}
		}(c2)
	}

	return ctx
}

func earliestDeadline(ctxts []context.Context) *time.Time {
	var d1 *time.Time
	for _, c := range ctxts {
		if c == nil {
			panic("given nil context.Context")
		}

		// use earliest deadline.
		d2, ok := c.Deadline()
		if !ok {
			continue
		}

		if d1 == nil || (*d1).After(d2) {
			d1 = &d2
		}
	}
	return d1
}

type errCtx struct {
	dead *time.Time
	done chan struct{}
	err  error
	mu   sync.RWMutex
}

func (c *errCtx) cancel(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	select {
	case <-c.Done():
		return
	default:
	}

	c.err = err
	close(c.done) // signal done to all
}

func (c *errCtx) Done() <-chan struct{} {
	return c.done
}

func (c *errCtx) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

func (c *errCtx) Value(key interface{}) interface{} {
	return nil
}

func (c *errCtx) Deadline() (deadline time.Time, ok bool) {
	if c.dead == nil {
		return
	}

	return *c.dead, true
}
